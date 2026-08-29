"""
Core OCR extraction engine for Bangla text.

Handles PDF-to-image conversion, OCR processing with EasyOCR,
and coordinates the full extraction pipeline with progress tracking
and multiprocessing support.
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import pypdfium2 as pdfium
from PIL import Image
from tqdm import tqdm

from bangla_ocr.preprocessor import ImagePreprocessor, PreprocessorConfig

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    """Result from OCR processing of a single text region."""
    text: str
    confidence: float
    bbox: list  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]


@dataclass
class PageResult:
    """Result from processing a single PDF page."""
    page_number: int
    text: str
    regions: list[OCRResult] = field(default_factory=list)
    processing_time: float = 0.0
    error: Optional[str] = None


@dataclass
class ExtractionResult:
    """Result from processing an entire PDF document."""
    input_file: str
    total_pages: int
    processed_pages: int
    pages: list[PageResult] = field(default_factory=list)
    total_time: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def full_text(self) -> str:
        """Get all extracted text concatenated."""
        parts = []
        for page in self.pages:
            if page.text.strip():
                parts.append(page.text)
        return "\n\n".join(parts)

    @property
    def success_rate(self) -> float:
        """Percentage of pages processed successfully."""
        if self.total_pages == 0:
            return 0.0
        return (self.processed_pages / self.total_pages) * 100


def _process_single_page(args: tuple) -> PageResult:
    """
    Process a single page — designed to run in a subprocess.

    This is a module-level function (not a method) so it can be
    pickled for multiprocessing.

    Args:
        args: Tuple of (page_number, image_bytes, width, height,
              preprocess_config, confidence_threshold, languages)

    Returns:
        PageResult with extracted text and metadata.
    """
    import easyocr

    (
        page_number,
        image_array,
        preprocess_enabled,
        confidence_threshold,
        languages,
    ) = args

    start_time = time.time()

    try:
        # Initialize reader in this process
        reader = easyocr.Reader(languages, gpu=False, verbose=False)

        # Preprocess if enabled
        if preprocess_enabled:
            preprocessor = PreprocessorConfig.for_scanned_book()
            pil_img = Image.fromarray(image_array)
            processed = preprocessor.process(pil_img)
        else:
            processed = image_array

        # Run OCR
        results = reader.readtext(processed)

        # Parse results
        regions = []
        text_parts = []

        for bbox, text, confidence in results:
            if confidence >= confidence_threshold:
                regions.append(OCRResult(
                    text=text,
                    confidence=confidence,
                    bbox=bbox,
                ))
                text_parts.append(text)

        full_text = "\n".join(text_parts)
        elapsed = time.time() - start_time

        return PageResult(
            page_number=page_number,
            text=full_text,
            regions=regions,
            processing_time=elapsed,
        )

    except Exception as e:
        elapsed = time.time() - start_time
        return PageResult(
            page_number=page_number,
            text="",
            processing_time=elapsed,
            error=str(e),
        )


class BanglaOCRExtractor:
    """
    Main OCR extraction engine for Bangla PDFs.

    Handles PDF loading, page rendering, OCR processing,
    and result aggregation.
    """

    def __init__(
        self,
        languages: Optional[list[str]] = None,
        gpu: bool = False,
        preprocess: bool = True,
        confidence_threshold: float = 0.2,
        dpi: int = 300,
        workers: int = 1,
    ):
        """
        Initialize the extractor.

        Args:
            languages: OCR languages. Defaults to ['bn'] (Bengali).
                       Use ['bn', 'en'] for mixed Bengali-English documents.
            gpu: Use GPU acceleration (requires CUDA).
            preprocess: Apply image preprocessing pipeline.
            confidence_threshold: Minimum confidence to include a text region.
            dpi: DPI for PDF page rendering. Higher = more detail but slower.
            workers: Number of parallel workers. Use 1 for sequential processing,
                     which reuses the OCR reader across pages (much faster).
        """
        self.languages = languages or ["bn"]
        self.gpu = gpu
        self.preprocess = preprocess
        self.confidence_threshold = confidence_threshold
        self.dpi = dpi
        self.workers = workers
        self._reader = None

    def _get_reader(self):
        """Lazy-initialize the EasyOCR reader (downloads models on first use)."""
        if self._reader is None:
            import easyocr
            logger.info(
                "Initializing EasyOCR reader for languages: %s (GPU: %s)",
                self.languages, self.gpu
            )
            print(
                f"🔤 Loading OCR models for: {', '.join(self.languages)}...",
                file=sys.stderr,
            )
            self._reader = easyocr.Reader(
                self.languages,
                gpu=self.gpu,
                verbose=False,
            )
            print("✅ OCR models loaded successfully!", file=sys.stderr)
        return self._reader

    def _render_page(self, pdf_page, page_num: int) -> np.ndarray:
        """
        Render a single PDF page to a numpy image array.

        Args:
            pdf_page: pypdfium2 page object.
            page_num: Page number (for logging).

        Returns:
            numpy array of the rendered page image.
        """
        # Scale factor: 72 DPI is the default PDF resolution
        scale = self.dpi / 72.0
        bitmap = pdf_page.render(scale=scale)
        pil_image = bitmap.to_pil()

        # Apply preprocessing if enabled
        if self.preprocess:
            preprocessor = PreprocessorConfig.for_scanned_book()
            preprocessor.target_dpi = self.dpi
            image_array = preprocessor.process(pil_image)
        else:
            image_array = np.array(pil_image)

        return image_array

    def _process_page_sequential(
        self, image_array: np.ndarray, page_num: int
    ) -> PageResult:
        """Process a single page using the shared reader (sequential mode)."""
        start_time = time.time()

        try:
            reader = self._get_reader()
            results = reader.readtext(image_array)

            regions = []
            text_parts = []

            for bbox, text, confidence in results:
                if confidence >= self.confidence_threshold:
                    regions.append(OCRResult(
                        text=text,
                        confidence=confidence,
                        bbox=[[float(c) for c in point] for point in bbox],
                    ))
                    text_parts.append(text)

            full_text = "\n".join(text_parts)
            elapsed = time.time() - start_time

            return PageResult(
                page_number=page_num,
                text=full_text,
                regions=regions,
                processing_time=elapsed,
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error("Error processing page %d: %s", page_num, e)
            return PageResult(
                page_number=page_num,
                text="",
                processing_time=elapsed,
                error=str(e),
            )

    def extract(
        self,
        pdf_path: str,
        page_range: Optional[tuple[int, int]] = None,
        callback=None,
    ) -> ExtractionResult:
        """
        Extract Bangla text from a PDF file.

        Args:
            pdf_path: Path to the input PDF file.
            page_range: Optional (start, end) page range (1-indexed, inclusive).
                        None = process all pages.
            callback: Optional callback function(page_num, total_pages, page_result)
                      called after each page is processed.

        Returns:
            ExtractionResult with all extracted text and metadata.
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        overall_start = time.time()

        # Open PDF
        logger.info("Opening PDF: %s", pdf_path)
        pdf = pdfium.PdfDocument(pdf_path)
        total_pages = len(pdf)
        logger.info("PDF has %d pages", total_pages)

        # Determine page range
        if page_range:
            start_page = max(1, page_range[0]) - 1  # Convert to 0-indexed
            end_page = min(total_pages, page_range[1])
        else:
            start_page = 0
            end_page = total_pages

        pages_to_process = list(range(start_page, end_page))
        num_pages = len(pages_to_process)

        print(
            f"\n📄 Processing {num_pages} pages from '{os.path.basename(pdf_path)}'...\n",
            file=sys.stderr,
        )

        result = ExtractionResult(
            input_file=pdf_path,
            total_pages=num_pages,
            processed_pages=0,
        )

        # Sequential processing (recommended — reuses OCR reader)
        if self.workers <= 1:
            self._extract_sequential(
                pdf, pages_to_process, result, callback
            )
        else:
            # Parallel processing — each worker loads its own reader
            self._extract_parallel(
                pdf, pages_to_process, result, callback
            )

        pdf.close()

        result.total_time = time.time() - overall_start

        # Summary
        print(
            f"\n{'='*50}\n"
            f"✅ Extraction complete!\n"
            f"   Pages processed: {result.processed_pages}/{result.total_pages}\n"
            f"   Success rate: {result.success_rate:.1f}%\n"
            f"   Total time: {result.total_time:.1f}s\n"
            f"   Avg time/page: {result.total_time/max(result.processed_pages,1):.1f}s\n"
            f"{'='*50}\n",
            file=sys.stderr,
        )

        if result.errors:
            print(
                f"⚠️  {len(result.errors)} page(s) had errors.",
                file=sys.stderr,
            )

        return result

    def _extract_sequential(self, pdf, pages, result, callback):
        """Process pages one at a time, reusing the OCR reader."""
        with tqdm(
            total=len(pages),
            desc="🔍 OCR Progress",
            unit="page",
            bar_format="{l_bar}{bar:30}{r_bar}",
        ) as pbar:
            for i, page_idx in enumerate(pages):
                page = pdf[page_idx]

                # Render page to image
                image_array = self._render_page(page, page_idx + 1)

                # Run OCR
                page_result = self._process_page_sequential(
                    image_array, page_idx + 1
                )

                result.pages.append(page_result)

                if page_result.error:
                    result.errors.append(
                        f"Page {page_idx+1}: {page_result.error}"
                    )
                else:
                    result.processed_pages += 1

                pbar.update(1)
                pbar.set_postfix(
                    page=page_idx + 1,
                    time=f"{page_result.processing_time:.1f}s",
                )

                if callback:
                    callback(page_idx + 1, len(pages), page_result)

    def _extract_parallel(self, pdf, pages, result, callback):
        """Process pages in parallel using multiple workers."""
        # Pre-render all pages to numpy arrays (can't pickle PDF objects)
        print("📸 Rendering pages...", file=sys.stderr)
        rendered_pages = []
        for page_idx in tqdm(pages, desc="Rendering", unit="page"):
            page = pdf[page_idx]
            # Render without preprocessing (will be done in worker)
            scale = self.dpi / 72.0
            bitmap = page.render(scale=scale)
            pil_image = bitmap.to_pil()
            image_array = np.array(pil_image)
            rendered_pages.append((page_idx + 1, image_array))

        # Prepare worker arguments
        worker_args = [
            (
                page_num,
                img_array,
                self.preprocess,
                self.confidence_threshold,
                self.languages,
            )
            for page_num, img_array in rendered_pages
        ]

        # Process in parallel
        print(
            f"\n🚀 Processing with {self.workers} workers...",
            file=sys.stderr,
        )
        page_results = {}

        with ProcessPoolExecutor(max_workers=self.workers) as executor:
            futures = {
                executor.submit(_process_single_page, args): args[0]
                for args in worker_args
            }

            with tqdm(
                total=len(futures),
                desc="🔍 OCR Progress",
                unit="page",
                bar_format="{l_bar}{bar:30}{r_bar}",
            ) as pbar:
                for future in as_completed(futures):
                    page_result = future.result()
                    page_results[page_result.page_number] = page_result

                    if page_result.error:
                        result.errors.append(
                            f"Page {page_result.page_number}: {page_result.error}"
                        )
                    else:
                        result.processed_pages += 1

                    pbar.update(1)

                    if callback:
                        callback(
                            page_result.page_number,
                            len(pages),
                            page_result,
                        )

        # Sort results by page number
        for page_idx in pages:
            page_num = page_idx + 1
            if page_num in page_results:
                result.pages.append(page_results[page_num])


    def extract_from_image(self, image_path: str) -> PageResult:
        """
        Extract Bangla text from a single image file.

        Args:
            image_path: Path to an image file.

        Returns:
            PageResult with extracted text.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        pil_image = Image.open(image_path)

        if self.preprocess:
            preprocessor = PreprocessorConfig.for_scanned_book()
            image_array = preprocessor.process(pil_image)
        else:
            image_array = np.array(pil_image)

        return self._process_page_sequential(image_array, 1)
