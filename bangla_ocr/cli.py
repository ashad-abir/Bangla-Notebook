"""
Command-line interface for Bangla OCR.

Usage:
    python -m bangla_ocr input.pdf -o output.txt
    python -m bangla_ocr input.pdf --format markdown --pages 1-50
    python -m bangla_ocr image.png -o output.txt
"""

import argparse
import os
import sys
import logging

from bangla_ocr.extractor import BanglaOCRExtractor
from bangla_ocr.output_formatter import OutputFormatter


BANNER = r"""
╔══════════════════════════════════════════════════╗
║                                                  ║
║    🇧🇩  বাংলা OCR — Bangla Text Extractor  🇧🇩    ║
║                                                  ║
║    Extract Bengali text from PDFs & images       ║
║    Powered by EasyOCR + Deep Learning            ║
║                                                  ║
╚══════════════════════════════════════════════════╝
"""


def parse_page_range(page_str: str) -> tuple[int, int]:
    """Parse page range string like '1-50' or '10-20'."""
    if "-" in page_str:
        parts = page_str.split("-", 1)
        start = int(parts[0].strip())
        end = int(parts[1].strip())
        if start > end:
            raise argparse.ArgumentTypeError(
                f"Invalid range: start ({start}) > end ({end})"
            )
        return (start, end)
    else:
        page = int(page_str.strip())
        return (page, page)


def detect_format(output_path: str, explicit_format: str = None) -> str:
    """Detect output format from file extension or explicit flag."""
    if explicit_format:
        return explicit_format

    ext = os.path.splitext(output_path)[1].lower()
    format_map = {
        ".txt": "text",
        ".text": "text",
        ".md": "markdown",
        ".markdown": "markdown",
        ".json": "json",
    }
    return format_map.get(ext, "text")


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        prog="bangla_ocr",
        description="🇧🇩 Extract Bangla (Bengali) text from PDF files and images using deep-learning OCR.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract text from a PDF (auto-detect format from extension)
  python -m bangla_ocr book.pdf -o output.txt

  # Extract as Markdown with metadata
  python -m bangla_ocr book.pdf -o output.md

  # Extract specific pages as JSON
  python -m bangla_ocr book.pdf -o output.json --pages 1-50

  # Extract from an image
  python -m bangla_ocr page.png -o output.txt

  # Mixed Bengali + English text
  python -m bangla_ocr book.pdf -o output.txt --lang bn en

  # Skip preprocessing (for clean digital PDFs)
  python -m bangla_ocr book.pdf -o output.txt --no-preprocess

  # Higher quality (slower)
  python -m bangla_ocr book.pdf -o output.txt --dpi 400
        """,
    )

    # Required arguments
    parser.add_argument(
        "input",
        help="Input PDF file or image path",
    )

    # Output
    parser.add_argument(
        "-o", "--output",
        help="Output file path. Format auto-detected from extension (.txt, .md, .json). "
             "If not specified, prints to stdout.",
    )
    parser.add_argument(
        "-f", "--format",
        choices=["text", "txt", "markdown", "md", "json"],
        help="Output format (overrides auto-detection from file extension)",
    )

    # Page range
    parser.add_argument(
        "--pages",
        type=parse_page_range,
        metavar="START-END",
        help="Page range to process (1-indexed, inclusive). Example: 1-50",
    )

    # OCR options
    parser.add_argument(
        "--lang",
        nargs="+",
        default=["bn"],
        help="OCR languages (default: bn). Use 'bn en' for mixed text. "
             "See EasyOCR docs for available languages.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="DPI for PDF page rendering (default: 300). Higher = better quality but slower.",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.2,
        metavar="THRESHOLD",
        help="Minimum confidence threshold (0.0-1.0, default: 0.2)",
    )

    # Processing options
    parser.add_argument(
        "--no-preprocess",
        action="store_true",
        help="Disable image preprocessing (use for clean digital PDFs)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1, recommended for CPU). "
             "Using >1 loads separate OCR models per worker — needs more RAM.",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (requires CUDA-enabled PyTorch)",
    )

    # Markdown-specific options
    parser.add_argument(
        "--show-confidence",
        action="store_true",
        help="Show confidence scores per text region (Markdown format only)",
    )

    # JSON-specific options
    parser.add_argument(
        "--no-bboxes",
        action="store_true",
        help="Exclude bounding box data from JSON output",
    )

    # Verbosity
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose/debug logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress progress output (only print result)",
    )

    return parser


def main(argv=None):
    """Main entry point for the CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Setup logging
    if args.verbose:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
    elif not args.quiet:
        logging.basicConfig(level=logging.WARNING)

    # Print banner
    if not args.quiet:
        print(BANNER, file=sys.stderr)

    # Validate input
    input_path = os.path.abspath(args.input)
    if not os.path.exists(input_path):
        parser.error(f"Input file not found: {input_path}")

    # Detect input type
    ext = os.path.splitext(input_path)[1].lower()
    is_image = ext in (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp")
    is_pdf = ext == ".pdf"

    if not is_image and not is_pdf:
        parser.error(
            f"Unsupported input format: {ext}. "
            "Supported: .pdf, .png, .jpg, .jpeg, .bmp, .tiff, .webp"
        )

    # Create extractor
    extractor = BanglaOCRExtractor(
        languages=args.lang,
        gpu=args.gpu,
        preprocess=not args.no_preprocess,
        confidence_threshold=args.confidence,
        dpi=args.dpi,
        workers=args.workers,
    )

    # Run extraction
    if is_pdf:
        result = extractor.extract(
            input_path,
            page_range=args.pages,
        )
    else:
        # Single image mode
        page_result = extractor.extract_from_image(input_path)
        from bangla_ocr.extractor import ExtractionResult
        result = ExtractionResult(
            input_file=input_path,
            total_pages=1,
            processed_pages=1 if not page_result.error else 0,
            pages=[page_result],
            total_time=page_result.processing_time,
            errors=[page_result.error] if page_result.error else [],
        )

    # Format output
    if args.output:
        fmt = detect_format(args.output, args.format)

        # Build format-specific kwargs
        kwargs = {}
        if fmt in ("markdown", "md"):
            kwargs["include_confidence"] = args.show_confidence
        elif fmt == "json":
            kwargs["include_bboxes"] = not args.no_bboxes

        output_path = OutputFormatter.save(result, args.output, fmt, **kwargs)
        if not args.quiet:
            print(
                f"\n💾 Output saved to: {output_path}",
                file=sys.stderr,
            )
    else:
        # Print to stdout
        fmt = args.format or "text"
        formatter = OutputFormatter()
        if fmt in ("text", "txt"):
            print(formatter.to_text(result))
        elif fmt in ("markdown", "md"):
            print(formatter.to_markdown(
                result, include_confidence=args.show_confidence
            ))
        elif fmt == "json":
            print(formatter.to_json(
                result, include_bboxes=not args.no_bboxes
            ))


if __name__ == "__main__":
    main()
