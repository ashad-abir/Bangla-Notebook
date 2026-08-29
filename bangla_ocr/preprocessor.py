"""
Image preprocessing pipeline for maximizing Bangla OCR accuracy.

Handles grayscale conversion, adaptive thresholding, denoising,
and deskew correction to improve recognition of complex Bengali
conjuncts (যুক্তাক্ষর) and vowel signs (কার).
"""

import cv2
import numpy as np
from PIL import Image


class ImagePreprocessor:
    """Preprocess page images before OCR to improve accuracy."""

    def __init__(
        self,
        denoise: bool = True,
        threshold: bool = True,
        deskew: bool = True,
        sharpen: bool = True,
        target_dpi: int = 300,
    ):
        """
        Initialize the preprocessor.

        Args:
            denoise: Apply non-local means denoising.
            threshold: Apply adaptive thresholding.
            deskew: Detect and correct page skew.
            sharpen: Apply sharpening to enhance text edges.
            target_dpi: Target DPI for rendering (used by extractor).
        """
        self.denoise = denoise
        self.threshold = threshold
        self.deskew = deskew
        self.sharpen = sharpen
        self.target_dpi = target_dpi

    def process(self, pil_image: Image.Image) -> np.ndarray:
        """
        Run the full preprocessing pipeline on a PIL image.

        Args:
            pil_image: Input PIL Image (from PDF page render).

        Returns:
            Preprocessed image as numpy array ready for OCR.
        """
        # Convert PIL Image to OpenCV format (BGR)
        img = np.array(pil_image)
        if len(img.shape) == 3 and img.shape[2] == 4:
            # RGBA → RGB
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        elif len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # Step 1: Convert to grayscale for processing
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Step 2: Deskew correction
        if self.deskew:
            gray = self._deskew(gray)

        # Step 3: Denoise
        if self.denoise:
            gray = self._denoise(gray)

        # Step 4: Adaptive thresholding
        if self.threshold:
            gray = self._adaptive_threshold(gray)

        # Step 5: Sharpen text edges
        if self.sharpen:
            gray = self._sharpen(gray)

        # Convert back to 3-channel for EasyOCR (it expects RGB/BGR)
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        return result

    def _deskew(self, img: np.ndarray) -> np.ndarray:
        """Detect and correct page skew using Hough line transform."""
        try:
            # Use a copy for edge detection
            edges = cv2.Canny(img, 50, 150, apertureSize=3)
            lines = cv2.HoughLinesP(
                edges,
                rho=1,
                theta=np.pi / 180,
                threshold=100,
                minLineLength=img.shape[1] // 4,
                maxLineGap=20,
            )

            if lines is None or len(lines) == 0:
                return img

            # Calculate median angle from detected lines
            angles = []
            for line in lines:
                # Flatten to handle varying shapes across OpenCV versions
                # HoughLinesP may return (N,1,4) or (N,4) arrays
                coords = np.array(line).flatten()
                if len(coords) < 4:
                    continue
                x1, y1, x2, y2 = coords[:4]
                if x2 - x1 == 0:
                    continue
                angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                # Only consider near-horizontal lines (text lines)
                if abs(angle) < 15:
                    angles.append(angle)

            if not angles:
                return img

            median_angle = np.median(angles)

            # Only correct if skew is significant but not too extreme
            if abs(median_angle) < 0.5 or abs(median_angle) > 10:
                return img

            # Rotate to correct skew
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            rotated = cv2.warpAffine(
                img,
                rotation_matrix,
                (w, h),
                flags=cv2.INTER_CUBIC,
                borderMode=cv2.BORDER_REPLICATE,
            )
            return rotated
        except Exception:
            # If deskew fails for any reason, return the original image
            return img

    def _denoise(self, img: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising — good for scanned documents."""
        return cv2.fastNlMeansDenoising(
            img,
            h=10,
            templateWindowSize=7,
            searchWindowSize=21,
        )

    def _adaptive_threshold(self, img: np.ndarray) -> np.ndarray:
        """
        Apply adaptive thresholding to handle uneven lighting.

        Uses Gaussian adaptive threshold which works well for
        printed text on paper with varying background brightness.
        """
        return cv2.adaptiveThreshold(
            img,
            maxValue=255,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY,
            blockSize=15,
            C=8,
        )

    def _sharpen(self, img: np.ndarray) -> np.ndarray:
        """Apply unsharp masking to enhance text edges."""
        blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
        return sharpened


class PreprocessorConfig:
    """Predefined preprocessing configurations for common scenarios."""

    @staticmethod
    def for_clean_pdf() -> ImagePreprocessor:
        """For clean, digitally-created PDFs — minimal processing."""
        return ImagePreprocessor(
            denoise=False,
            threshold=False,
            deskew=False,
            sharpen=False,
            target_dpi=200,
        )

    @staticmethod
    def for_scanned_book() -> ImagePreprocessor:
        """For scanned NCTB books — full pipeline (recommended)."""
        return ImagePreprocessor(
            denoise=True,
            threshold=True,
            deskew=True,
            sharpen=True,
            target_dpi=300,
        )

    @staticmethod
    def for_photo() -> ImagePreprocessor:
        """For photos of book pages — heavy preprocessing."""
        return ImagePreprocessor(
            denoise=True,
            threshold=True,
            deskew=True,
            sharpen=True,
            target_dpi=400,
        )
