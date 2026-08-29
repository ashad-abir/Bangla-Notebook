"""
Bangla OCR — Extract Bengali text from PDF files.

A Python tool for extracting Bangla (Bengali) text from large PDF documents
such as NCTB textbooks using deep-learning-based OCR.
"""

__version__ = "1.0.0"
__author__ = "Bangla Notebook"

from bangla_ocr.extractor import BanglaOCRExtractor
from bangla_ocr.preprocessor import ImagePreprocessor
from bangla_ocr.output_formatter import OutputFormatter

__all__ = ["BanglaOCRExtractor", "ImagePreprocessor", "OutputFormatter"]
