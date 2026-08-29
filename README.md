# 🇧🇩 বাংলা OCR — Bangla Text Extractor

Extract Bengali (Bangla) text from large PDF files like NCTB textbooks using deep-learning-based OCR.

**Powered by:** EasyOCR · OpenCV · pypdfium2

---

## ✨ Features

- 🔤 **Bengali OCR** — Deep-learning-based text recognition optimized for Bangla script
- 📄 **PDF Support** — Process entire books, page by page, with progress tracking
- 🖼️ **Image Support** — Also works on standalone images (PNG, JPG, TIFF, etc.)
- 🔧 **Smart Preprocessing** — Adaptive thresholding, denoising, deskew correction
- 📊 **Multiple Output Formats** — Plain text, Markdown, and JSON with bounding boxes
- ⚡ **Parallel Processing** — Optional multi-worker mode for faster extraction
- 🎯 **Page Range Selection** — Process specific pages instead of the full document

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Create a virtual environment (recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Extract Text from a PDF

```bash
# Basic extraction (output to terminal)
python -m bangla_ocr book.pdf

# Save as plain text
python -m bangla_ocr book.pdf -o output.txt

# Save as Markdown (with metadata)
python -m bangla_ocr book.pdf -o output.md

# Save as JSON (with bounding boxes & confidence)
python -m bangla_ocr book.pdf -o output.json
```

### 3. Advanced Options

```bash
# Process only pages 10-30
python -m bangla_ocr book.pdf -o output.txt --pages 10-30

# Mixed Bengali + English text
python -m bangla_ocr book.pdf -o output.txt --lang bn en

# Higher quality rendering (slower)
python -m bangla_ocr book.pdf -o output.txt --dpi 400

# Skip preprocessing for clean digital PDFs
python -m bangla_ocr book.pdf -o output.txt --no-preprocess

# Show confidence scores in Markdown output
python -m bangla_ocr book.pdf -o output.md --show-confidence

# Extract from a single image
python -m bangla_ocr page_scan.png -o output.txt
```

---

## 📖 Usage as a Python Library

```python
from bangla_ocr import BanglaOCRExtractor, OutputFormatter

# Initialize the extractor
extractor = BanglaOCRExtractor(
    languages=["bn"],       # Bengali
    dpi=300,                 # Rendering quality
    preprocess=True,         # Enable image preprocessing
    confidence_threshold=0.2 # Minimum detection confidence
)

# Extract from PDF
result = extractor.extract("nctb_book.pdf", page_range=(1, 50))

# Get the full text
print(result.full_text)

# Save in different formats
OutputFormatter.save(result, "output.txt", format="text")
OutputFormatter.save(result, "output.md", format="markdown")
OutputFormatter.save(result, "output.json", format="json")

# Access individual pages
for page in result.pages:
    print(f"Page {page.page_number}: {len(page.regions)} text regions")
    for region in page.regions:
        print(f"  [{region.confidence:.0%}] {region.text}")
```

---

## ⚙️ CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `input` | (required) | Input PDF or image file path |
| `-o, --output` | stdout | Output file path |
| `-f, --format` | auto | Output format: `text`, `markdown`, `json` |
| `--pages` | all | Page range (e.g., `1-50`) |
| `--lang` | `bn` | OCR languages (e.g., `bn en`) |
| `--dpi` | `300` | PDF rendering DPI |
| `--confidence` | `0.2` | Min confidence threshold |
| `--no-preprocess` | off | Disable image preprocessing |
| `--workers` | `1` | Parallel workers (needs more RAM) |
| `--gpu` | off | Enable GPU acceleration |
| `--show-confidence` | off | Show confidence in Markdown |
| `--no-bboxes` | off | Exclude bounding boxes from JSON |
| `-v, --verbose` | off | Debug logging |
| `-q, --quiet` | off | Suppress progress output |

---

## 🧰 Preprocessing Pipeline

The image preprocessor applies these steps to improve OCR accuracy on scanned documents:

1. **Grayscale conversion** — Simplifies the image for text detection
2. **Deskew correction** — Detects and corrects page rotation using Hough line transform
3. **Denoising** — Non-local means denoising removes scan artifacts
4. **Adaptive thresholding** — Handles uneven lighting/shadows on scanned pages
5. **Sharpening** — Unsharp masking enhances text edges

For clean, digitally-created PDFs, use `--no-preprocess` to skip this pipeline.

---

## 📝 Notes

- **First run** will download the EasyOCR Bengali model (~100MB). Subsequent runs use the cached model.
- **Processing time** on CPU: approximately 5-10 seconds per page at 300 DPI.
- **Memory usage**: ~2-3 GB during processing (mostly the OCR model).
- For best results on scanned books, use **300 DPI** (default).

---

## 📂 Project Structure

```
bangla_ocr/
├── __init__.py          # Package initialization
├── __main__.py          # Module entry point
├── cli.py               # Command-line interface
├── preprocessor.py      # Image preprocessing pipeline
├── extractor.py         # Core OCR engine
└── output_formatter.py  # Output formatting (text/md/json)
```
