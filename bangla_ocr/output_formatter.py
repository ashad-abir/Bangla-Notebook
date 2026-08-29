"""
Output formatters for OCR extraction results.

Supports plain text, Markdown, and JSON output formats,
preserving page boundaries and optional metadata.
"""

import json
from datetime import datetime
from typing import Optional

from bangla_ocr.extractor import ExtractionResult


class OutputFormatter:
    """Format OCR extraction results into various output formats."""

    @staticmethod
    def to_text(
        result: ExtractionResult,
        include_page_numbers: bool = True,
        separator: str = "─" * 50,
    ) -> str:
        """
        Format results as plain text.

        Args:
            result: Extraction result to format.
            include_page_numbers: Include page number headers.
            separator: Page separator string.

        Returns:
            Formatted plain text string.
        """
        lines = []

        for page in result.pages:
            if include_page_numbers:
                lines.append(f"\n{separator}")
                lines.append(f"  পৃষ্ঠা {page.page_number} | Page {page.page_number}")
                lines.append(f"{separator}\n")

            if page.text.strip():
                lines.append(page.text)
            elif page.error:
                lines.append(f"[Error: {page.error}]")
            else:
                lines.append("[No text detected on this page]")

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def to_markdown(
        result: ExtractionResult,
        include_metadata: bool = True,
        include_confidence: bool = False,
    ) -> str:
        """
        Format results as Markdown.

        Args:
            result: Extraction result to format.
            include_metadata: Include document metadata header.
            include_confidence: Include per-region confidence scores.

        Returns:
            Formatted Markdown string.
        """
        lines = []

        # Document header
        if include_metadata:
            import os
            filename = os.path.basename(result.input_file)
            lines.append(f"# 📖 {filename}\n")
            lines.append(f"> **Extracted on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ")
            lines.append(f"> **Total pages:** {result.total_pages}  ")
            lines.append(f"> **Successfully processed:** {result.processed_pages}  ")
            lines.append(f"> **Processing time:** {result.total_time:.1f}s  ")
            lines.append(f"> **Success rate:** {result.success_rate:.1f}%\n")
            lines.append("---\n")

        for page in result.pages:
            lines.append(f"## পৃষ্ঠা {page.page_number}\n")

            if page.error:
                lines.append(f"> ⚠️ **Error:** {page.error}\n")
                continue

            if not page.text.strip():
                lines.append("*এই পৃষ্ঠায় কোনো টেক্সট পাওয়া যায়নি।*\n")
                continue

            if include_confidence and page.regions:
                # Show text with confidence annotations
                for region in page.regions:
                    conf_emoji = "🟢" if region.confidence > 0.8 else (
                        "🟡" if region.confidence > 0.5 else "🔴"
                    )
                    lines.append(
                        f"{region.text} {conf_emoji} *({region.confidence:.0%})*  "
                    )
            else:
                lines.append(page.text)

            lines.append("")
            lines.append(
                f"*⏱ Processing time: {page.processing_time:.1f}s*\n"
            )
            lines.append("---\n")

        return "\n".join(lines)

    @staticmethod
    def to_json(
        result: ExtractionResult,
        include_bboxes: bool = True,
        indent: int = 2,
    ) -> str:
        """
        Format results as JSON with full metadata.

        Args:
            result: Extraction result to format.
            include_bboxes: Include bounding box coordinates.
            indent: JSON indentation level.

        Returns:
            Formatted JSON string.
        """
        data = {
            "metadata": {
                "input_file": result.input_file,
                "extraction_date": datetime.now().isoformat(),
                "total_pages": result.total_pages,
                "processed_pages": result.processed_pages,
                "success_rate": round(result.success_rate, 2),
                "total_time_seconds": round(result.total_time, 2),
                "errors": result.errors,
            },
            "pages": [],
        }

        for page in result.pages:
            page_data = {
                "page_number": page.page_number,
                "text": page.text,
                "processing_time_seconds": round(page.processing_time, 2),
            }

            if page.error:
                page_data["error"] = page.error

            if include_bboxes and page.regions:
                page_data["regions"] = [
                    {
                        "text": region.text,
                        "confidence": round(region.confidence, 4),
                        "bounding_box": region.bbox,
                    }
                    for region in page.regions
                ]

            data["pages"].append(page_data)

        return json.dumps(data, ensure_ascii=False, indent=indent)

    @staticmethod
    def save(
        result: ExtractionResult,
        output_path: str,
        format: str = "text",
        **kwargs,
    ) -> str:
        """
        Save extraction results to a file.

        Args:
            result: Extraction result to save.
            output_path: Output file path.
            format: Output format - 'text', 'markdown', or 'json'.
            **kwargs: Additional arguments passed to the formatter.

        Returns:
            The output file path.
        """
        formatter = OutputFormatter()

        if format == "text" or format == "txt":
            content = formatter.to_text(result, **kwargs)
        elif format == "markdown" or format == "md":
            content = formatter.to_markdown(result, **kwargs)
        elif format == "json":
            content = formatter.to_json(result, **kwargs)
        else:
            raise ValueError(
                f"Unknown format: {format}. Use 'text', 'markdown', or 'json'."
            )

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        return output_path
