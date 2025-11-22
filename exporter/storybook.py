from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any, Sequence

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer


_DEFAULT_OUTPUT_DIR = "exports"


def generate_storybook(
    scenes: Sequence[Any],
    image_paths: Sequence[str],
    overall_summary: str | None = None,
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> str:
    if not scenes:
        raise ValueError("At least one scene is required.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"storybook_{int(time())}.pdf"

    doc = SimpleDocTemplate(str(filename), pagesize=A4)
    styles = getSampleStyleSheet()
    story: list = []

    if overall_summary:
        story.append(Paragraph("Story Summary", styles["Heading1"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(overall_summary, styles["BodyText"]))
        story.append(PageBreak())

    for index, scene in enumerate(scenes):
        text = getattr(scene, "text", None)
        if text is None and isinstance(scene, dict):
            text = scene.get("text", "")

        img_path = None
        if index < len(image_paths):
            candidate = Path(image_paths[index])
            if candidate.is_file():
                img_path = candidate

        if img_path is not None:
            story.append(RLImage(str(img_path), width=doc.width))
            story.append(Spacer(1, 8))

        if text:
            story.append(Paragraph(str(text), styles["BodyText"]))
        else:
            story.append(Paragraph("(No text for this scene.)", styles["BodyText"]))

        if index != len(scenes) - 1:
            story.append(PageBreak())

    doc.build(story)
    return str(filename.resolve())
