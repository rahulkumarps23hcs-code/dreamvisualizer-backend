from __future__ import annotations

from pathlib import Path
from time import time
from typing import Any, Sequence

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Image as RLImage, PageBreak, Paragraph, SimpleDocTemplate, Spacer


_DEFAULT_OUTPUT_DIR = "exports"


def generate_comic(
    scenes: Sequence[Any],
    image_paths: Sequence[str],
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> str:
    if not scenes:
        raise ValueError("At least one scene is required.")

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"comicbook_{int(time())}.pdf"

    doc = SimpleDocTemplate(str(filename), pagesize=A4)
    styles = getSampleStyleSheet()
    story: list = []

    story.append(Paragraph("Comic Book", styles["Heading1"]))
    story.append(Spacer(1, 16))

    for index, scene in enumerate(scenes, start=1):
        text = getattr(scene, "text", None)
        if text is None and isinstance(scene, dict):
            text = scene.get("text", "")

        img_path = None
        scene_idx = index - 1
        if scene_idx < len(image_paths):
            candidate = Path(image_paths[scene_idx])
            if candidate.is_file():
                img_path = candidate

        if img_path is not None:
            story.append(RLImage(str(img_path), width=doc.width * 0.8))
            story.append(Spacer(1, 4))

        bubble = f"{text}" if text else "(No dialogue for this panel.)"
        story.append(Paragraph(bubble, styles["BodyText"]))
        story.append(Spacer(1, 12))

        if index % 3 == 0 and index != len(scenes):
            story.append(PageBreak())

    doc.build(story)
    return str(filename.resolve())
