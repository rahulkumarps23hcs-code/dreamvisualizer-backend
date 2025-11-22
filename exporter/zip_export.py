from __future__ import annotations

import json
import zipfile
from pathlib import Path
from time import time
from typing import Any, Mapping, Sequence


_DEFAULT_OUTPUT_DIR = "exports"


def generate_zip(
    all_assets: Mapping[str, Any],
    output_dir: str = _DEFAULT_OUTPUT_DIR,
) -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    filename = out_dir / f"bundle_{int(time())}.zip"

    images: Sequence[str] = all_assets.get("images", []) or []
    audio: Sequence[str] = all_assets.get("audio", []) or []
    video: str | None = all_assets.get("video") or None
    metadata: Any = all_assets.get("metadata")

    with zipfile.ZipFile(str(filename), "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in images:
            p = Path(path)
            if p.is_file():
                zf.write(str(p), arcname=f"images/{p.name}")

        for path in audio:
            p = Path(path)
            if p.is_file():
                zf.write(str(p), arcname=f"audio/{p.name}")

        if video:
            vp = Path(video)
            if vp.is_file():
                zf.write(str(vp), arcname="video/final.mp4")

        if metadata is not None:
            data = json.dumps(metadata, indent=2, ensure_ascii=False)
            zf.writestr("metadata.json", data)

    return str(filename.resolve())
