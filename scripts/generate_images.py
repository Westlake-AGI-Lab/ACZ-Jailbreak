#!/usr/bin/env python3
"""Generate ACZ text-rendered image datasets from text records."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from acz_jailbreak.io import iter_limited, load_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV, JSONL, or JSON records.")
    parser.add_argument("--output", required=True, help="Output image root.")
    parser.add_argument("--text-column", default="query", help="Column containing prompt text.")
    parser.add_argument("--id-column", default="id", help="Column containing stable item IDs.")
    parser.add_argument(
        "--dpi",
        nargs="+",
        type=int,
        default=[15, 30, 45, 60, 90, 120, 150, 200, 300],
        help="One or more DPI values.",
    )
    parser.add_argument("--font-path", default=None, help="TTF/TTC font path. Defaults to a system font.")
    parser.add_argument("--font-size", type=int, default=9)
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--page-size", default="A4")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crop", action="store_true", help="Crop whitespace around rendered text.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from acz_jailbreak.text2image import convert_text_to_images, find_default_font

    records = load_records(args.input)
    font_path = args.font_path or find_default_font()
    output_root = Path(args.output)

    generated = 0
    for record in iter_limited(records, args.limit):
        text = str(record.get(args.text_column, "")).strip()
        item_id = str(record.get(args.id_column, generated + 1)).strip()
        if not text:
            continue
        for dpi in args.dpi:
            item_dir = output_root / f"dpi{dpi}" / item_id
            expected = item_dir / f"{item_id}_001.png"
            if expected.exists() and not args.overwrite:
                continue
            convert_text_to_images(
                text=text,
                output_path=item_dir / f"{item_id}.png",
                font_path=font_path,
                dpi=dpi,
                font_size=args.font_size,
                margin=args.margin,
                page_size=args.page_size,
                auto_crop_to_content=args.crop,
            )
        generated += 1
    print(f"Generated images for {generated} records under {output_root}")


if __name__ == "__main__":
    main()
