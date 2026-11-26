#!/usr/bin/env python3
"""Check whether chosen reference crops reached the benchmark manifest.

``configs/references.json`` records intent; ``data/<dataset>/manifest.json`` is
what the benchmark actually reads. Choosing a crop does not rebuild the
manifest, so the two can disagree, and a run then measures the old crop while
looking like it used the new one.

Usage:
    uv run python scripts/ui/check_references.py
    uv run python scripts/ui/check_references.py --dataset rpc
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

__all__ = ["main", "compare"]

TOLERANCE = 0.5


def manifest_references(manifest: Path) -> dict[str, dict[str, object]]:
    """Return the reference used by the manifest for each instance.

    Args:
        manifest: Manifest path.

    Returns:
        Instance to reference path, box, and pair count.
    """
    records = json.loads(manifest.read_text())
    references: dict[str, dict[str, object]] = {}
    for record in records:
        name = str(record["instance_id"])
        entry = references.setdefault(
            name,
            {
                "reference": Path(str(record["reference"])).name,
                "box": record.get("reference_box"),
                "pairs": 0,
            },
        )
        entry["pairs"] = int(entry["pairs"]) + 1
    return references


def compare(
    selection: Mapping[str, Mapping[str, object]],
    references: Mapping[str, Mapping[str, object]],
    root: Path,
) -> list[dict[str, object]]:
    """Compare chosen crops against what the manifest uses.

    Args:
        selection: Dataset section of ``configs/references.json``.
        references: Manifest references per instance.
        root: Dataset directory, used to check that the image exists.

    Returns:
        One status record per chosen instance.
    """
    rows: list[dict[str, object]] = []
    for instance, choice in sorted(selection.items()):
        chosen_image = choice.get("image")
        chosen_box = choice.get("box")
        current = references.get(instance)
        if current is None:
            rows.append(
                {
                    "instance": instance,
                    "status": "unknown-instance",
                    "chosen": chosen_image,
                    "manifest": "",
                    "note": "not present in the manifest",
                }
            )
            continue
        image_matches = str(current["reference"]) == str(chosen_image)
        box_matches = _boxes_match(current.get("box"), chosen_box)
        applied = image_matches and box_matches
        exists = (root / "images" / instance / str(chosen_image)).exists()
        status = "applied" if applied else "pending"
        if not image_matches:
            note = f"manifest uses {current['reference']}"
        elif not box_matches:
            note = "same image, different crop box"
        elif not exists:
            status = "missing-image"
            note = "chosen image is not on disk"
        else:
            note = "rebuild needed" if not applied else ""
        rows.append(
            {
                "instance": instance,
                "status": status,
                "chosen": chosen_image,
                "manifest": current["reference"],
                "note": note or str(choice.get("status", "")),
            }
        )
    return rows


def _boxes_match(left: object, right: object) -> bool:
    """Compare two reference boxes within a pixel tolerance.

    Args:
        left: Manifest box.
        right: Chosen box.

    Returns:
        True when both are missing or numerically close.
    """
    if left is None or right is None:
        return left is None and right is None
    if not isinstance(left, (list, tuple)) or not isinstance(right, (list, tuple)):
        return False
    if len(left) != 4 or len(right) != 4:
        return False
    return all(abs(float(a) - float(b)) <= TOLERANCE for a, b in zip(left, right))


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reference check.

    Args:
        argv: Optional argument list.

    Returns:
        Process exit code: 0 when every choice is applied, 1 otherwise.
    """
    parser = argparse.ArgumentParser(
        description="Check that chosen reference crops reached the manifest."
    )
    parser.add_argument("--dataset", default="flickrlogos32")
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("configs/references.json"),
    )
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args(argv)

    manifest = args.manifest or Path("data") / args.dataset / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"manifest not found: {manifest}")
    if not args.selection.exists():
        print(f"no selection file at {args.selection}; nothing to check")
        return 0

    payload = json.loads(args.selection.read_text())
    selection = payload.get(args.dataset, payload)
    if not selection:
        print(f"no choices recorded for {args.dataset}")
        return 0

    references = manifest_references(manifest)
    rows = compare(selection, references, manifest.parent)
    width = max(len(str(row["instance"])) for row in rows)
    print(f"{'instance'.ljust(width)}  {'status':16s}  chosen -> manifest")
    for row in rows:
        arrow = (
            f"{row['chosen']} -> {row['manifest']}"
            if row["chosen"] != row["manifest"]
            else f"{row['chosen']}"
        )
        print(
            f"{str(row['instance']).ljust(width)}  "
            f"{str(row['status']).ljust(16)}  {arrow}"
            + (f"  ({row['note']})" if row["note"] else "")
        )

    pending = [row for row in rows if row["status"] != "applied"]
    print()
    print(f"{len(rows) - len(pending)}/{len(rows)} choices applied")
    if pending:
        print(
            "rebuild with: make data-flickr "
            'ARGS="--skip-download --references configs/references.json"'
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
