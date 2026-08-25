#!/usr/bin/env python3
"""Prepare apple and capsicum subsets from kaggle_gpt.zip for ATOH.

Source annotations are single-class YOLO detection labels (class 0 inside
each crop-specific dataset). Target classes use the fixed ATOH mapping:

* apple -> class 0
* capsicum -> class 2 (bell_pepper)

The source train/valid/test splits are preserved. The script performs a dry
run by default and writes nothing unless ``--execute`` is supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError


CROPS = {
    "apple": {"target_id": 0, "target_name": "apple"},
    "capsicum": {"target_id": 2, "target_name": "bell_pepper"},
}
SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

EXPECTED_IMAGES = {
    "apple": {"train": 49, "valid": 7, "test": 7},
    "capsicum": {"train": 98, "valid": 12, "test": 12},
}
EXPECTED_BOXES = {
    "apple": {"train": 292, "valid": 23, "test": 39},
    "capsicum": {"train": 583, "valid": 65, "test": 76},
}
EXPECTED_TARGET_BOXES = {0: 354, 2: 724}


@dataclass(frozen=True)
class Record:
    crop: str
    split: str
    source_image: str
    output_image: str
    output_label: str
    label_text: str


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Convert kaggle_gpt apple/capsicum labels to fixed ATOH IDs."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=home / "Downloads" / "kaggle_gpt.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            home
            / "projects"
            / "atoh_agriculture"
            / "datasets"
            / "staging"
            / "kaggle_gpt"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write converted files. Without this flag nothing is written.",
    )
    return parser.parse_args()


def inspect_image(data: bytes) -> int:
    try:
        with Image.open(BytesIO(data)) as image:
            orientation = int(image.getexif().get(274, 1))
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error
    return orientation


def convert_label_line(
    source_line: str,
    target_class_id: int,
) -> tuple[str | None, bool, str | None]:
    parts = source_line.split()
    if len(parts) != 5:
        return None, False, "malformed"

    try:
        source_class = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])
    except ValueError:
        return None, False, "malformed"

    values = (center_x, center_y, width, height)
    if source_class != 0:
        return None, False, "unknown_class"
    if not all(math.isfinite(value) for value in values):
        return None, False, "invalid_box"
    if width <= 0 or height <= 0:
        return None, False, "invalid_box"

    raw_edges = (
        center_x - width / 2.0,
        center_y - height / 2.0,
        center_x + width / 2.0,
        center_y + height / 2.0,
    )
    x1 = max(0.0, min(1.0, raw_edges[0]))
    y1 = max(0.0, min(1.0, raw_edges[1]))
    x2 = max(0.0, min(1.0, raw_edges[2]))
    y2 = max(0.0, min(1.0, raw_edges[3]))

    if x2 <= x1 or y2 <= y1:
        return None, False, "invalid_box"

    clipped = any(
        abs(raw - clean) > 1e-9
        for raw, clean in zip(raw_edges, (x1, y1, x2, y2))
    )
    output_center_x = (x1 + x2) / 2.0
    output_center_y = (y1 + y2) / 2.0
    output_width = x2 - x1
    output_height = y2 - y1
    output_line = (
        f"{target_class_id} {output_center_x:.8f} {output_center_y:.8f} "
        f"{output_width:.8f} {output_height:.8f}"
    )
    return output_line, clipped, None


def make_output_names(crop: str, split: str, source_name: str) -> tuple[str, str]:
    basename = PurePosixPath(source_name).name
    image_name = f"kaggle_gpt_{crop}_{split}_{basename}"
    label_name = str(Path(image_name).with_suffix(".txt"))
    return image_name, label_name


def analyze(archive_path: Path) -> tuple[dict[str, Any], list[Record]]:
    stats: dict[str, Any] = {
        "subsets": {},
        "total_images": 0,
        "total_boxes": 0,
        "target_class_box_counts": {"0": 0, "2": 0},
        "missing_images": 0,
        "missing_labels": 0,
        "duplicate_image_stems": 0,
        "duplicate_label_stems": 0,
        "duplicate_output_names": 0,
        "empty_labels": 0,
        "malformed_lines": 0,
        "unknown_source_classes": 0,
        "invalid_boxes": 0,
        "clipped_boxes": 0,
        "corrupt_images": 0,
        "nonstandard_exif_orientations": 0,
        "duplicate_content_groups": 0,
        "duplicate_content_files": 0,
        "cross_split_duplicate_groups": 0,
        "yaml_files_present": 0,
        "errors": [],
    }
    records: list[Record] = []
    output_names: set[tuple[str, str]] = set()
    content_locations: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    with zipfile.ZipFile(archive_path) as archive:
        member_list = [name for name in archive.namelist() if not name.endswith("/")]
        members = set(member_list)

        for crop, crop_config in CROPS.items():
            yaml_member = f"{crop}/data.yaml"
            if yaml_member in members:
                stats["yaml_files_present"] += 1
            else:
                stats["errors"].append(f"missing metadata: {yaml_member}")

            for split in SPLITS:
                image_prefix = f"{crop}/{split}/images/"
                label_prefix = f"{crop}/{split}/labels/"

                image_members = sorted(
                    name
                    for name in member_list
                    if name.startswith(image_prefix)
                    and PurePosixPath(name).suffix.lower() in IMAGE_SUFFIXES
                )
                label_members = sorted(
                    name
                    for name in member_list
                    if name.startswith(label_prefix) and name.endswith(".txt")
                )

                images_by_stem: dict[str, str] = {}
                labels_by_stem: dict[str, str] = {}
                for name in image_members:
                    stem = PurePosixPath(name).stem
                    if stem in images_by_stem:
                        stats["duplicate_image_stems"] += 1
                    else:
                        images_by_stem[stem] = name
                for name in label_members:
                    stem = PurePosixPath(name).stem
                    if stem in labels_by_stem:
                        stats["duplicate_label_stems"] += 1
                    else:
                        labels_by_stem[stem] = name

                missing_label_stems = set(images_by_stem) - set(labels_by_stem)
                missing_image_stems = set(labels_by_stem) - set(images_by_stem)
                stats["missing_labels"] += len(missing_label_stems)
                stats["missing_images"] += len(missing_image_stems)

                subset_images = 0
                subset_boxes = 0
                for stem in sorted(set(images_by_stem) & set(labels_by_stem)):
                    image_member = images_by_stem[stem]
                    label_member = labels_by_stem[stem]
                    image_data = archive.read(image_member)

                    try:
                        orientation = inspect_image(image_data)
                    except RuntimeError as error:
                        stats["corrupt_images"] += 1
                        stats["errors"].append(
                            f"corrupt image: {image_member}: {error}"
                        )
                        continue

                    if orientation != 1:
                        stats["nonstandard_exif_orientations"] += 1

                    digest = hashlib.sha256(image_data).hexdigest()
                    content_locations[digest].append((crop, split, image_member))

                    source_text = archive.read(label_member).decode(
                        "utf-8-sig", errors="replace"
                    ).strip()
                    source_lines = source_text.splitlines() if source_text else []
                    if not source_lines:
                        stats["empty_labels"] += 1

                    converted_lines: list[str] = []
                    for source_line in source_lines:
                        output_line, clipped, error_kind = convert_label_line(
                            source_line,
                            int(crop_config["target_id"]),
                        )
                        if error_kind == "malformed":
                            stats["malformed_lines"] += 1
                            continue
                        if error_kind == "unknown_class":
                            stats["unknown_source_classes"] += 1
                            continue
                        if error_kind == "invalid_box" or output_line is None:
                            stats["invalid_boxes"] += 1
                            continue

                        converted_lines.append(output_line)
                        subset_boxes += 1
                        target_key = str(crop_config["target_id"])
                        stats["target_class_box_counts"][target_key] += 1
                        if clipped:
                            stats["clipped_boxes"] += 1

                    output_image, output_label = make_output_names(
                        crop, split, image_member
                    )
                    output_key = (split, output_image)
                    if output_key in output_names:
                        stats["duplicate_output_names"] += 1
                    output_names.add(output_key)

                    label_text = "\n".join(converted_lines)
                    if label_text:
                        label_text += "\n"
                    records.append(
                        Record(
                            crop=crop,
                            split=split,
                            source_image=image_member,
                            output_image=output_image,
                            output_label=output_label,
                            label_text=label_text,
                        )
                    )
                    subset_images += 1

                stats["subsets"][f"{crop}_{split}"] = {
                    "images": subset_images,
                    "boxes": subset_boxes,
                }
                stats["total_images"] += subset_images
                stats["total_boxes"] += subset_boxes

    duplicate_groups = [
        locations for locations in content_locations.values() if len(locations) > 1
    ]
    stats["duplicate_content_groups"] = len(duplicate_groups)
    stats["duplicate_content_files"] = sum(
        len(locations) - 1 for locations in duplicate_groups
    )
    stats["cross_split_duplicate_groups"] = sum(
        1
        for locations in duplicate_groups
        if len({split for _, split, _ in locations}) > 1
    )
    return stats, records


def is_ready(stats: dict[str, Any]) -> bool:
    expected_subsets = {
        f"{crop}_{split}": {
            "images": EXPECTED_IMAGES[crop][split],
            "boxes": EXPECTED_BOXES[crop][split],
        }
        for crop in CROPS
        for split in SPLITS
    }
    return all(
        [
            stats["subsets"] == expected_subsets,
            stats["total_images"] == 185,
            stats["total_boxes"] == 1078,
            stats["target_class_box_counts"]
            == {str(key): value for key, value in EXPECTED_TARGET_BOXES.items()},
            stats["missing_images"] == 0,
            stats["missing_labels"] == 0,
            stats["duplicate_image_stems"] == 0,
            stats["duplicate_label_stems"] == 0,
            stats["duplicate_output_names"] == 0,
            stats["empty_labels"] == 0,
            stats["malformed_lines"] == 0,
            stats["unknown_source_classes"] == 0,
            stats["invalid_boxes"] == 0,
            stats["clipped_boxes"] == 0,
            stats["corrupt_images"] == 0,
            stats["nonstandard_exif_orientations"] == 0,
            stats["duplicate_content_groups"] == 0,
            stats["cross_split_duplicate_groups"] == 0,
            stats["yaml_files_present"] == 2,
            not stats["errors"],
        ]
    )


def print_summary(stats: dict[str, Any], mode: str, ready: bool) -> None:
    print(f"Mode: {mode}")
    for crop in CROPS:
        for split in SPLITS:
            subset = stats["subsets"].get(f"{crop}_{split}", {})
            print(
                f"{crop} {split}: "
                f"{subset.get('images', 0)} images, "
                f"{subset.get('boxes', 0)} boxes"
            )
    print(f"Total: {stats['total_images']} images, {stats['total_boxes']} boxes")
    print(f"Target boxes: {stats['target_class_box_counts']}")
    print(
        "Labels: "
        f"clipped={stats['clipped_boxes']}, "
        f"invalid={stats['invalid_boxes']}, "
        f"malformed={stats['malformed_lines']}, "
        f"empty={stats['empty_labels']}"
    )
    print(
        "Files: "
        f"missing_images={stats['missing_images']}, "
        f"missing_labels={stats['missing_labels']}, "
        f"corrupt_images={stats['corrupt_images']}"
    )
    print(
        "Duplicates: "
        f"groups={stats['duplicate_content_groups']}, "
        f"cross_split={stats['cross_split_duplicate_groups']}"
    )
    print(f"READY: {'yes' if ready else 'no'}")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_output(
    archive_path: Path,
    output: Path,
    records: list[Record],
    stats: dict[str, Any],
) -> None:
    temporary_output = output.with_name(output.name + ".part")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if temporary_output.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_output}")

    temporary_output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            image_destination = (
                temporary_output / record.split / "images" / record.output_image
            )
            label_destination = (
                temporary_output / record.split / "labels" / record.output_label
            )
            write_bytes(image_destination, archive.read(record.source_image))
            write_bytes(label_destination, record.label_text.encode("utf-8"))

    manifest = {
        "source": "kaggle_gpt.zip crop-specific YOLO datasets",
        "source_archive": str(archive_path.resolve()),
        "included_sources": ["apple", "capsicum"],
        "excluded_sources": [
            "avocado",
            "blueberry",
            "cherry",
            "kiwi",
            "mango",
            "orange",
            "rockmelon",
            "strawberry",
            "wheat",
        ],
        "target_task": "YOLO object detection",
        "class_mapping": {
            "apple/source_class_0": {"id": 0, "name": "apple"},
            "capsicum/source_class_0": {"id": 2, "name": "bell_pepper"},
        },
        "split_policy": "Source train/valid/test splits are preserved.",
        "image_policy": "Original image bytes are preserved.",
        "license": {
            "status": "verify against the official Kaggle source pages",
            "project_rule": (
                "Technical preparation may proceed for the academic/internship "
                "prototype; reassess redistribution and commercial use separately."
            ),
        },
    }
    write_bytes(
        temporary_output / "source_manifest.json",
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    write_bytes(
        temporary_output / "conversion_report.json",
        (json.dumps(stats, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    temporary_output.replace(output)


def main() -> int:
    args = parse_args()
    if not args.archive.is_file():
        print(f"ERROR: archive not found: {args.archive}", file=sys.stderr)
        return 1

    try:
        stats, records = analyze(args.archive)
        ready = is_ready(stats)
        mode = "execute" if args.execute else "dry-run"
        print_summary(stats, mode, ready)

        if not ready:
            if stats["errors"]:
                for error in stats["errors"][:10]:
                    print(f"ERROR DETAIL: {error}", file=sys.stderr)
            print("No files were written because the audit did not pass.", file=sys.stderr)
            return 2

        if not args.execute:
            print("No files were written.")
            return 0

        write_output(args.archive, args.output, records, stats)
        print(f"Output: {args.output.resolve()}")
        return 0
    except (RuntimeError, FileExistsError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
