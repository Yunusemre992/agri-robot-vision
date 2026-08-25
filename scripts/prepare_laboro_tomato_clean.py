#!/usr/bin/env python3
"""Prepare Laboro Tomato for the ATOH YOLO detection dataset.

Source annotations: bundled YOLO TXT files with six ripeness classes.
Target annotation: one ``tomato`` class with fixed ATOH class ID 1.

The script performs a dry run by default and prints a compact audit. Use
``--execute`` only after the audit reports ``READY: yes``.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image, UnidentifiedImageError


TARGET_CLASS_ID = 1
TARGET_CLASS_NAME = "tomato"
SOURCE_CLASS_IDS = set(range(6))
EXPECTED_CATEGORY_NAMES = {
    "b_fully_ripened",
    "b_half_ripened",
    "b_green",
    "l_fully_ripened",
    "l_half_ripened",
    "l_green",
}

SPLITS = {
    "train": {
        "json": "annotations/train.json",
        "images": "train/images/",
        "labels": "train/labels/",
    },
    # The source archive stores official test.json files under val/.
    "test": {
        "json": "annotations/test.json",
        "images": "val/images/",
        "labels": "val/labels/",
    },
}


@dataclass(frozen=True)
class Record:
    split: str
    source_image: str
    output_image: str
    output_label: str
    label_text: str


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Merge Laboro Tomato ripeness classes into ATOH tomato class 1."
    )
    parser.add_argument(
        "--archive",
        type=Path,
        default=home / "Downloads" / "labora_tomato.zip",
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
            / "laboro_tomato"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write converted files. Without this flag nothing is written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing Laboro staging output.",
    )
    return parser.parse_args()


def read_coco(archive: zipfile.ZipFile, member: str) -> dict[str, Any]:
    try:
        data = json.loads(archive.read(member))
    except KeyError as error:
        raise RuntimeError(f"Missing JSON: {member}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"Invalid JSON: {member}: {error}") from error
    if not isinstance(data, dict):
        raise RuntimeError(f"JSON root is not an object: {member}")
    return data


def inspect_image(data: bytes) -> tuple[tuple[int, int], tuple[int, int], int]:
    try:
        with Image.open(BytesIO(data)) as image:
            raw_size = image.size
            orientation = int(image.getexif().get(274, 1))
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error

    if orientation in {5, 6, 7, 8}:
        oriented_size = (raw_size[1], raw_size[0])
    else:
        oriented_size = raw_size
    return raw_size, oriented_size, orientation


def convert_label_line(source_line: str) -> tuple[str | None, bool, str | None]:
    parts = source_line.split()
    if len(parts) != 5:
        return None, False, "malformed"

    try:
        source_class = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])
    except ValueError:
        return None, False, "malformed"

    if source_class not in SOURCE_CLASS_IDS:
        return None, False, "unknown_class"
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
        f"{TARGET_CLASS_ID} {output_center_x:.8f} {output_center_y:.8f} "
        f"{output_width:.8f} {output_height:.8f}"
    )
    return output_line, clipped, None


def make_output_names(split: str, filename: str) -> tuple[str, str]:
    basename = PurePosixPath(filename).name
    image_name = f"laboro_tomato_{split}_{basename}"
    label_name = str(Path(image_name).with_suffix(".txt"))
    return image_name, label_name


def analyze(
    archive_path: Path,
) -> tuple[dict[str, Any], list[Record]]:
    stats: dict[str, Any] = {
        "train_images": 0,
        "train_boxes": 0,
        "test_images": 0,
        "test_boxes": 0,
        "total_images": 0,
        "total_boxes": 0,
        "json_annotations": 0,
        "raw_dimension_matches": 0,
        "exif_dimension_matches": 0,
        "dimension_unresolved": 0,
        "orientation_1": 0,
        "orientation_6": 0,
        "other_orientations": 0,
        "clipped_boxes": 0,
        "invalid_boxes": 0,
        "malformed_lines": 0,
        "unknown_source_classes": 0,
        "missing_images": 0,
        "missing_labels": 0,
        "corrupt_images": 0,
        "categories_match": True,
        "duplicate_image_ids": 0,
        "invalid_json_annotations": 0,
        "json_label_difference": 0,
        "errors": [],
    }
    records: list[Record] = []

    with zipfile.ZipFile(archive_path) as archive:
        members = set(archive.namelist())

        for split, config in SPLITS.items():
            coco = read_coco(archive, config["json"])
            images = coco.get("images", [])
            annotations = coco.get("annotations", [])
            categories = coco.get("categories", [])
            if not all(isinstance(value, list) for value in (images, annotations, categories)):
                raise RuntimeError(f"Invalid COCO structure: {config['json']}")

            category_names = {str(item.get("name", "")) for item in categories}
            category_ids = {int(item["id"]) for item in categories}
            if category_names != EXPECTED_CATEGORY_NAMES:
                stats["categories_match"] = False

            image_ids = [int(item["id"]) for item in images]
            valid_image_ids = set(image_ids)
            stats["duplicate_image_ids"] += len(image_ids) - len(valid_image_ids)
            for annotation in annotations:
                try:
                    image_id = int(annotation["image_id"])
                    category_id = int(annotation["category_id"])
                except (KeyError, TypeError, ValueError):
                    stats["invalid_json_annotations"] += 1
                    continue
                if image_id not in valid_image_ids or category_id not in category_ids:
                    stats["invalid_json_annotations"] += 1

            stats["json_annotations"] += len(annotations)
            split_images = 0
            split_boxes = 0

            for info in images:
                source_filename = PurePosixPath(str(info.get("file_name", ""))).name
                if not source_filename:
                    stats["missing_images"] += 1
                    stats["errors"].append(f"{split}: JSON image entry has no filename")
                    continue

                image_member = config["images"] + source_filename
                label_member = config["labels"] + str(
                    Path(source_filename).with_suffix(".txt")
                )
                if image_member not in members:
                    stats["missing_images"] += 1
                    stats["errors"].append(f"missing image: {image_member}")
                    continue
                if label_member not in members:
                    stats["missing_labels"] += 1
                    stats["errors"].append(f"missing label: {label_member}")
                    continue

                try:
                    raw_size, oriented_size, orientation = inspect_image(
                        archive.read(image_member)
                    )
                except RuntimeError as error:
                    stats["corrupt_images"] += 1
                    stats["errors"].append(f"corrupt image: {image_member}: {error}")
                    continue

                if orientation == 1:
                    stats["orientation_1"] += 1
                elif orientation == 6:
                    stats["orientation_6"] += 1
                else:
                    stats["other_orientations"] += 1

                expected_size = (int(info["width"]), int(info["height"]))
                if raw_size == expected_size:
                    stats["raw_dimension_matches"] += 1
                elif oriented_size == expected_size:
                    stats["exif_dimension_matches"] += 1
                else:
                    stats["dimension_unresolved"] += 1
                    stats["errors"].append(
                        f"dimension mismatch: {image_member}: raw={raw_size}, "
                        f"oriented={oriented_size}, json={expected_size}"
                    )

                source_text = archive.read(label_member).decode(
                    "utf-8", errors="replace"
                ).strip()
                converted_lines: list[str] = []
                for source_line in source_text.splitlines() if source_text else []:
                    output_line, clipped, error_kind = convert_label_line(source_line)
                    if error_kind == "malformed":
                        stats["malformed_lines"] += 1
                        continue
                    if error_kind == "unknown_class":
                        stats["unknown_source_classes"] += 1
                        continue
                    if error_kind == "invalid_box":
                        stats["invalid_boxes"] += 1
                        continue
                    if output_line is None:
                        stats["invalid_boxes"] += 1
                        continue
                    converted_lines.append(output_line)
                    split_boxes += 1
                    if clipped:
                        stats["clipped_boxes"] += 1

                output_image, output_label = make_output_names(split, source_filename)
                label_text = "\n".join(converted_lines)
                if label_text:
                    label_text += "\n"
                records.append(
                    Record(
                        split=split,
                        source_image=image_member,
                        output_image=output_image,
                        output_label=output_label,
                        label_text=label_text,
                    )
                )
                split_images += 1

            stats[f"{split}_images"] = split_images
            stats[f"{split}_boxes"] = split_boxes

    stats["total_images"] = stats["train_images"] + stats["test_images"]
    stats["total_boxes"] = stats["train_boxes"] + stats["test_boxes"]
    stats["json_label_difference"] = stats["json_annotations"] - stats["total_boxes"]
    return stats, records


def is_ready(stats: dict[str, Any]) -> bool:
    return all(
        [
            stats["train_images"] == 643,
            stats["test_images"] == 161,
            stats["total_images"] == 804,
            stats["json_annotations"] == 9777,
            stats["total_boxes"] == 9777,
            stats["clipped_boxes"] == 60,
            stats["invalid_boxes"] == 0,
            stats["malformed_lines"] == 0,
            stats["unknown_source_classes"] == 0,
            stats["missing_images"] == 0,
            stats["missing_labels"] == 0,
            stats["corrupt_images"] == 0,
            stats["dimension_unresolved"] == 0,
            stats["categories_match"] is True,
            stats["duplicate_image_ids"] == 0,
            stats["invalid_json_annotations"] == 0,
            stats["json_label_difference"] == 0,
        ]
    )


def print_summary(stats: dict[str, Any], mode: str, ready: bool) -> None:
    print(f"Mode: {mode}")
    print(f"Train: {stats['train_images']} images, {stats['train_boxes']} boxes")
    print(f"Test: {stats['test_images']} images, {stats['test_boxes']} boxes")
    print(f"Total: {stats['total_images']} images, {stats['total_boxes']} boxes")
    print(
        "Dimensions: "
        f"raw={stats['raw_dimension_matches']}, "
        f"exif={stats['exif_dimension_matches']}, "
        f"unresolved={stats['dimension_unresolved']}"
    )
    print(
        "Boxes: "
        f"clipped={stats['clipped_boxes']}, "
        f"invalid={stats['invalid_boxes']}, "
        f"malformed={stats['malformed_lines']}"
    )
    print(
        "Files: "
        f"missing_images={stats['missing_images']}, "
        f"missing_labels={stats['missing_labels']}, "
        f"corrupt_images={stats['corrupt_images']}"
    )
    print(f"JSON/label difference: {stats['json_label_difference']}")
    print(
        "JSON integrity: "
        f"duplicate_images={stats['duplicate_image_ids']}, "
        f"invalid_annotations={stats['invalid_json_annotations']}"
    )
    print(f"READY: {'yes' if ready else 'no'}")


def atomic_write_bytes(path: Path, data: bytes, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".part")
    temporary.write_bytes(data)
    temporary.replace(path)


def write_output(
    archive_path: Path,
    output: Path,
    records: list[Record],
    stats: dict[str, Any],
    overwrite: bool,
) -> None:
    if output.exists() and any(output.rglob("*")) and not overwrite:
        raise FileExistsError(
            f"Output directory is not empty: {output}. Use --overwrite only after review."
        )

    with zipfile.ZipFile(archive_path) as archive:
        for record in records:
            image_destination = output / record.split / "images" / record.output_image
            label_destination = output / record.split / "labels" / record.output_label
            atomic_write_bytes(
                image_destination,
                archive.read(record.source_image),
                overwrite,
            )
            atomic_write_bytes(
                label_destination,
                record.label_text.encode("utf-8"),
                overwrite,
            )

    manifest = {
        "source": "Laboro Tomato",
        "source_archive": str(archive_path.resolve()),
        "target_task": "YOLO object detection",
        "target_class": {"id": TARGET_CLASS_ID, "name": TARGET_CLASS_NAME},
        "annotation_authority": "Bundled YOLO TXT labels",
        "mapping": "All six source ripeness classes -> tomato (class 1)",
        "split_policy": {
            "train": "source train split",
            "test": "source test.json paired with val images/labels",
        },
        "image_policy": (
            "Original JPEG bytes and EXIF orientation are preserved. "
            "YOLO labels are normalized to the EXIF-oriented view."
        ),
        "license": {
            "status": "verify against the official source page",
            "project_rule": (
                "Technical preparation may proceed for the academic/internship prototype; "
                "reassess redistribution and commercial use separately."
            ),
        },
    }
    atomic_write_bytes(
        output / "source_manifest.json",
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        overwrite,
    )
    atomic_write_bytes(
        output / "conversion_report.json",
        (json.dumps(stats, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
        overwrite,
    )


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
            print("No files were written because the audit did not pass.", file=sys.stderr)
            return 2

        if not args.execute:
            print("No files were written.")
            return 0

        write_output(args.archive, args.output, records, stats, args.overwrite)
        print(f"Output: {args.output.resolve()}")
        return 0
    except (RuntimeError, FileExistsError, zipfile.BadZipFile) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
