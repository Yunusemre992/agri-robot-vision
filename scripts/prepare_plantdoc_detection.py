#!/usr/bin/env python3
"""Prepare target PlantDoc Detection leaf classes for the ATOH dataset.

The CSV files are the annotation authority. Pascal VOC XML files are ignored.
PlantDoc disease names are merged into crop-specific healthy/diseased classes:

* 5  apple_leaf_healthy
* 6  apple_leaf_diseased
* 7  tomato_leaf_healthy
* 8  tomato_leaf_diseased
* 9  bell_pepper_leaf_healthy
* 10 bell_pepper_leaf_diseased

``Tomato two spotted spider mites leaf`` is intentionally excluded. The
script performs a dry run by default and writes only with ``--execute``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SPLITS = {
    "train": {"csv": "train_labels.csv", "images": "TRAIN"},
    "test": {"csv": "test_labels.csv", "images": "TEST"},
}
TARGET_NAMES = {
    5: "apple_leaf_healthy",
    6: "apple_leaf_diseased",
    7: "tomato_leaf_healthy",
    8: "tomato_leaf_diseased",
    9: "bell_pepper_leaf_healthy",
    10: "bell_pepper_leaf_diseased",
}
EXPECTED_CLASS_IMAGES = {5: 91, 6: 182, 7: 72, 8: 682, 9: 61, 10: 71}
EXPECTED_CLASS_BOXES = {5: 247, 6: 350, 7: 396, 8: 2522, 9: 323, 10: 264}
EXPECTED_TOTAL_IMAGES = 1147
EXPECTED_TOTAL_BOXES = 4102
EXPECTED_BAD_IMAGES = 1
EXPECTED_SPIDER_MITE_IMAGES = 2
EXPECTED_CROSS_SPLIT_EXCLUSIONS = 3
EXPECTED_CROSS_SPLIT_EXCLUDED_BOXES = 9
REQUIRED_COLUMNS = {
    "filename",
    "width",
    "height",
    "class",
    "xmin",
    "ymin",
    "xmax",
    "ymax",
}
SPIDER_MITE_CLASS = "tomato two spotted spider mites leaf"
EXCLUDED_TEST_DUPLICATES = {
    "tomato_V8.jpg",
    "early-blight-septoria-ls-fig-3.jpg",
    "tylcv-seminar-1-638.jpg",
}


@dataclass(frozen=True)
class ParsedRow:
    source_class: str
    target_id: int
    width: int
    height: int
    xmin: float
    ymin: float
    xmax: float
    ymax: float


@dataclass(frozen=True)
class Record:
    split: str
    source_filename: str
    source_path: Path
    output_image: str
    output_label: str
    label_text: str
    target_ids: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(
        description="Convert PlantDoc CSV boxes to fixed ATOH leaf class IDs."
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=home / "Downloads" / "plantdoc_detection",
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
            / "plantdoc_detection"
        ),
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Write converted files. Without this flag nothing is written.",
    )
    return parser.parse_args()


def normalize_class(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()


def target_for_class(source_class: str) -> int | None:
    normalized = normalize_class(source_class)
    if normalized == "apple leaf":
        return 5
    if normalized.startswith("apple "):
        return 6
    if normalized == "tomato leaf":
        return 7
    if normalized == SPIDER_MITE_CLASS:
        return None
    if normalized.startswith("tomato "):
        return 8
    if normalized == "bell pepper leaf":
        return 9
    if normalized.startswith("bell pepper "):
        return 10
    return None


def inspect_image(path: Path) -> tuple[tuple[int, int], tuple[int, int], int, str]:
    try:
        with Image.open(path) as image:
            raw_size = image.size
            orientation = int(image.getexif().get(274, 1))
            image_format = str(image.format or "").upper()
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as error:
        raise RuntimeError(str(error)) from error

    oriented_size = (
        (raw_size[1], raw_size[0])
        if orientation in {5, 6, 7, 8}
        else raw_size
    )
    extensions = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "BMP": ".bmp",
        "WEBP": ".webp",
    }
    if image_format not in extensions:
        raise RuntimeError(f"unsupported image format: {image_format or 'unknown'}")
    return raw_size, oriented_size, orientation, extensions[image_format]


def safe_output_name(split: str, filename: str, extension: str) -> str:
    base_without_query = Path(filename.split("?", 1)[0]).stem
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", base_without_query).strip("._-")
    if not safe_stem:
        safe_stem = "image"
    safe_stem = safe_stem[:80]
    digest = hashlib.sha1(f"{split}/{filename}".encode("utf-8")).hexdigest()[:10]
    return f"plantdoc_{split}_{safe_stem}_{digest}{extension}"


def parse_csv_row(row: dict[str, str]) -> ParsedRow:
    source_class = str(row["class"])
    target_id = target_for_class(source_class)
    if target_id is None:
        raise ValueError("row is not a selected target class")

    width = int(float(row["width"]))
    height = int(float(row["height"]))
    xmin = float(row["xmin"])
    ymin = float(row["ymin"])
    xmax = float(row["xmax"])
    ymax = float(row["ymax"])
    values = (xmin, ymin, xmax, ymax)
    if width <= 0 or height <= 0 or not all(math.isfinite(v) for v in values):
        raise ValueError("invalid dimensions or coordinates")
    return ParsedRow(
        source_class=source_class,
        target_id=target_id,
        width=width,
        height=height,
        xmin=xmin,
        ymin=ymin,
        xmax=xmax,
        ymax=ymax,
    )


def convert_box(row: ParsedRow) -> tuple[str | None, bool]:
    raw_edges = (row.xmin, row.ymin, row.xmax, row.ymax)
    xmin = max(0.0, min(float(row.width), row.xmin))
    ymin = max(0.0, min(float(row.height), row.ymin))
    xmax = max(0.0, min(float(row.width), row.xmax))
    ymax = max(0.0, min(float(row.height), row.ymax))
    if xmax <= xmin or ymax <= ymin:
        return None, False

    clipped = any(
        abs(raw - clean) > 1e-9
        for raw, clean in zip(raw_edges, (xmin, ymin, xmax, ymax))
    )
    center_x = ((xmin + xmax) / 2.0) / row.width
    center_y = ((ymin + ymax) / 2.0) / row.height
    box_width = (xmax - xmin) / row.width
    box_height = (ymax - ymin) / row.height
    line = (
        f"{row.target_id} {center_x:.8f} {center_y:.8f} "
        f"{box_width:.8f} {box_height:.8f}"
    )
    return line, clipped


def analyze(source: Path) -> tuple[dict[str, Any], list[Record]]:
    stats: dict[str, Any] = {
        "split_stats": {split: {"images": 0, "boxes": 0} for split in SPLITS},
        "total_images": 0,
        "total_boxes": 0,
        "target_class_image_counts": {str(key): 0 for key in TARGET_NAMES},
        "target_class_box_counts": {str(key): 0 for key in TARGET_NAMES},
        "selected_source_classes": {},
        "csv_rows": 0,
        "selected_csv_rows": 0,
        "csv_headers_valid": True,
        "invalid_csv_rows": 0,
        "duplicate_annotation_rows": 0,
        "invalid_boxes": 0,
        "clipped_boxes": 0,
        "missing_images": 0,
        "corrupt_images": 0,
        "dimension_mismatches": 0,
        "inconsistent_csv_dimensions": 0,
        "excluded_bad_images": 0,
        "excluded_bad_boxes": 0,
        "bad_image_details": [],
        "excluded_spider_mite_images": 0,
        "excluded_spider_mite_boxes": 0,
        "excluded_cross_split_duplicate_images": 0,
        "excluded_cross_split_duplicate_boxes": 0,
        "nonstandard_exif_orientations": 0,
        "duplicate_output_names": 0,
        "duplicate_content_groups": 0,
        "duplicate_content_files": 0,
        "cross_split_duplicate_groups": 0,
        "license_first_line": "",
        "errors": [],
    }
    grouped_rows: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    spider_images: set[tuple[str, str]] = set()
    excluded_test_duplicate_images: set[tuple[str, str]] = set()
    annotation_signatures: set[tuple[str, ...]] = set()
    source_class_rows: Counter[str] = Counter()

    license_path = source / "LICENSE.txt"
    if license_path.is_file():
        license_lines = license_path.read_text(errors="replace").splitlines()
        stats["license_first_line"] = license_lines[0].strip() if license_lines else ""
    else:
        stats["errors"].append("missing LICENSE.txt")

    for split, config in SPLITS.items():
        csv_path = source / config["csv"]
        if not csv_path.is_file():
            stats["errors"].append(f"missing CSV: {csv_path}")
            continue

        with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
            reader = csv.DictReader(csv_file)
            if set(reader.fieldnames or []) != REQUIRED_COLUMNS:
                stats["csv_headers_valid"] = False
                continue

            for row in reader:
                stats["csv_rows"] += 1
                filename = str(row.get("filename", "")).strip()
                source_class = str(row.get("class", ""))
                normalized = normalize_class(source_class)
                if normalized == SPIDER_MITE_CLASS:
                    spider_images.add((split, filename))
                    stats["excluded_spider_mite_boxes"] += 1
                    continue

                target_id = target_for_class(source_class)
                if target_id is None:
                    continue

                if split == "test" and filename in EXCLUDED_TEST_DUPLICATES:
                    excluded_test_duplicate_images.add((split, filename))
                    stats["excluded_cross_split_duplicate_boxes"] += 1
                    continue

                stats["selected_csv_rows"] += 1
                source_class_rows[source_class] += 1
                signature = (
                    split,
                    filename,
                    source_class,
                    str(row.get("xmin", "")),
                    str(row.get("ymin", "")),
                    str(row.get("xmax", "")),
                    str(row.get("ymax", "")),
                )
                if signature in annotation_signatures:
                    stats["duplicate_annotation_rows"] += 1
                annotation_signatures.add(signature)
                grouped_rows[(split, filename)].append(row)

    stats["selected_source_classes"] = dict(sorted(source_class_rows.items()))
    stats["excluded_spider_mite_images"] = len(spider_images)
    stats["excluded_cross_split_duplicate_images"] = len(
        excluded_test_duplicate_images
    )

    records: list[Record] = []
    bad_images: set[tuple[str, str]] = set()
    output_names: set[tuple[str, str]] = set()
    content_locations: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for (split, filename), rows in sorted(grouped_rows.items()):
        image_path = source / SPLITS[split]["images"] / filename
        if not image_path.is_file():
            stats["missing_images"] += 1
            bad_images.add((split, filename))
            stats["excluded_bad_boxes"] += len(rows)
            stats["bad_image_details"].append(
                {"split": split, "filename": filename, "reason": "missing"}
            )
            continue

        parsed_rows: list[ParsedRow] = []
        for row in rows:
            try:
                parsed_rows.append(parse_csv_row(row))
            except (KeyError, TypeError, ValueError):
                stats["invalid_csv_rows"] += 1

        if len(parsed_rows) != len(rows):
            continue

        expected_sizes = {(row.width, row.height) for row in parsed_rows}
        if len(expected_sizes) != 1:
            stats["inconsistent_csv_dimensions"] += 1
            bad_images.add((split, filename))
            stats["excluded_bad_boxes"] += len(rows)
            stats["bad_image_details"].append(
                {"split": split, "filename": filename, "reason": "inconsistent_csv_dimensions"}
            )
            continue

        try:
            raw_size, oriented_size, orientation, extension = inspect_image(image_path)
        except RuntimeError as error:
            stats["corrupt_images"] += 1
            bad_images.add((split, filename))
            stats["excluded_bad_boxes"] += len(rows)
            stats["bad_image_details"].append(
                {"split": split, "filename": filename, "reason": f"corrupt: {error}"}
            )
            continue

        expected_size = next(iter(expected_sizes))
        if expected_size not in {raw_size, oriented_size}:
            stats["dimension_mismatches"] += 1
            bad_images.add((split, filename))
            stats["excluded_bad_boxes"] += len(rows)
            stats["bad_image_details"].append(
                {
                    "split": split,
                    "filename": filename,
                    "reason": "dimension_mismatch",
                    "csv": expected_size,
                    "raw": raw_size,
                    "oriented": oriented_size,
                }
            )
            continue
        if orientation != 1:
            stats["nonstandard_exif_orientations"] += 1

        converted_lines: list[str] = []
        target_ids: set[int] = set()
        for parsed_row in parsed_rows:
            output_line, clipped = convert_box(parsed_row)
            if output_line is None:
                stats["invalid_boxes"] += 1
                continue
            converted_lines.append(output_line)
            target_ids.add(parsed_row.target_id)
            stats["target_class_box_counts"][str(parsed_row.target_id)] += 1
            stats["split_stats"][split]["boxes"] += 1
            stats["total_boxes"] += 1
            if clipped:
                stats["clipped_boxes"] += 1

        if not converted_lines:
            continue

        output_image = safe_output_name(split, filename, extension)
        output_label = str(Path(output_image).with_suffix(".txt"))
        output_key = (split, output_image)
        if output_key in output_names:
            stats["duplicate_output_names"] += 1
        output_names.add(output_key)

        for target_id in target_ids:
            stats["target_class_image_counts"][str(target_id)] += 1
        stats["split_stats"][split]["images"] += 1
        stats["total_images"] += 1

        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        content_locations[digest].append((split, filename))
        records.append(
            Record(
                split=split,
                source_filename=filename,
                source_path=image_path,
                output_image=output_image,
                output_label=output_label,
                label_text="\n".join(converted_lines) + "\n",
                target_ids=tuple(sorted(target_ids)),
            )
        )

    stats["excluded_bad_images"] = len(bad_images)
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
        if len({split for split, _ in locations}) > 1
    )
    return stats, records


def is_ready(stats: dict[str, Any]) -> bool:
    return all(
        [
            stats["csv_headers_valid"] is True,
            stats["total_images"] == EXPECTED_TOTAL_IMAGES,
            stats["total_boxes"] == EXPECTED_TOTAL_BOXES,
            stats["target_class_image_counts"]
            == {str(key): value for key, value in EXPECTED_CLASS_IMAGES.items()},
            stats["target_class_box_counts"]
            == {str(key): value for key, value in EXPECTED_CLASS_BOXES.items()},
            stats["excluded_bad_images"] == EXPECTED_BAD_IMAGES,
            stats["excluded_spider_mite_images"] == EXPECTED_SPIDER_MITE_IMAGES,
            stats["excluded_cross_split_duplicate_images"]
            == EXPECTED_CROSS_SPLIT_EXCLUSIONS,
            stats["excluded_cross_split_duplicate_boxes"]
            == EXPECTED_CROSS_SPLIT_EXCLUDED_BOXES,
            stats["invalid_csv_rows"] == 0,
            stats["duplicate_annotation_rows"] == 0,
            stats["invalid_boxes"] == 0,
            stats["clipped_boxes"] == 0,
            stats["duplicate_output_names"] == 0,
            stats["duplicate_content_groups"] == 0,
            stats["cross_split_duplicate_groups"] == 0,
            stats["license_first_line"] == "Attribution 4.0 International",
            not stats["errors"],
        ]
    )


def print_summary(stats: dict[str, Any], mode: str, ready: bool) -> None:
    print(f"Mode: {mode}")
    for split in SPLITS:
        split_stats = stats["split_stats"][split]
        print(
            f"{split}: {split_stats['images']} images, "
            f"{split_stats['boxes']} boxes"
        )
    print(f"Total: {stats['total_images']} images, {stats['total_boxes']} boxes")
    print(f"Class images: {stats['target_class_image_counts']}")
    print(f"Class boxes: {stats['target_class_box_counts']}")
    print(
        "Excluded: "
        f"bad_images={stats['excluded_bad_images']}, "
        f"bad_boxes={stats['excluded_bad_boxes']}, "
        f"spider_mite_images={stats['excluded_spider_mite_images']}, "
        f"test_duplicates={stats['excluded_cross_split_duplicate_images']}"
    )
    print(
        "Annotations: "
        f"clipped={stats['clipped_boxes']}, "
        f"invalid={stats['invalid_boxes']}, "
        f"invalid_csv={stats['invalid_csv_rows']}, "
        f"duplicate_rows={stats['duplicate_annotation_rows']}"
    )
    print(
        "Duplicates: "
        f"groups={stats['duplicate_content_groups']}, "
        f"cross_split={stats['cross_split_duplicate_groups']}"
    )
    print(f"License: CC BY 4.0 ({stats['license_first_line']})")
    print(f"READY: {'yes' if ready else 'no'}")


def write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def write_output(
    source: Path,
    output: Path,
    records: list[Record],
    stats: dict[str, Any],
) -> None:
    temporary_output = output.with_name(output.name + ".part")
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")
    if temporary_output.exists():
        raise FileExistsError(f"Temporary output already exists: {temporary_output}")

    index_buffer = io.StringIO()
    index_writer = csv.DictWriter(
        index_buffer,
        fieldnames=["split", "output_image", "source_filename", "target_ids"],
    )
    index_writer.writeheader()

    for record in records:
        image_destination = (
            temporary_output / record.split / "images" / record.output_image
        )
        label_destination = (
            temporary_output / record.split / "labels" / record.output_label
        )
        write_bytes(image_destination, record.source_path.read_bytes())
        write_bytes(label_destination, record.label_text.encode("utf-8"))
        index_writer.writerow(
            {
                "split": record.split,
                "output_image": record.output_image,
                "source_filename": record.source_filename,
                "target_ids": "|".join(map(str, record.target_ids)),
            }
        )

    manifest = {
        "source": "PlantDoc Detection",
        "source_directory": str(source.resolve()),
        "annotation_authority": "train_labels.csv and test_labels.csv",
        "ignored_annotations": "Pascal VOC XML files",
        "target_task": "YOLO object detection",
        "target_classes": {str(key): value for key, value in TARGET_NAMES.items()},
        "mapping_policy": (
            "Healthy crop leaf names map to healthy IDs; all other selected "
            "crop disease names merge into the matching diseased ID."
        ),
        "excluded_class": "Tomato two spotted spider mites leaf",
        "excluded_test_duplicates": sorted(EXCLUDED_TEST_DUPLICATES),
        "split_policy": "Source train/test splits are preserved.",
        "license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
    }
    write_bytes(
        temporary_output / "source_manifest.json",
        (json.dumps(manifest, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    write_bytes(
        temporary_output / "conversion_report.json",
        (json.dumps(stats, indent=2, ensure_ascii=False) + "\n").encode("utf-8"),
    )
    write_bytes(
        temporary_output / "source_index.csv",
        index_buffer.getvalue().encode("utf-8"),
    )
    write_bytes(
        temporary_output / "SOURCE_LICENSE.txt",
        (source / "LICENSE.txt").read_bytes(),
    )
    temporary_output.replace(output)


def main() -> int:
    args = parse_args()
    if not args.source.is_dir():
        print(f"ERROR: source directory not found: {args.source}", file=sys.stderr)
        return 1

    try:
        stats, records = analyze(args.source)
        ready = is_ready(stats)
        mode = "execute" if args.execute else "dry-run"
        print_summary(stats, mode, ready)

        if not ready:
            for error in stats["errors"][:10]:
                print(f"ERROR DETAIL: {error}", file=sys.stderr)
            print("No files were written because the audit did not pass.", file=sys.stderr)
            return 2

        if not args.execute:
            print("No files were written.")
            return 0

        write_output(args.source, args.output, records, stats)
        print(f"Output: {args.output.resolve()}")
        return 0
    except (RuntimeError, FileExistsError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
