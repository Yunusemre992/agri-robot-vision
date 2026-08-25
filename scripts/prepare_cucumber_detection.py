#!/usr/bin/env python3
"""Prepare the audited Roboflow cucumber dataset for the 15-class project.

Dry-run is the default. Pass --execute to write the staging dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import shutil
import sys
import tempfile
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path, PurePosixPath


PROJECT_CLASSES = {
    0: "apple",
    1: "tomato",
    2: "bell_pepper",
    3: "cucumber",
    4: "goji_berry",
    5: "apple_leaf_healthy",
    6: "apple_leaf_diseased",
    7: "tomato_leaf_healthy",
    8: "tomato_leaf_diseased",
    9: "bell_pepper_leaf_healthy",
    10: "bell_pepper_leaf_diseased",
    11: "cucumber_leaf_healthy",
    12: "cucumber_leaf_diseased",
    13: "goji_berry_leaf_healthy",
    14: "goji_berry_leaf_diseased",
}

SOURCE_CLASSES = {
    0: "Anthracnose",
    1: "Bacterial-Wilt",
    2: "Belly-Rot",
    3: "Downy-Mildew",
    4: "Fresh-Cucumber",
    5: "Fresh-Leaf",
    6: "Gummy-Stem-Blight",
    7: "Pythium-Fruit-Rot",
}

# Bacterial-Wilt (source class 1) is intentionally omitted because its boxes
# surround wilted whole plants/clusters rather than individual leaves.
CLASS_MAP = {
    0: 12,  # Anthracnose -> cucumber_leaf_diseased
    2: 3,   # Belly-Rot -> cucumber
    3: 12,  # Downy-Mildew -> cucumber_leaf_diseased
    4: 3,   # Fresh-Cucumber -> cucumber
    5: 11,  # Fresh-Leaf -> cucumber_leaf_healthy
    6: 12,  # Gummy-Stem-Blight -> cucumber_leaf_diseased
    7: 3,   # Pythium-Fruit-Rot -> cucumber
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SOURCE_URL = "https://universe.roboflow.com/thomas-j6iuv/cucumber_disease-6/dataset/2"
ORIGINAL_DATASET_URL = "https://data.mendeley.com/datasets/y6d3z6f8z9/1"
LICENSE = "CC BY 4.0 (Attribution 4.0 International)"


@dataclass(frozen=True)
class Box:
    source_class: int
    target_class: int
    x_center: float
    y_center: float
    width: float
    height: float

    def to_yolo(self) -> str:
        return (
            f"{self.target_class} {self.x_center:.10f} {self.y_center:.10f} "
            f"{self.width:.10f} {self.height:.10f}"
        )


@dataclass
class Record:
    split: str
    image_member: str
    label_member: str
    output_name: str
    image_bytes: bytes
    boxes: list[Box]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path.home() / "Downloads" / "Cucumber_disease 6.v2i.yolo26.zip",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path.home()
            / "projects"
            / "atoh_agriculture"
            / "datasets"
            / "staging"
            / "cucumber_detection"
        ),
    )
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def normalized_split(member: str) -> str | None:
    parts = [part.lower() for part in PurePosixPath(member).parts]
    if "train" in parts:
        return "train"
    if "valid" in parts or "val" in parts:
        return "valid"
    if "test" in parts:
        return "test"
    return None


def paired_label(image_member: str, members_by_lower: dict[str, str]) -> str | None:
    normalized = image_member.replace("\\", "/")
    lowered = normalized.lower()
    marker = "/images/"
    if marker not in lowered:
        return None
    index = lowered.index(marker)
    candidate = normalized[:index] + "/labels/" + normalized[index + len(marker):]
    candidate = str(PurePosixPath(candidate).with_suffix(".txt"))
    return members_by_lower.get(candidate.lower())


def parse_boxes(text: str, label_member: str) -> tuple[list[Box], int, list[str]]:
    boxes: list[Box] = []
    excluded_boxes = 0
    errors: list[str] = []

    for line_number, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_member}:{line_number}: expected 5 fields")
            continue
        try:
            raw_class = float(parts[0])
            source_class = int(raw_class)
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"{label_member}:{line_number}: non-numeric value")
            continue
        if raw_class != source_class or source_class not in SOURCE_CLASSES:
            errors.append(f"{label_member}:{line_number}: invalid class {parts[0]}")
            continue
        coordinates = (x_center, y_center, width, height)
        if not all(math.isfinite(value) for value in coordinates):
            errors.append(f"{label_member}:{line_number}: non-finite coordinate")
            continue
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        if (
            not 0 <= x_center <= 1
            or not 0 <= y_center <= 1
            or not 0 < width <= 1
            or not 0 < height <= 1
            or x1 < 0
            or y1 < 0
            or x2 > 1
            or y2 > 1
        ):
            errors.append(f"{label_member}:{line_number}: box outside image")
            continue
        if source_class not in CLASS_MAP:
            excluded_boxes += 1
            continue
        boxes.append(
            Box(
                source_class=source_class,
                target_class=CLASS_MAP[source_class],
                x_center=x_center,
                y_center=y_center,
                width=width,
                height=height,
            )
        )
    return boxes, excluded_boxes, errors


def safe_output_name(member: str) -> str:
    name = PurePosixPath(member).name
    return name.replace(" ", "_")


def build_records(source: Path) -> tuple[list[Record], dict[str, object]]:
    statistics: dict[str, object] = {
        "errors": [],
        "source_images": Counter(),
        "source_boxes": Counter(),
        "excluded_images": Counter(),
        "excluded_boxes": Counter(),
        "duplicate_images": [],
        "cross_split_duplicates": [],
    }

    with zipfile.ZipFile(source) as archive:
        corrupt_member = archive.testzip()
        if corrupt_member:
            raise RuntimeError(f"Corrupt ZIP member: {corrupt_member}")

        members = [item.filename for item in archive.infolist() if not item.is_dir()]
        members_by_lower = {member.lower(): member for member in members}
        image_members = [
            member
            for member in members
            if PurePosixPath(member).suffix.lower() in IMAGE_EXTENSIONS
        ]

        provisional: list[Record] = []
        output_keys: set[tuple[str, str]] = set()

        for image_member in image_members:
            split = normalized_split(image_member)
            if split is None:
                cast_errors = statistics["errors"]
                assert isinstance(cast_errors, list)
                cast_errors.append(f"Unknown split: {image_member}")
                continue
            label_member = paired_label(image_member, members_by_lower)
            if label_member is None:
                cast_errors = statistics["errors"]
                assert isinstance(cast_errors, list)
                cast_errors.append(f"Missing label: {image_member}")
                continue

            image_bytes = archive.read(image_member)
            label_text = archive.read(label_member).decode("utf-8", errors="replace")
            boxes, excluded_count, box_errors = parse_boxes(label_text, label_member)

            cast_errors = statistics["errors"]
            assert isinstance(cast_errors, list)
            cast_errors.extend(box_errors)

            source_class_ids = {
                int(line.split()[0])
                for line in label_text.splitlines()
                if len(line.split()) == 5 and line.split()[0].lstrip("-").isdigit()
            }
            source_images = statistics["source_images"]
            assert isinstance(source_images, Counter)
            for class_id in source_class_ids:
                source_images[class_id] += 1

            source_boxes = statistics["source_boxes"]
            assert isinstance(source_boxes, Counter)
            for box in boxes:
                source_boxes[box.source_class] += 1

            if excluded_count:
                excluded_boxes = statistics["excluded_boxes"]
                assert isinstance(excluded_boxes, Counter)
                excluded_boxes[1] += excluded_count

            if not boxes:
                excluded_images = statistics["excluded_images"]
                assert isinstance(excluded_images, Counter)
                for class_id in source_class_ids:
                    excluded_images[class_id] += 1
                continue

            output_name = safe_output_name(image_member)
            output_key = (split, output_name.lower())
            if output_key in output_keys:
                cast_errors.append(f"Output filename collision: {split}/{output_name}")
                continue
            output_keys.add(output_key)
            provisional.append(
                Record(
                    split=split,
                    image_member=image_member,
                    label_member=label_member,
                    output_name=output_name,
                    image_bytes=image_bytes,
                    boxes=boxes,
                )
            )

    # Prefer test, then validation, then train if an exact duplicate ever spans splits.
    split_priority = {"test": 0, "valid": 1, "train": 2}
    provisional.sort(key=lambda record: (split_priority[record.split], record.image_member))
    hashes: dict[str, Record] = {}
    records: list[Record] = []

    for record in provisional:
        digest = hashlib.sha256(record.image_bytes).hexdigest()
        if digest in hashes:
            kept = hashes[digest]
            duplicate_images = statistics["duplicate_images"]
            assert isinstance(duplicate_images, list)
            duplicate_images.append((kept.image_member, record.image_member))
            if kept.split != record.split:
                cross_split = statistics["cross_split_duplicates"]
                assert isinstance(cross_split, list)
                cross_split.append((kept.image_member, record.image_member))
            continue
        hashes[digest] = record
        records.append(record)

    records.sort(key=lambda record: (record.split, record.output_name.lower()))
    return records, statistics


def data_yaml() -> str:
    names = "\n".join(f"  {class_id}: {name}" for class_id, name in PROJECT_CLASSES.items())
    return (
        "path: .\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n"
        "names:\n"
        f"{names}\n"
    )


def source_readme() -> str:
    return f"""Cucumber detection staging source

Annotated source: {SOURCE_URL}
Original image dataset: {ORIGINAL_DATASET_URL}
License: {LICENSE}

Accepted source-class mapping:
0 Anthracnose -> 12 cucumber_leaf_diseased
2 Belly-Rot -> 3 cucumber
3 Downy-Mildew -> 12 cucumber_leaf_diseased
4 Fresh-Cucumber -> 3 cucumber
5 Fresh-Leaf -> 11 cucumber_leaf_healthy
6 Gummy-Stem-Blight -> 12 cucumber_leaf_diseased
7 Pythium-Fruit-Rot -> 3 cucumber

Excluded:
1 Bacterial-Wilt: annotations surround whole wilted plants/clusters rather than
individual leaves, so they are incompatible with cucumber_leaf_diseased.

The Roboflow archive is an annotated/augmented derivative of the Mendeley
dataset. Do not merge the unannotated original ZIP into training, because that
would duplicate source images.
"""


def print_summary(records: list[Record], statistics: dict[str, object], execute: bool, output: Path) -> bool:
    split_images = Counter(record.split for record in records)
    split_boxes = Counter()
    target_boxes = Counter()
    target_images: dict[int, set[tuple[str, str]]] = defaultdict(set)
    source_boxes = Counter()

    for record in records:
        split_boxes[record.split] += len(record.boxes)
        for box in record.boxes:
            target_boxes[box.target_class] += 1
            source_boxes[box.source_class] += 1
            target_images[box.target_class].add((record.split, record.output_name))

    errors = statistics["errors"]
    duplicate_images = statistics["duplicate_images"]
    cross_split = statistics["cross_split_duplicates"]
    excluded_images = statistics["excluded_images"]
    excluded_boxes = statistics["excluded_boxes"]
    assert isinstance(errors, list)
    assert isinstance(duplicate_images, list)
    assert isinstance(cross_split, list)
    assert isinstance(excluded_images, Counter)
    assert isinstance(excluded_boxes, Counter)

    print(f"Mode: {'execute' if execute else 'dry-run'}")
    for split in ("train", "valid", "test"):
        print(f"{split}: {split_images[split]} images, {split_boxes[split]} boxes")
    print(f"Total: {len(records)} images, {sum(split_boxes.values())} boxes")
    print(
        "Class images:",
        {str(class_id): len(target_images[class_id]) for class_id in sorted(target_images)},
    )
    print(
        "Class boxes:",
        {str(class_id): target_boxes[class_id] for class_id in sorted(target_boxes)},
    )
    print(
        "Source boxes used:",
        {
            f"{class_id}:{SOURCE_CLASSES[class_id]}": source_boxes[class_id]
            for class_id in sorted(source_boxes)
        },
    )
    print(
        "Excluded:",
        f"bacterial_wilt_images={excluded_images[1]}, "
        f"bacterial_wilt_boxes={excluded_boxes[1]}, "
        f"exact_duplicate_images={len(duplicate_images)}",
    )
    print(
        "Duplicates:",
        f"groups_removed={len(duplicate_images)}, cross_split={len(cross_split)}",
    )
    print(f"Errors: {len(errors)}")
    if errors:
        for error in errors[:20]:
            print(f"  {error}")
    print(f"License: {LICENSE}")
    ready = not errors and not cross_split and bool(records)
    print(f"READY: {'yes' if ready else 'no'}")
    if execute and ready:
        print(f"Output: {output}")
    elif not execute:
        print("No files were written.")
    return ready


def write_dataset(records: list[Record], output: Path) -> None:
    output_parent = output.parent
    output_parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(
            f"Output already exists; it was not overwritten: {output}"
        )

    temporary = Path(tempfile.mkdtemp(prefix=".cucumber_detection_", dir=output_parent))
    try:
        for split in ("train", "valid", "test"):
            (temporary / split / "images").mkdir(parents=True, exist_ok=True)
            (temporary / split / "labels").mkdir(parents=True, exist_ok=True)

        for record in records:
            image_path = temporary / record.split / "images" / record.output_name
            label_path = temporary / record.split / "labels" / Path(record.output_name).with_suffix(".txt")
            image_path.write_bytes(record.image_bytes)
            label_path.write_text(
                "\n".join(box.to_yolo() for box in record.boxes) + "\n",
                encoding="utf-8",
            )

        (temporary / "data.yaml").write_text(data_yaml(), encoding="utf-8")
        (temporary / "SOURCE.md").write_text(source_readme(), encoding="utf-8")
        temporary.rename(output)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    args = parse_args()
    source = args.source.expanduser().resolve()
    output = args.output.expanduser().resolve()

    if not source.is_file():
        print(f"ERROR: Source ZIP not found: {source}", file=sys.stderr)
        return 2
    if not zipfile.is_zipfile(source):
        print(f"ERROR: Source is not a valid ZIP: {source}", file=sys.stderr)
        return 2

    try:
        records, statistics = build_records(source)
        ready = print_summary(records, statistics, args.execute, output)
        if not ready:
            return 1
        if args.execute:
            write_dataset(records, output)
            # Reprint the destination after the atomic write succeeds.
            print(f"Written successfully: {output}")
        return 0
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
