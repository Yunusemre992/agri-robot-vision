#!/usr/bin/env python3
"""Build the final ATOH YOLO detection dataset from accepted staging sources.

The builder is intentionally non-destructive:
  * it never changes staging files;
  * it refuses to overwrite an existing output directory;
  * it writes into a temporary directory and renames it only after verification.

Expected project layout:
  datasets/staging/{minneapple,laboro_tomato,kaggle_gpt,
                    plantdoc_detection,cucumber_detection}

Usage:
  python3 scripts/build_final_detection_dataset.py \
      --project "$HOME/projects/atoh_agriculture"
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import math
import os
import random
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
SPLITS = ("train", "valid", "test")

# Staging used the original 15-class design. Classes without any accepted data
# are removed from v1 and reserved for a later dataset version.
STAGING_NAMES = {
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

ACTIVE_REMAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    5: 4,
    6: 5,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
    11: 10,
    12: 11,
}

ACTIVE_NAMES = {
    new: STAGING_NAMES[old]
    for old, new in sorted(ACTIVE_REMAP.items(), key=lambda item: item[1])
}

RESERVED_NAMES = {
    4: "goji_berry",
    13: "goji_berry_leaf_healthy",
    14: "goji_berry_leaf_diseased",
}


@dataclass(frozen=True)
class SourceConfig:
    key: str
    expected_staging_classes: frozenset[int]
    needs_valid_from_train: bool
    title: str
    url: str
    license_text: str
    notes: str


SOURCE_CONFIGS = (
    SourceConfig(
        key="minneapple",
        expected_staging_classes=frozenset({0}),
        needs_valid_from_train=True,
        title="MinneApple",
        url="https://rsn.cs.umn.edu/index.php/MinneApple",
        license_text="CC BY-NC-SA 3.0 US",
        notes="Orchard apple detection source; original masks converted to bounding boxes.",
    ),
    SourceConfig(
        key="laboro_tomato",
        expected_staging_classes=frozenset({1}),
        needs_valid_from_train=True,
        title="Laboro Tomato",
        url="https://github.com/laboroai/LaboroTomato",
        license_text="CC BY-NC-SA 4.0",
        notes="Greenhouse tomato instance annotations converted to bounding boxes.",
    ),
    SourceConfig(
        key="kaggle_gpt",
        expected_staging_classes=frozenset({0, 2}),
        needs_valid_from_train=False,
        title="deepNIR apple and capsicum subset",
        url="https://zenodo.org/records/6324489",
        license_text="NOT VERIFIED on the official Zenodo record",
        notes=(
            "The staged subset was proven byte-identical, image and label, to the "
            "official deepNIR apple/capsicum subset (185/185). The staging folder "
            "keeps its historical name kaggle_gpt for reproducibility."
        ),
    ),
    SourceConfig(
        key="plantdoc_detection",
        expected_staging_classes=frozenset({5, 6, 7, 8, 9, 10}),
        needs_valid_from_train=True,
        title="PlantDoc detection subset",
        url="https://github.com/pratikkayal/PlantDoc-Dataset",
        license_text="See copied PlantDoc SOURCE_LICENSE.txt; verify before redistribution",
        notes="Apple, tomato and bell-pepper healthy/diseased leaf boxes.",
    ),
    SourceConfig(
        key="cucumber_detection",
        expected_staging_classes=frozenset({3, 11, 12}),
        needs_valid_from_train=False,
        title="Cucumber detection merged subset",
        url="See copied staging SOURCE.md",
        license_text="CC BY 4.0 according to staging SOURCE.md",
        notes="Cucumber fruit plus healthy/diseased cucumber leaf boxes.",
    ),
)


@dataclass
class Record:
    source: str
    source_split: str
    image_path: Path
    label_path: Path
    rows_old: list[tuple[int, float, float, float, float]]
    rows_new: list[tuple[int, float, float, float, float]]
    sha256: str = ""
    final_split: str = ""
    output_image_name: str = ""
    output_label_name: str = ""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_signature(rows: list[tuple[int, float, float, float, float]]) -> tuple:
    return tuple(sorted((cls, round(x, 9), round(y, 9), round(w, 9), round(h, 9)) for cls, x, y, w, h in rows))


def parse_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    rows = []
    errors = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            errors.append(f"{path}:{line_number}: expected 5 fields, got {len(fields)}")
            continue
        try:
            cls_float = float(fields[0])
            cls = int(cls_float)
            if cls_float != cls:
                raise ValueError("non-integer class")
            x, y, w, h = map(float, fields[1:])
        except ValueError as exc:
            errors.append(f"{path}:{line_number}: numeric parse error ({exc})")
            continue
        if not all(math.isfinite(value) for value in (x, y, w, h)):
            errors.append(f"{path}:{line_number}: non-finite coordinate")
            continue
        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
            errors.append(f"{path}:{line_number}: invalid normalized bbox")
            continue
        if cls not in ACTIVE_REMAP:
            errors.append(
                f"{path}:{line_number}: class {cls} ({STAGING_NAMES.get(cls, 'unknown')}) "
                "has no accepted v1 data/remap"
            )
            continue
        rows.append((cls, x, y, w, h))
    return rows, errors


def collect_records(staging: Path) -> tuple[list[Record], list[str]]:
    records = []
    errors = []
    for config in SOURCE_CONFIGS:
        source_root = staging / config.key
        if not source_root.is_dir():
            errors.append(f"Missing source directory: {source_root}")
            continue
        for split in SPLITS:
            image_dir = source_root / split / "images"
            label_dir = source_root / split / "labels"
            if not image_dir.exists() and not label_dir.exists():
                continue
            if not image_dir.is_dir() or not label_dir.is_dir():
                errors.append(f"Incomplete image/label directories: {source_root / split}")
                continue
            images = {
                path.stem: path
                for path in image_dir.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
            }
            labels = {
                path.stem: path
                for path in label_dir.iterdir()
                if path.is_file() and path.suffix.lower() == ".txt"
            }
            for stem in sorted(images.keys() - labels.keys()):
                errors.append(f"Image without label: {images[stem]}")
            for stem in sorted(labels.keys() - images.keys()):
                errors.append(f"Label without image: {labels[stem]}")
            for stem in sorted(images.keys() & labels.keys()):
                rows_old, label_errors = parse_label(labels[stem])
                errors.extend(label_errors)
                found_classes = {row[0] for row in rows_old}
                unexpected = found_classes - config.expected_staging_classes
                if unexpected:
                    errors.append(
                        f"Unexpected classes in {labels[stem]}: {sorted(unexpected)}; "
                        f"expected {sorted(config.expected_staging_classes)}"
                    )
                rows_new = [
                    (ACTIVE_REMAP[cls], x, y, w, h)
                    for cls, x, y, w, h in rows_old
                ]
                records.append(
                    Record(
                        source=config.key,
                        source_split=split,
                        image_path=images[stem],
                        label_path=labels[stem],
                        rows_old=rows_old,
                        rows_new=rows_new,
                    )
                )
    return records, errors


def choose_and_deduplicate(records: list[Record]) -> tuple[list[Record], list[list[Record]], list[str]]:
    print(f"Hashing {len(records)} staged images...", flush=True)
    for index, record in enumerate(records, 1):
        record.sha256 = sha256_file(record.image_path)
        if index % 500 == 0 or index == len(records):
            print(f"  {index}/{len(records)}", flush=True)

    groups: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        groups[record.sha256].append(record)

    source_order = {config.key: index for index, config in enumerate(SOURCE_CONFIGS)}
    split_order = {"test": 0, "valid": 1, "train": 2}
    kept = []
    duplicates = []
    conflicts = []
    for digest, group in groups.items():
        signatures = {normalized_signature(record.rows_new) for record in group}
        if len(signatures) > 1:
            conflicts.append(
                f"Duplicate image has conflicting labels: sha256={digest}\n  "
                + "\n  ".join(str(record.image_path) for record in group)
            )
            continue
        ordered = sorted(
            group,
            key=lambda record: (
                split_order[record.source_split],
                source_order[record.source],
                str(record.image_path),
            ),
        )
        kept.append(ordered[0])
        if len(ordered) > 1:
            duplicates.append(ordered)
    return kept, duplicates, conflicts


def deterministic_score(record: Record) -> int:
    return int(hashlib.sha256(f"atoh-v1|{record.source}|{record.sha256}".encode()).hexdigest(), 16)


def assign_splits(records: list[Record], valid_fraction: float) -> None:
    config_by_key = {config.key: config for config in SOURCE_CONFIGS}
    for record in records:
        record.final_split = record.source_split

    by_source: dict[str, list[Record]] = collections.defaultdict(list)
    for record in records:
        by_source[record.source].append(record)

    for source, group in by_source.items():
        if not config_by_key[source].needs_valid_from_train:
            continue
        train_records = [record for record in group if record.source_split == "train"]
        if not train_records:
            continue
        target = max(1, round(len(train_records) * valid_fraction))
        selected = sorted(train_records, key=deterministic_score)[:target]
        for record in selected:
            record.final_split = "valid"


def safe_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return value.strip("._") or "item"


def make_unique_output_names(records: list[Record]) -> None:
    used = set()
    for record in sorted(records, key=lambda item: (item.final_split, item.source, str(item.image_path))):
        base = safe_component(f"{record.source}__{record.image_path.stem}")
        candidate = base
        if candidate.lower() in used:
            candidate = f"{base}__{record.sha256[:12]}"
        if candidate.lower() in used:
            raise RuntimeError(f"Could not create unique output name for {record.image_path}")
        used.add(candidate.lower())
        record.output_image_name = candidate + record.image_path.suffix.lower()
        record.output_label_name = candidate + ".txt"


def write_label(path: Path, rows: list[tuple[int, float, float, float, float]]) -> None:
    lines = [f"{cls} {x:.9f} {y:.9f} {w:.9f} {h:.9f}" for cls, x, y, w, h in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def write_data_yaml(root: Path) -> None:
    lines = [
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        f"nc: {len(ACTIVE_NAMES)}",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in ACTIVE_NAMES.items())
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def copy_source_notes(staging: Path, output: Path) -> None:
    notes_dir = output / "source_records"
    notes_dir.mkdir(parents=True, exist_ok=True)
    candidates = (
        staging / "plantdoc_detection" / "SOURCE_LICENSE.txt",
        staging / "cucumber_detection" / "SOURCE.md",
    )
    for path in candidates:
        if path.is_file():
            shutil.copy2(path, notes_dir / f"{path.parent.name}__{path.name}")


def write_sources(output: Path) -> None:
    lines = [
        "# ATOH Agriculture Detection Dataset v1 - Sources",
        "",
        "This dataset is a derived academic/internship prototype. Each upstream source",
        "retains its own license. The combined dataset must not be redistributed until",
        "all source licenses, especially deepNIR and PlantDoc, are independently verified.",
        "",
        "## Active class schema",
        "",
    ]
    for index, name in ACTIVE_NAMES.items():
        lines.append(f"- `{index}`: `{name}`")
    lines.extend(["", "## Reserved for dataset v2 (not trained in v1)", ""])
    for old_index, name in RESERVED_NAMES.items():
        lines.append(f"- staging `{old_index}`: `{name}`")
    lines.extend(["", "## Upstream sources", ""])
    for config in SOURCE_CONFIGS:
        lines.extend(
            [
                f"### {config.title}",
                "",
                f"- Staging key: `{config.key}`",
                f"- Official/reference URL: {config.url}",
                f"- License: {config.license_text}",
                f"- Notes: {config.notes}",
                "",
            ]
        )
    (output / "SOURCES.md").write_text("\n".join(lines), encoding="utf-8")


def stats(records: list[Record]):
    split_images = collections.Counter(record.final_split for record in records)
    split_boxes = collections.Counter()
    class_boxes = collections.Counter()
    class_images: dict[int, set[str]] = collections.defaultdict(set)
    for record in records:
        split_boxes[record.final_split] += len(record.rows_new)
        for cls, *_ in record.rows_new:
            class_boxes[cls] += 1
            class_images[cls].add(f"{record.final_split}/{record.output_image_name}")
    return split_images, split_boxes, class_boxes, class_images


def write_manifest(output: Path, records: list[Record]) -> None:
    with (output / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "final_split",
                "image",
                "label",
                "source",
                "source_split",
                "source_image",
                "sha256",
            ]
        )
        for record in sorted(records, key=lambda item: (item.final_split, item.output_image_name)):
            writer.writerow(
                [
                    record.final_split,
                    record.output_image_name,
                    record.output_label_name,
                    record.source,
                    record.source_split,
                    str(record.image_path),
                    record.sha256,
                ]
            )


def write_report(output: Path, input_count: int, records: list[Record], duplicates: list[list[Record]]) -> None:
    split_images, split_boxes, class_boxes, class_images = stats(records)
    lines = [
        "AToH AGRICULTURE FINAL DETECTION DATASET v1",
        "=" * 58,
        f"Staged paired images: {input_count}",
        f"Exact duplicate groups removed: {len(duplicates)}",
        f"Exact duplicate images removed: {sum(len(group) - 1 for group in duplicates)}",
        f"Final images: {len(records)}",
        f"Final boxes: {sum(class_boxes.values())}",
        "",
        "Splits:",
    ]
    for split in SPLITS:
        lines.append(f"  {split}: images={split_images[split]}, boxes={split_boxes[split]}")
    lines.extend(["", "Class distribution:"])
    for cls, name in ACTIVE_NAMES.items():
        lines.append(f"  {cls}: {name} | images={len(class_images[cls])}, boxes={class_boxes[cls]}")
    lines.extend(
        [
            "",
            "Reserved/missing from v1:",
            "  goji_berry",
            "  goji_berry_leaf_healthy",
            "  goji_berry_leaf_diseased",
            "",
            "License status:",
            "  CONDITIONAL - deepNIR and PlantDoc redistribution terms require verification.",
            "  Suitable for local academic prototype work; do not publish the combined archive yet.",
            "",
            "MANDATORY HUMAN CHECK:",
            "  Open FINAL_AUDIT.jpg and inspect image content, class names and bounding boxes",
            "  before uploading the dataset to Kaggle or starting training.",
        ]
    )
    (output / "BUILD_REPORT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_audit(output: Path, records: list[Record], samples_per_class: int) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        (output / "AUDIT_NOT_CREATED.txt").write_text(
            "Pillow is not installed. Install/enable Pillow and rerun the builder to create FINAL_AUDIT.jpg.\n",
            encoding="utf-8",
        )
        return

    rng = random.Random(260825)
    selected = []
    by_class: dict[int, list[Record]] = collections.defaultdict(list)
    for record in records:
        for cls in {row[0] for row in record.rows_new}:
            by_class[cls].append(record)
    for cls in ACTIVE_NAMES:
        candidates = sorted(by_class[cls], key=lambda item: (item.final_split, item.output_image_name))
        rng.shuffle(candidates)
        selected.extend((cls, record) for record in candidates[:samples_per_class])

    tile_w, tile_h = 420, 330
    cols = 4
    rows_count = max(1, math.ceil(len(selected) / cols))
    sheet = Image.new("RGB", (cols * tile_w, rows_count * tile_h), "white")
    sheet_draw = ImageDraw.Draw(sheet)
    palette = ["#e53935", "#43a047", "#1e88e5", "#fb8c00", "#8e24aa", "#00897b"]

    for index, (focus_cls, record) in enumerate(selected):
        image_path = output / record.final_split / "images" / record.output_image_name
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for cls, xc, yc, bw, bh in record.rows_new:
            color = palette[cls % len(palette)]
            x1 = max(0, (xc - bw / 2) * width)
            y1 = max(0, (yc - bh / 2) * height)
            x2 = min(width - 1, (xc + bw / 2) * width)
            y2 = min(height - 1, (yc + bh / 2) * height)
            line_width = max(2, min(width, height) // 250)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=line_width)
            draw.text((x1 + 2, max(0, y1 - 14)), ACTIVE_NAMES[cls], fill=color)
        image.thumbnail((tile_w - 12, tile_h - 50))
        col = index % cols
        row = index // cols
        x = col * tile_w + (tile_w - image.width) // 2
        y = row * tile_h + 42
        sheet.paste(image, (x, y))
        heading = f"CHECK {ACTIVE_NAMES[focus_cls]} | {record.final_split} | {record.source}"
        sheet_draw.text((col * tile_w + 6, row * tile_h + 7), heading[:64], fill="black")
    sheet.save(output / "FINAL_AUDIT.jpg", quality=92)


def verify_output(output: Path, records: list[Record]) -> list[str]:
    errors = []
    expected = collections.Counter(record.final_split for record in records)
    for split in SPLITS:
        image_dir = output / split / "images"
        label_dir = output / split / "labels"
        images = [path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS]
        labels = [path for path in label_dir.iterdir() if path.suffix.lower() == ".txt"]
        if len(images) != expected[split]:
            errors.append(f"Output {split} image count mismatch: {len(images)} != {expected[split]}")
        if len(labels) != expected[split]:
            errors.append(f"Output {split} label count mismatch: {len(labels)} != {expected[split]}")
        if {path.stem for path in images} != {path.stem for path in labels}:
            errors.append(f"Output {split} image/label stems do not match")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--output-name", default="atoh_detection_v1")
    parser.add_argument("--valid-fraction", type=float, default=0.10)
    parser.add_argument("--samples-per-class", type=int, default=4)
    args = parser.parse_args()

    project = args.project.expanduser().resolve()
    staging = project / "datasets" / "staging"
    final_parent = project / "datasets" / "final"
    output = final_parent / args.output_name
    temporary = final_parent / f".{args.output_name}.building-{os.getpid()}"

    if not staging.is_dir():
        print(f"ERROR: staging directory not found: {staging}", file=sys.stderr)
        return 2
    if output.exists() or temporary.exists():
        print(f"ERROR: output/build directory already exists: {output} or {temporary}", file=sys.stderr)
        print("Nothing was overwritten.", file=sys.stderr)
        return 2
    if not (0 < args.valid_fraction < 0.5):
        print("ERROR: --valid-fraction must be between 0 and 0.5", file=sys.stderr)
        return 2

    print("Collecting and validating accepted staging sources...")
    records, errors = collect_records(staging)
    if errors:
        print(f"ERROR: staging validation failed with {len(errors)} issue(s):", file=sys.stderr)
        for error in errors[:100]:
            print(f"  {error}", file=sys.stderr)
        if len(errors) > 100:
            print(f"  ... {len(errors) - 100} more", file=sys.stderr)
        return 1

    original_count = len(records)
    print(f"Validated paired images: {original_count}")
    records, duplicates, conflicts = choose_and_deduplicate(records)
    if conflicts:
        print(f"ERROR: {len(conflicts)} duplicate-label conflict(s):", file=sys.stderr)
        for conflict in conflicts[:20]:
            print(conflict, file=sys.stderr)
        return 1

    assign_splits(records, args.valid_fraction)
    make_unique_output_names(records)
    final_parent.mkdir(parents=True, exist_ok=True)
    temporary.mkdir()
    try:
        for split in SPLITS:
            (temporary / split / "images").mkdir(parents=True)
            (temporary / split / "labels").mkdir(parents=True)

        print(f"Copying {len(records)} deduplicated images and labels...", flush=True)
        for index, record in enumerate(records, 1):
            shutil.copy2(
                record.image_path,
                temporary / record.final_split / "images" / record.output_image_name,
            )
            write_label(
                temporary / record.final_split / "labels" / record.output_label_name,
                record.rows_new,
            )
            if index % 500 == 0 or index == len(records):
                print(f"  {index}/{len(records)}", flush=True)

        write_data_yaml(temporary)
        write_sources(temporary)
        copy_source_notes(staging, temporary)
        write_manifest(temporary, records)
        write_report(temporary, original_count, records, duplicates)
        make_audit(temporary, records, args.samples_per_class)

        output_errors = verify_output(temporary, records)
        if output_errors:
            raise RuntimeError("; ".join(output_errors))
        temporary.rename(output)
    except Exception as exc:
        print(f"ERROR during build: {exc}", file=sys.stderr)
        print(f"Incomplete build retained for inspection: {temporary}", file=sys.stderr)
        return 1

    split_images, split_boxes, class_boxes, class_images = stats(records)
    print("\nBUILD COMPLETED")
    print(f"Output: {output}")
    for split in SPLITS:
        print(f"  {split}: images={split_images[split]}, boxes={split_boxes[split]}")
    print(f"Total: images={len(records)}, boxes={sum(class_boxes.values())}")
    print(f"Exact duplicate groups removed: {len(duplicates)}")
    print("\nMANDATORY NEXT ACTION:")
    print(f"  Open and inspect: {output / 'FINAL_AUDIT.jpg'}")
    print("  Do not upload to Kaggle before this visual check.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
