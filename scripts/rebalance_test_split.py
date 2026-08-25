#!/usr/bin/env python3
"""
rebalance_test_split.py -- give every class a measurable test set.

Problem it fixes:
  MinneApple's original test split was preserved, so test is 83% apple boxes
  while leaf classes get 10-27 boxes each. You cannot measure AP on 10 boxes.

What it does:
  Moves train images into test until each under-represented class reaches a
  target share of its total boxes. Never touches valid. Never moves an image
  that would drop a class below its train floor.

Non-destructive: reads <input>, writes <output>. Source stays untouched.

  python3 rebalance_test_split.py \
      --input  ~/projects/atoh_agriculture/datasets/final/atoh_detection_v2 \
      --output ~/projects/atoh_agriculture/datasets/final/atoh_detection_v3

  add --dry-run to see the plan without writing anything.
"""

import argparse
import collections
import random
import shutil
import sys
from pathlib import Path

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLITS = ("train", "valid", "test")

# Classes that need rescuing. Fruit classes already have plenty of test boxes.
TARGET_CLASSES = (4, 5, 6, 7, 8, 9, 10, 11)

# Aim for this share of each class's total boxes to live in test.
TARGET_SHARE = 0.15

# Never let train fall below this share of a class's boxes.
TRAIN_FLOOR = 0.60


def read_label(path):
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 5:
            rows.append((int(float(parts[0])), *parts[1:]))
    return rows


def load(root):
    """Return {split: {stem: (image_path, label_path, [class ids])}}"""
    data = {}
    for split in SPLITS:
        idir, ldir = root / split / "images", root / split / "labels"
        if not idir.is_dir():
            sys.exit(f"missing: {idir}")
        images = {p.stem: p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT}
        entries = {}
        for stem, img in images.items():
            lab = ldir / f"{stem}.txt"
            if not lab.exists():
                continue
            rows = read_label(lab)
            entries[stem] = (img, lab, [r[0] for r in rows])
        data[split] = entries
    return data


def class_counts(entries):
    c = collections.Counter()
    for _, _, classes in entries.values():
        c.update(classes)
    return c


def plan_moves(data, seed=20260825):
    train, test = data["train"], data["test"]
    totals = collections.Counter()
    for split in SPLITS:
        totals.update(class_counts(data[split]))

    cur_test = class_counts(test)
    cur_train = class_counts(train)

    need = {}
    for cls in TARGET_CLASSES:
        want = int(totals[cls] * TARGET_SHARE)
        gap = want - cur_test.get(cls, 0)
        if gap > 0:
            need[cls] = gap

    if not need:
        return [], totals, cur_test, {}

    # Score each train image by how much it helps the classes still in deficit.
    rng = random.Random(seed)
    candidates = list(train.items())
    rng.shuffle(candidates)

    moves = []
    moved_train = collections.Counter()
    for stem, (img, lab, classes) in sorted(
        candidates,
        key=lambda kv: -sum(1 for c in kv[1][2] if c in need),
    ):
        helpful = [c for c in classes if need.get(c, 0) > 0]
        if not helpful:
            continue

        # would moving this image starve train for any class it contains?
        blocked = False
        for cls in set(classes):
            n = classes.count(cls)
            remaining = cur_train[cls] - moved_train[cls] - n
            if remaining < totals[cls] * TRAIN_FLOOR:
                blocked = True
                break
        if blocked:
            continue

        moves.append(stem)
        for cls in classes:
            moved_train[cls] += 1
            if cls in need:
                need[cls] -= 1
        need = {k: v for k, v in need.items() if v > 0}
        if not need:
            break

    return moves, totals, cur_test, moved_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    src = args.input.expanduser().resolve()
    dst = args.output.expanduser().resolve()
    if not args.dry_run and dst.exists():
        sys.exit(f"output already exists, refusing to overwrite: {dst}")

    print(f"reading {src} ...")
    data = load(src)
    for s in SPLITS:
        print(f"  {s}: {len(data[s])} images")

    moves, totals, before_test, moved = plan_moves(data)
    move_set = set(moves)

    # ---- report ----
    after_test = collections.Counter(before_test)
    for stem in moves:
        for cls in data["train"][stem][2]:
            after_test[cls] += 1

    names = {}
    yaml = src / "data.yaml"
    if yaml.exists():
        for line in yaml.read_text().splitlines():
            line = line.strip()
            if line and line[0].isdigit() and ":" in line:
                k, v = line.split(":", 1)
                names[int(k)] = v.strip()

    print(f"\nmoving {len(moves)} images from train -> test\n")
    print(f"{'class':<28}{'total':>8}{'test onc':>10}{'test son':>10}{'pay':>8}")
    print("-" * 64)
    for cls in sorted(totals):
        share = after_test[cls] / totals[cls] * 100 if totals[cls] else 0
        flag = "  <-" if cls in TARGET_CLASSES and after_test[cls] > before_test[cls] else ""
        print(f"{names.get(cls, cls):<28}{totals[cls]:>8}{before_test[cls]:>10}"
              f"{after_test[cls]:>10}{share:>7.1f}%{flag}")

    if args.dry_run:
        print("\n[dry-run] hicbir dosya yazilmadi")
        return

    # ---- write ----
    print(f"\nwriting {dst} ...")
    for split in SPLITS:
        (dst / split / "images").mkdir(parents=True, exist_ok=True)
        (dst / split / "labels").mkdir(parents=True, exist_ok=True)

    n = 0
    for split in SPLITS:
        for stem, (img, lab, _) in data[split].items():
            target = "test" if (split == "train" and stem in move_set) else split
            shutil.copy2(img, dst / target / "images" / img.name)
            shutil.copy2(lab, dst / target / "labels" / lab.name)
            n += 1
            if n % 1000 == 0:
                print(f"  {n}")

    for extra in ("data.yaml", "SOURCES.md", "MANIFEST.csv", "BUILD_REPORT.txt"):
        p = src / extra
        if p.exists():
            shutil.copy2(p, dst / extra)
    notes = src / "source_records"
    if notes.is_dir():
        shutil.copytree(notes, dst / "source_records", dirs_exist_ok=True)

    (dst / "REBALANCE_NOTE.txt").write_text(
        f"Rebalanced from {src.name}\n"
        f"Moved {len(moves)} images train -> test\n"
        f"Target share per leaf class: {TARGET_SHARE:.0%}\n"
        f"Train floor: {TRAIN_FLOOR:.0%}\n\n"
        "NOTE: split is image-level. Near-duplicate images from the same source\n"
        "folder may now straddle train and test. Check before trusting test AP.\n",
        encoding="utf-8",
    )

    print(f"\nDONE -> {dst}")
    for split in SPLITS:
        c = len(list((dst / split / "images").iterdir()))
        print(f"  {split}: {c} images")


if __name__ == "__main__":
    main()
