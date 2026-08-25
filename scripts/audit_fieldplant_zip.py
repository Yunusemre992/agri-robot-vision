#!/usr/bin/env python3
"""Audit a Roboflow YOLO ZIP without extracting it.

Usage:
  python3 audit_fieldplant_zip.py DATASET.zip AUDIT.jpg

Optional slow checks:
  --test-integrity   Read every member and run ZipFile.testzip().
  --exact-hashes     Hash every image to find byte-identical duplicates.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import io
import re
import sys
import zipfile
from pathlib import Path, PurePosixPath

try:
    import yaml
except ImportError:
    yaml = None


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
RF_SUFFIX = re.compile(r"\.rf\.[0-9a-f]{16,64}$", re.IGNORECASE)


def split_of(name: str) -> str:
    parts = [part.lower() for part in PurePosixPath(name).parts]
    for split in ("train", "valid", "val", "test"):
        if split in parts:
            return "valid" if split == "val" else split
    return "unknown"


def source_key(name: str) -> str:
    """Remove Roboflow's generated .rf.<hash> suffix from a filename stem."""
    stem = PurePosixPath(name).stem
    return RF_SUFFIX.sub("", stem)


def load_yaml(raw: bytes):
    text = raw.decode("utf-8", errors="replace")
    if yaml is None:
        return text, None
    try:
        return text, yaml.safe_load(text)
    except Exception:
        return text, None


def normalize_names(value) -> dict[int, str]:
    if isinstance(value, list):
        return {i: str(name) for i, name in enumerate(value)}
    if isinstance(value, dict):
        result = {}
        for key, name in value.items():
            try:
                result[int(key)] = str(name)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def counterpart_image(label_name: str, image_names: set[str]) -> str | None:
    path = PurePosixPath(label_name)
    parts = list(path.parts)
    try:
        idx = [p.lower() for p in parts].index("labels")
    except ValueError:
        return None
    parts[idx] = "images"
    base = str(PurePosixPath(*parts).with_suffix(""))
    for ext in IMAGE_EXTENSIONS:
        candidate = base + ext
        if candidate in image_names:
            return candidate
        candidate_upper = base + ext.upper()
        if candidate_upper in image_names:
            return candidate_upper
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path", type=Path)
    parser.add_argument("audit_image", type=Path, nargs="?")
    parser.add_argument("--test-integrity", action="store_true")
    parser.add_argument("--exact-hashes", action="store_true")
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()

    if not args.zip_path.is_file():
        print(f"HATA: ZIP bulunamadi: {args.zip_path}", file=sys.stderr)
        return 2

    print("=" * 82)
    print("FIELDPLANT YOLO ZIP DENETIMI")
    print("=" * 82)
    print(f"ZIP: {args.zip_path}")
    print(f"Boyut: {args.zip_path.stat().st_size / (1024 ** 3):.2f} GiB")

    with zipfile.ZipFile(args.zip_path) as zf:
        bad = None
        if args.test_integrity:
            print("ZIP butunluk testi: CALISIYOR (buyuk arsivde uzun surebilir)", flush=True)
            bad = zf.testzip()
            print("ZIP butunlugu:", "SAGLAM" if bad is None else f"BOZUK DOSYA: {bad}")
        else:
            print("ZIP merkezi dizini: OK (tam butunluk testi yapilmadi)")

        infos = [info for info in zf.infolist() if not info.is_dir()]
        names_all = [info.filename for info in infos]
        image_infos = [info for info in infos if PurePosixPath(info.filename).suffix.lower() in IMAGE_EXTENSIONS]
        label_infos = [info for info in infos if PurePosixPath(info.filename).suffix.lower() == ".txt"]
        yaml_infos = [info for info in infos if PurePosixPath(info.filename).suffix.lower() in {".yaml", ".yml"}]
        image_names = {info.filename for info in image_infos}

        print(f"Toplam dosya: {len(infos)}")
        print(f"Goruntu: {len(image_infos)}")
        print(f"Etiket dosyasi: {len(label_infos)}")
        print(f"YAML: {len(yaml_infos)}")

        top = collections.Counter(PurePosixPath(name).parts[0] for name in names_all if PurePosixPath(name).parts)
        print("\nUst duzey klasorler:")
        for key, count in top.most_common():
            print(f"  {key}: {count} dosya")

        class_names: dict[int, str] = {}
        print("\nYAML kayitlari:")
        for info in yaml_infos:
            text, data = load_yaml(zf.read(info))
            print(f"\n--- {info.filename} ---")
            print(text.strip())
            if isinstance(data, dict) and not class_names:
                class_names = normalize_names(data.get("names"))

        split_images = collections.Counter(split_of(info.filename) for info in image_infos)
        split_labels = collections.Counter(split_of(info.filename) for info in label_infos)
        print("\nSplit sayilari:")
        for split in ("train", "valid", "test", "unknown"):
            print(f"  {split}: images={split_images[split]}, labels={split_labels[split]}")

        box_counts = collections.Counter()
        image_classes: dict[int, set[str]] = collections.defaultdict(set)
        bbox_lines = polygon_lines = invalid_lines = empty_labels = 0
        labels_without_image = []
        label_rows: dict[str, list[tuple[int, list[float]]]] = {}

        for info in label_infos:
            raw_text = zf.read(info).decode("utf-8", errors="replace")
            rows = []
            nonempty = [line.strip() for line in raw_text.splitlines() if line.strip()]
            if not nonempty:
                empty_labels += 1
            image_name = counterpart_image(info.filename, image_names)
            if image_name is None:
                labels_without_image.append(info.filename)
            for line in nonempty:
                fields = line.split()
                try:
                    cls = int(float(fields[0]))
                    coords = [float(value) for value in fields[1:]]
                    if any(value < 0 or value > 1 for value in coords):
                        raise ValueError
                except (ValueError, IndexError):
                    invalid_lines += 1
                    continue
                if len(fields) == 5:
                    bbox_lines += 1
                elif len(fields) >= 7 and (len(fields) - 1) % 2 == 0:
                    polygon_lines += 1
                else:
                    invalid_lines += 1
                    continue
                box_counts[cls] += 1
                if image_name:
                    image_classes[cls].add(image_name)
                rows.append((cls, coords))
            if image_name:
                label_rows[image_name] = rows

        labelled_images = set(label_rows)
        images_without_label = image_names - labelled_images

        print("\nSinif dagilimi:")
        for cls in sorted(box_counts):
            name = class_names.get(cls, f"UNKNOWN_{cls}")
            print(f"  {cls}: {name} | images={len(image_classes[cls])}, objects={box_counts[cls]}")

        print("\nEtiket kontrolleri:")
        print(f"  Bbox satiri: {bbox_lines}")
        print(f"  Polygon satiri: {polygon_lines}")
        print(f"  Gecersiz satir: {invalid_lines}")
        print(f"  Bos etiket: {empty_labels}")
        print(f"  Etiketsiz goruntu: {len(images_without_label)}")
        print(f"  Goruntusuz etiket: {len(labels_without_image)}")

        family_members: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
        for info in image_infos:
            family_members[source_key(info.filename)].append((split_of(info.filename), info.filename))
        repeated_families = {key: items for key, items in family_members.items() if len(items) > 1}
        cross_split_families = {
            key: items for key, items in repeated_families.items()
            if len({split for split, _ in items}) > 1
        }
        total_extra_variants = sum(len(items) - 1 for items in repeated_families.values())
        max_family = max((len(items) for items in family_members.values()), default=0)

        print("\nRoboflow kaynak-ailesi kontrolu (.rf.<hash> kaldirilarak):")
        print(f"  Benzersiz kaynak anahtari: {len(family_members)}")
        print(f"  Birden cok varyanti olan grup: {len(repeated_families)}")
        print(f"  Ek varyant goruntu: {total_extra_variants}")
        print(f"  Splitler arasi ayni kaynak grubu: {len(cross_split_families)}")
        print(f"  En buyuk kaynak grubu: {max_family} goruntu")
        for idx, (key, items) in enumerate(sorted(repeated_families.items(), key=lambda item: -len(item[1]))[:10], 1):
            print(f"  Grup {idx}: key={key!r}, variants={len(items)}, splits={sorted({s for s, _ in items})}")
            for split, name in items[:4]:
                print(f"    {split} | {name}")

        if args.exact_hashes:
            print("\nTam kopya taramasi: CALISIYOR", flush=True)
            hashes: dict[str, list[tuple[str, str]]] = collections.defaultdict(list)
            for index, info in enumerate(image_infos, 1):
                digest = hashlib.sha256(zf.read(info)).hexdigest()
                hashes[digest].append((split_of(info.filename), info.filename))
                if index % 1000 == 0:
                    print(f"  {index}/{len(image_infos)} goruntu hashlendi", flush=True)
            duplicate_hashes = {key: items for key, items in hashes.items() if len(items) > 1}
            cross_hashes = {
                key: items for key, items in duplicate_hashes.items()
                if len({split for split, _ in items}) > 1
            }
            print(f"  Tam kopya grubu: {len(duplicate_hashes)}")
            print(f"  Splitler arasi tam kopya grubu: {len(cross_hashes)}")

        tomato_ids = [
            cls for cls, name in class_names.items()
            if "tomato" in name.lower()
        ]
        print("\nTomato hedef siniflari:")
        if tomato_ids:
            for cls in tomato_ids:
                print(f"  {cls}: {class_names[cls]} | images={len(image_classes[cls])}, objects={box_counts[cls]}")
        else:
            print("  Bulunamadi")

        if args.audit_image:
            try:
                from PIL import Image, ImageDraw
            except ImportError:
                print("\nUYARI: Pillow kurulu degil; gorsel denetim olusturulamadi.")
            else:
                candidates = []
                for cls in tomato_ids:
                    candidates.extend(sorted(image_classes[cls]))
                candidates = list(dict.fromkeys(candidates))[: args.samples]
                if not candidates:
                    candidates = sorted(image_names)[: args.samples]

                tile_w, tile_h = 420, 330
                cols = 4
                rows_count = max(1, (len(candidates) + cols - 1) // cols)
                sheet = Image.new("RGB", (cols * tile_w, rows_count * tile_h), "white")
                palette = ["#e53935", "#43a047", "#1e88e5", "#fb8c00", "#8e24aa", "#00897b"]
                for idx, image_name in enumerate(candidates):
                    try:
                        image = Image.open(io.BytesIO(zf.read(image_name))).convert("RGB")
                    except Exception:
                        continue
                    draw = ImageDraw.Draw(image)
                    width, height = image.size
                    for cls, coords in label_rows.get(image_name, []):
                        color = palette[cls % len(palette)]
                        if len(coords) == 4:
                            xc, yc, bw, bh = coords
                            x1 = (xc - bw / 2) * width
                            y1 = (yc - bh / 2) * height
                            x2 = (xc + bw / 2) * width
                            y2 = (yc + bh / 2) * height
                            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, width // 300))
                            draw.text((max(0, x1), max(0, y1 - 14)), class_names.get(cls, str(cls)), fill=color)
                    image.thumbnail((tile_w - 12, tile_h - 42))
                    x = (idx % cols) * tile_w + (tile_w - image.width) // 2
                    y = (idx // cols) * tile_h + 30
                    sheet.paste(image, (x, y))
                    label = f"{split_of(image_name)} | {PurePosixPath(image_name).name}"
                    ImageDraw.Draw(sheet).text(((idx % cols) * tile_w + 6, (idx // cols) * tile_h + 6), label[:62], fill="black")
                args.audit_image.parent.mkdir(parents=True, exist_ok=True)
                sheet.save(args.audit_image, quality=90)
                print(f"\nGorsel denetim dosyasi: {args.audit_image}")

    print("\nDENETIM TAMAMLANDI")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
