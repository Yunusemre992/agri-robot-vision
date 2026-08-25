#!/usr/bin/env python3
import argparse
import hashlib
import io
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import PurePosixPath

from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def split_name(path):
    parts = [p.lower() for p in PurePosixPath(path).parts]
    if "train" in parts:
        return "train"
    if "valid" in parts or "val" in parts:
        return "valid"
    if "test" in parts:
        return "test"
    return "unknown"


def corresponding_label(image_path):
    path = image_path.replace("\\", "/")
    lower = path.lower()
    if "/images/" in lower:
        i = lower.index("/images/")
        path = path[:i] + "/labels/" + path[i + len("/images/"):]
    return str(PurePosixPath(path).with_suffix(".txt"))


def parse_names(yaml_text):
    names = {}
    lines = yaml_text.splitlines()
    in_names = False
    list_index = 0
    for raw in lines:
        line = raw.strip()
        if line.startswith("names:"):
            in_names = True
            rest = line.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                values = [x.strip().strip("'\"") for x in rest[1:-1].split(",")]
                return {i: value for i, value in enumerate(values)}
            continue
        if in_names:
            if line.startswith("-"):
                names[list_index] = line[1:].strip().strip("'\"")
                list_index += 1
            elif ":" in line and line.split(":", 1)[0].strip().isdigit():
                key, value = line.split(":", 1)
                names[int(key.strip())] = value.strip().strip("'\"")
            elif line and not raw.startswith((" ", "\t")):
                break
    return names


def color_for(class_id):
    palette = [
        "#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de",
        "#00c7be", "#ff2d55", "#5856d6", "#a2845e", "#64d2ff",
    ]
    return palette[class_id % len(palette)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("audit_image")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 76)
    print("WOLFBERRY / GOJI YOLO DATASET DENETIMI")
    print("=" * 76)
    print(f"ZIP: {args.zip_path}")

    with zipfile.ZipFile(args.zip_path) as zf:
        bad = zf.testzip()
        print("ZIP butunlugu:", "SAGLAM" if bad is None else f"BOZUK: {bad}")
        members = [m for m in zf.namelist() if not m.endswith("/")]
        lower_to_real = {m.lower(): m for m in members}
        images = [m for m in members if PurePosixPath(m).suffix.lower() in IMAGE_EXTENSIONS]
        labels = [m for m in members if PurePosixPath(m).suffix.lower() == ".txt" and "/labels/" in m.lower()]
        yamls = [m for m in members if PurePosixPath(m).suffix.lower() in {".yaml", ".yml"}]

        print(f"Toplam dosya: {len(members)}")
        print(f"Goruntu: {len(images)}")
        print(f"Etiket dosyasi: {len(labels)}")
        print(f"YAML: {yamls}")

        names = {}
        for yaml_path in yamls:
            text = zf.read(yaml_path).decode("utf-8", errors="replace")
            print(f"\n--- {yaml_path} ---\n{text.strip()}")
            parsed = parse_names(text)
            if parsed:
                names.update(parsed)

        split_images = Counter(split_name(p) for p in images)
        split_labels = Counter(split_name(p) for p in labels)
        image_counts = Counter()
        box_counts = Counter()
        boxes_by_image = defaultdict(list)
        invalid_lines = []
        empty_labels = []
        missing_labels = []

        for image_path in images:
            label_path = corresponding_label(image_path)
            actual = lower_to_real.get(label_path.lower())
            if actual is None:
                missing_labels.append(image_path)
                continue
            text = zf.read(actual).decode("utf-8", errors="replace").strip()
            if not text:
                empty_labels.append(actual)
                continue
            present = set()
            for line_no, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("5 alan bekleniyor")
                    class_id = int(float(parts[0]))
                    x, y, w, h = map(float, parts[1:])
                    if not all(0 <= v <= 1 for v in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError("koordinat aralik disi")
                except Exception as error:
                    invalid_lines.append((actual, line_no, line, str(error)))
                    continue
                boxes_by_image[image_path].append((class_id, x, y, w, h))
                box_counts[class_id] += 1
                present.add(class_id)
            for class_id in present:
                image_counts[class_id] += 1

        label_set = {p.lower() for p in labels}
        orphan_labels = []
        for label_path in labels:
            base = label_path.replace("\\", "/")
            lower = base.lower()
            if "/labels/" in lower:
                i = lower.index("/labels/")
                base = base[:i] + "/images/" + base[i + len("/labels/"):]
            stem = str(PurePosixPath(base).with_suffix(""))
            if not any((stem + ext).lower() in lower_to_real for ext in IMAGE_EXTENSIONS):
                orphan_labels.append(label_path)

        print("\nSplit sayilari:")
        for split in ("train", "valid", "test", "unknown"):
            print(f"  {split}: images={split_images[split]}, labels={split_labels[split]}")

        print("\nSinif dagilimi:")
        for class_id in sorted(set(image_counts) | set(box_counts)):
            print(f"  {class_id}: {names.get(class_id, 'UNKNOWN')} | images={image_counts[class_id]}, boxes={box_counts[class_id]}")

        print("\nEtiket kontrolleri:")
        print(f"  Gecersiz satir: {len(invalid_lines)}")
        print(f"  Bos etiket: {len(empty_labels)}")
        print(f"  Etiketsiz goruntu: {len(missing_labels)}")
        print(f"  Goruntusuz etiket: {len(orphan_labels)}")

        hashes = defaultdict(list)
        for image_path in images:
            hashes[hashlib.sha256(zf.read(image_path)).hexdigest()].append(image_path)
        duplicate_groups = [group for group in hashes.values() if len(group) > 1]
        cross_split = [group for group in duplicate_groups if len({split_name(p) for p in group}) > 1]
        print("\nTam kopya kontrolleri:")
        print(f"  Kopya grubu: {len(duplicate_groups)}")
        print(f"  Splitler arasi kopya grubu: {len(cross_split)}")
        for index, group in enumerate(duplicate_groups[:10], 1):
            print(f"  Grup {index}:")
            for path in group:
                print(f"    {split_name(path)} | {path}")

        candidates = [p for p in images if boxes_by_image.get(p)]
        random.Random(args.seed).shuffle(candidates)
        selected = []
        by_class = defaultdict(list)
        for path in candidates:
            for class_id, *_ in boxes_by_image[path]:
                by_class[class_id].append(path)
        for class_id in sorted(by_class):
            choices = list(dict.fromkeys(by_class[class_id]))
            random.Random(args.seed + class_id).shuffle(choices)
            selected.extend(choices[:4])
        selected = list(dict.fromkeys(selected))[:args.samples]
        if len(selected) < args.samples:
            selected.extend([p for p in candidates if p not in selected][:args.samples - len(selected)])

        thumb_w, thumb_h = 360, 300
        cols = 4
        rows = max(1, (len(selected) + cols - 1) // cols)
        canvas = Image.new("RGB", (cols * thumb_w, rows * thumb_h), "white")
        font = ImageFont.load_default()
        resampling = getattr(Image, "Resampling", Image)

        for i, image_path in enumerate(selected):
            image = Image.open(io.BytesIO(zf.read(image_path))).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for class_id, x, y, w, h in boxes_by_image[image_path]:
                x1 = int((x - w / 2) * width)
                y1 = int((y - h / 2) * height)
                x2 = int((x + w / 2) * width)
                y2 = int((y + h / 2) * height)
                color = color_for(class_id)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, width // 300))
                draw.text((max(0, x1 + 2), max(0, y1 + 2)), names.get(class_id, str(class_id)), fill=color, font=font)
            image.thumbnail((thumb_w - 12, thumb_h - 38), resampling.LANCZOS)
            tile = Image.new("RGB", (thumb_w, thumb_h), "white")
            tile.paste(image, ((thumb_w - image.width) // 2, 24))
            ImageDraw.Draw(tile).text((5, 5), f"{split_name(image_path)} | {PurePosixPath(image_path).name[:42]}", fill="black", font=font)
            canvas.paste(tile, ((i % cols) * thumb_w, (i // cols) * thumb_h))

        canvas.save(args.audit_image, quality=92)
        print(f"\nGorsel denetim dosyasi: {args.audit_image}")
        print("DENETIM TAMAMLANDI")


if __name__ == "__main__":
    main()
