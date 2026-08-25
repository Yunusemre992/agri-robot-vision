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
TARGET_WORDS = ("capsicum", "pepper", "sweetpepper", "apple")


def split_name(path):
    parts = [part.lower() for part in PurePosixPath(path).parts]
    if "train" in parts:
        return "train"
    if "valid" in parts or "val" in parts:
        return "valid"
    if "test" in parts:
        return "test"
    return "unknown"


def parse_names(text):
    names = {}
    lines = text.splitlines()
    active = False
    next_id = 0
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("names:"):
            active = True
            rest = stripped.split(":", 1)[1].strip()
            if rest.startswith("[") and rest.endswith("]"):
                values = [v.strip().strip("'\"") for v in rest[1:-1].split(",")]
                return {i: value for i, value in enumerate(values)}
            continue
        if not active:
            continue
        if stripped.startswith("-"):
            names[next_id] = stripped[1:].strip().strip("'\"")
            next_id += 1
        elif ":" in stripped and stripped.split(":", 1)[0].strip().isdigit():
            key, value = stripped.split(":", 1)
            names[int(key.strip())] = value.strip().strip("'\"")
        elif stripped and not raw.startswith((" ", "\t")):
            break
    return names


def corresponding_label(image_path):
    path = image_path.replace("\\", "/")
    lower = path.lower()
    if "/images/" in lower:
        index = lower.index("/images/")
        path = path[:index] + "/labels/" + path[index + len("/images/"):]
    return str(PurePosixPath(path).with_suffix(".txt"))


def yaml_for_path(path, yaml_records):
    normalized = path.replace("\\", "/")
    matches = []
    for root, yaml_path, names in yaml_records:
        if not root or normalized.startswith(root.rstrip("/") + "/"):
            matches.append((len(root), yaml_path, names))
    if not matches:
        return None, {}
    _, yaml_path, names = max(matches, key=lambda item: item[0])
    return yaml_path, names


def color_for(class_id):
    colors = ["#ff3b30", "#34c759", "#007aff", "#ff9500", "#af52de", "#00c7be"]
    return colors[class_id % len(colors)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("zip_path")
    parser.add_argument("audit_image")
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 78)
    print("deepNIR YOLO ARSIV DENETIMI")
    print("=" * 78)
    print(f"ZIP: {args.zip_path}")

    with zipfile.ZipFile(args.zip_path) as zf:
        bad = zf.testzip()
        print("ZIP butunlugu:", "SAGLAM" if bad is None else f"BOZUK: {bad}")

        members = [name for name in zf.namelist() if not name.endswith("/")]
        lower_to_real = {name.lower(): name for name in members}
        images = [name for name in members if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS]
        labels = [name for name in members if PurePosixPath(name).suffix.lower() == ".txt" and "/labels/" in name.lower()]
        yamls = [name for name in members if PurePosixPath(name).suffix.lower() in {".yaml", ".yml"}]

        print(f"Toplam dosya: {len(members)}")
        print(f"Goruntu: {len(images)}")
        print(f"Etiket: {len(labels)}")
        print(f"YAML: {len(yamls)}")

        top_level = Counter(PurePosixPath(name).parts[0] if PurePosixPath(name).parts else "" for name in members)
        print("\nUst duzey klasorler:")
        for key, count in top_level.most_common():
            print(f"  {key}: {count} dosya")

        yaml_records = []
        print("\nYAML kayitlari:")
        for yaml_path in yamls:
            text = zf.read(yaml_path).decode("utf-8", errors="replace")
            names = parse_names(text)
            root = str(PurePosixPath(yaml_path).parent)
            if root == ".":
                root = ""
            yaml_records.append((root, yaml_path, names))
            print(f"  {yaml_path}")
            print(f"    names={names if names else 'OKUNAMADI'}")

        split_images = Counter(split_name(path) for path in images)
        split_labels = Counter(split_name(path) for path in labels)
        class_images = Counter()
        class_boxes = Counter()
        format_counts = Counter()
        boxes_by_image = defaultdict(list)
        missing_labels = []
        empty_labels = []
        invalid_lines = []

        for image_path in images:
            label_path = corresponding_label(image_path)
            actual_label = lower_to_real.get(label_path.lower())
            if actual_label is None:
                missing_labels.append(image_path)
                continue
            text = zf.read(actual_label).decode("utf-8", errors="replace").strip()
            if not text:
                empty_labels.append(actual_label)
                continue

            yaml_path, names = yaml_for_path(image_path, yaml_records)
            source_key = yaml_path or str(PurePosixPath(image_path).parts[0])
            present = set()
            for line_number, line in enumerate(text.splitlines(), 1):
                parts = line.split()
                try:
                    class_id = int(float(parts[0]))
                    if len(parts) == 5:
                        x, y, w, h = map(float, parts[1:])
                        format_counts["bbox"] += 1
                    elif len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                        coords = list(map(float, parts[1:]))
                        xs, ys = coords[0::2], coords[1::2]
                        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
                        x, y = (x1 + x2) / 2, (y1 + y2) / 2
                        w, h = x2 - x1, y2 - y1
                        format_counts["polygon"] += 1
                    else:
                        raise ValueError("bbox/polygon formati degil")
                    if not all(0 <= value <= 1 for value in (x, y, w, h)) or w <= 0 or h <= 0:
                        raise ValueError("koordinat gecersiz")
                except Exception as error:
                    invalid_lines.append((actual_label, line_number, str(error), line[:160]))
                    continue

                class_name = names.get(class_id, f"class_{class_id}")
                key = (source_key, class_id, class_name)
                class_boxes[key] += 1
                present.add(key)
                boxes_by_image[image_path].append((class_id, class_name, x, y, w, h))
            for key in present:
                class_images[key] += 1

        print("\nSplit sayilari:")
        for split in ("train", "valid", "test", "unknown"):
            print(f"  {split}: images={split_images[split]}, labels={split_labels[split]}")

        print("\nKaynak ve sinif dagilimi:")
        for key in sorted(class_boxes, key=lambda k: (k[0], k[1], k[2])):
            source_key, class_id, class_name = key
            print(f"  {source_key} | {class_id}:{class_name} | images={class_images[key]}, boxes={class_boxes[key]}")

        print("\nEtiket kontrolleri:")
        print(f"  Bbox satiri: {format_counts['bbox']}")
        print(f"  Polygon satiri: {format_counts['polygon']}")
        print(f"  Gecersiz satir: {len(invalid_lines)}")
        print(f"  Bos etiket: {len(empty_labels)}")
        print(f"  Etiketsiz goruntu: {len(missing_labels)}")
        for item in invalid_lines[:10]:
            print(f"    {item}")

        hashes = defaultdict(list)
        for image_path in images:
            hashes[hashlib.sha256(zf.read(image_path)).hexdigest()].append(image_path)
        duplicate_groups = [group for group in hashes.values() if len(group) > 1]
        cross_split_groups = [group for group in duplicate_groups if len({split_name(path) for path in group}) > 1]
        print("\nTam kopya kontrolleri:")
        print(f"  Kopya grubu: {len(duplicate_groups)}")
        print(f"  Splitler arasi kopya grubu: {len(cross_split_groups)}")
        for index, group in enumerate(duplicate_groups[:10], 1):
            print(f"  Grup {index}:")
            for path in group:
                print(f"    {split_name(path)} | {path}")

        target_images = []
        for image_path, boxes in boxes_by_image.items():
            if any(any(word in class_name.lower() for word in TARGET_WORDS) for _, class_name, *_ in boxes):
                target_images.append(image_path)

        random.Random(args.seed).shuffle(target_images)
        capsicum = [p for p in target_images if any(any(w in name.lower() for w in ("capsicum", "pepper", "sweetpepper")) for _, name, *_ in boxes_by_image[p])]
        apple = [p for p in target_images if any("apple" in name.lower() for _, name, *_ in boxes_by_image[p])]
        selected = capsicum[:18] + apple[:6]
        selected = list(dict.fromkeys(selected))[:args.samples]

        if not selected:
            selected = list(boxes_by_image)[:args.samples]
            print("\nUYARI: Isminde capsicum/pepper/apple bulunan sinif bulunamadi; genel ornekler ciziliyor.")

        tile_w, tile_h, columns = 360, 300, 4
        rows = max(1, (len(selected) + columns - 1) // columns)
        canvas = Image.new("RGB", (tile_w * columns, tile_h * rows), "white")
        font = ImageFont.load_default()
        resampling = getattr(Image, "Resampling", Image)

        for index, image_path in enumerate(selected):
            image = Image.open(io.BytesIO(zf.read(image_path))).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            for class_id, class_name, x, y, w, h in boxes_by_image[image_path]:
                x1, y1 = int((x - w / 2) * width), int((y - h / 2) * height)
                x2, y2 = int((x + w / 2) * width), int((y + h / 2) * height)
                color = color_for(class_id)
                draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, width // 300))
                draw.text((max(0, x1 + 2), max(0, y1 + 2)), class_name, fill=color, font=font)
            image.thumbnail((tile_w - 12, tile_h - 40), resampling.LANCZOS)
            tile = Image.new("RGB", (tile_w, tile_h), "white")
            title = f"{split_name(image_path)} | {PurePosixPath(image_path).name[:44]}"
            ImageDraw.Draw(tile).text((5, 5), title, fill="black", font=font)
            tile.paste(image, ((tile_w - image.width) // 2, 28))
            canvas.paste(tile, ((index % columns) * tile_w, (index // columns) * tile_h))

        canvas.save(args.audit_image, quality=92)
        print(f"\nHedef sinifli gorsel sayisi: {len(target_images)}")
        print(f"Gorsel denetim dosyasi: {args.audit_image}")
        print("DENETIM TAMAMLANDI")


if __name__ == "__main__":
    main()
