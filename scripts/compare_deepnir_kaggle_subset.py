#!/usr/bin/env python3
import argparse
import hashlib
import io
import zipfile
from collections import defaultdict
from pathlib import PurePosixPath

from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def corresponding_label(image_path):
    path = image_path.replace("\\", "/")
    lower = path.lower()
    if "/images/" in lower:
        index = lower.index("/images/")
        path = path[:index] + "/labels/" + path[index + len("/images/"):]
    return str(PurePosixPath(path).with_suffix(".txt"))


def normalized_label(data):
    text = data.decode("utf-8", errors="replace")
    return "\n".join(" ".join(line.split()) for line in text.splitlines() if line.strip())


def dhash(data):
    image = Image.open(io.BytesIO(data)).convert("L").resize((9, 8))
    pixels = list(image.getdata())
    value = 0
    for row in range(8):
        for column in range(8):
            left = pixels[row * 9 + column]
            right = pixels[row * 9 + column + 1]
            value = (value << 1) | int(left > right)
    return value


def hamming(a, b):
    return (a ^ b).bit_count()


def load_records(zip_path, target_only=False):
    records = []
    with zipfile.ZipFile(zip_path) as zf:
        members = [name for name in zf.namelist() if not name.endswith("/")]
        lower_to_real = {name.lower(): name for name in members}
        images = [name for name in members if PurePosixPath(name).suffix.lower() in IMAGE_EXTENSIONS]
        for image_path in images:
            lower = image_path.lower().replace("\\", "/")
            if target_only and "/apple/" not in lower and "/capsicum/" not in lower:
                continue
            data = zf.read(image_path)
            label_path = corresponding_label(image_path)
            actual_label = lower_to_real.get(label_path.lower())
            label = normalized_label(zf.read(actual_label)) if actual_label else None
            crop = "apple" if "/apple/" in lower else "capsicum" if "/capsicum/" in lower else "unknown"
            records.append({
                "path": image_path,
                "crop": crop,
                "sha256": hashlib.sha256(data).hexdigest(),
                "dhash": dhash(data),
                "label": label,
            })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("official_zip")
    parser.add_argument("kaggle_zip")
    args = parser.parse_args()

    official = load_records(args.official_zip, target_only=True)
    kaggle = load_records(args.kaggle_zip, target_only=False)
    kaggle_by_sha = defaultdict(list)
    for record in kaggle:
        kaggle_by_sha[record["sha256"]].append(record)

    exact_images = 0
    exact_labels = 0
    unmatched = []
    for record in official:
        matches = kaggle_by_sha.get(record["sha256"], [])
        if matches:
            exact_images += 1
            if any(match["label"] == record["label"] for match in matches):
                exact_labels += 1
        else:
            unmatched.append(record)

    perceptual = []
    for record in unmatched:
        best = min(kaggle, key=lambda other: hamming(record["dhash"], other["dhash"]), default=None)
        if best is not None:
            distance = hamming(record["dhash"], best["dhash"])
            if distance <= 4:
                perceptual.append((record, best, distance))

    crop_counts = defaultdict(int)
    for record in official:
        crop_counts[record["crop"]] += 1

    print("=" * 74)
    print("deepNIR RESMI ARSIV - kaggle_gpt ICERIK KARSILASTIRMASI")
    print("=" * 74)
    print(f"Resmi hedef goruntu: {len(official)}")
    print(f"  apple: {crop_counts['apple']}")
    print(f"  capsicum: {crop_counts['capsicum']}")
    print(f"kaggle_gpt toplam goruntu: {len(kaggle)}")
    print(f"Birebir ayni goruntu: {exact_images}/{len(official)}")
    print(f"Goruntu + etiket birebir ayni: {exact_labels}/{len(official)}")
    print(f"Birebir eslesmeyen: {len(unmatched)}")
    print(f"Birebir degil ama algisal yakin (dHash <= 4): {len(perceptual)}")

    if unmatched:
        print("\nIlk eslesmeyen resmi kayitlar:")
        perceptual_by_path = {item[0]["path"]: item for item in perceptual}
        for record in unmatched[:20]:
            print(f"  {record['crop']} | {record['path']}")
            if record["path"] in perceptual_by_path:
                _, best, distance = perceptual_by_path[record["path"]]
                print(f"    yakin: distance={distance} | {best['path']}")

    if exact_images == len(official) and exact_labels == len(official):
        print("\nSONUC: Apple ve capsicum alt kumeleri birebir ayni.")
    elif exact_images + len(perceptual) == len(official):
        print("\nSONUC: Alt kumeler ayni gorunuyor; bazi dosyalar yeniden kodlanmis.")
    else:
        print("\nSONUC: Alt kumeler tamamen ayni degil; ayrintili inceleme gerekli.")


if __name__ == "__main__":
    main()
