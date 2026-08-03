from PIL import Image
from pathlib import Path

IMAGE_DIR = "/Users/riku/info3dm/group/info3-4_G2/data/images"

broken = []

for img_path in Path(IMAGE_DIR).glob("*.jpg"):
    try:
        img = Image.open(img_path).convert("RGB")
        img.load()  # 実際にデコード
    except Exception as e:
        broken.append((img_path, str(e)))

print("Broken images:", len(broken))

for b in broken[:20]:
    print(b)