import sys
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import importlib
import random

# プロジェクトルートにパスを通す
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

module = importlib.import_module("experiments.sho-high-res.utility")
LetterboxResize = module.LetterboxResize

def compare_resizes(image_path: str | Path, size: int = 384):
    # (既存のcompare_resizes関数はそのまま)
    img = Image.open(image_path).convert("RGB")
    legacy_transform = T.Resize((size, size))
    img_legacy = legacy_transform(img)
    letterbox_transform = LetterboxResize(size)
    img_letterbox = letterbox_transform(img)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(img); axes[0].set_title(f"Original\n({img.size[0]}x{img.size[1]})"); axes[0].axis("off")
    axes[1].imshow(img_legacy); axes[1].set_title(f"Legacy Resize\n({size}x{size})"); axes[1].axis("off")
    axes[2].imshow(img_letterbox); axes[2].set_title(f"Letterbox Resize\n({size}x{size})"); axes[2].axis("off")
    
    plt.tight_layout()
    print(f"画像 {image_path.name} を処理しました。")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="画像リサイズ比較ツール")
    parser.add_argument("--id", type=str, help="指定する画像のID（ファイル名の一部を指定）")
    args = parser.parse_args()

    image_dir = Path("data/images")
    image_files = list(image_dir.glob("*.jpg"))
    
    if not image_files:
        print(f"エラー: {image_dir} に .jpg ファイルが見つかりませんでした。")
        sys.exit(1)

    target_image = None

    if args.id:
        # IDが含まれるファイルを検索
        matches = [f for f in image_files if args.id in f.name]
        if matches:
            target_image = matches[0]
            print(f"ID '{args.id}' に一致する画像を選択しました: {target_image.name}")
        else:
            print(f"エラー: ID '{args.id}' を含む画像が見つかりませんでした。")
            sys.exit(1)
    else:
        # ID指定がない場合はランダム
        target_image = random.choice(image_files)
        print(f"全 {len(image_files)} 件からランダムに選択しました: {target_image.name}")

    compare_resizes(target_image, size=414)