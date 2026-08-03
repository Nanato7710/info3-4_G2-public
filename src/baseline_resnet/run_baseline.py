import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import sys
import os
import pandas as pd

# 現在のファイル(run_baseline.py)の親の親ディレクトリを絶対パスで指定
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 作成したモジュールのインポート
from src.preprocessing.dataset_utils import load_dataset, load_image, GENRE_COLS
from src.baseline_resnet.model import AnimeResNet
from src.baseline_resnet.train import train_one_epoch
from src.baseline_resnet.evaluate import evaluate_model

# ==========================================
# 1. PyTorch用 データセットクラスの定義
# PandasのDataFrameから、画像とラベルを1件ずつ取り出す仕組み
# ==========================================
class AnimeDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # 1行分のデータを取得
        row = self.df.iloc[idx]

        # アニメIDを取得し、チームの新関数に DataFrame と一緒に入力する
        anime_id = row["ID"]
        image = load_image(self.df, anime_id)

        # 万が一、画像が取得できなかった場合のエラーハンドリング
        if image is None:
            raise ValueError(f"ID: {anime_id} の画像が取得できませんでした。データを確認してください。")

        # 前処理（リサイズやテンソル変換）の適用
        if self.transform:
            image = self.transform(image)

        # ラベルをテンソル化（19クラス分の0/1配列）
        labels = row[GENRE_COLS].values.astype('float32')
        labels = torch.tensor(labels)

        return image, labels

# ==========================================
# 2. メインの学習パイプライン
# ==========================================
def main():
    # デバイスの設定（PCでもサーバーでも自動判定）
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}")

    os.makedirs("src/baseline_resnet/model", exist_ok=True)

    # ハイパーパラメータ
    batch_size = 64
    num_epochs = 100
    learning_rate = 1e-3

    # 画像の前処理（ResNetの標準指定に合わせる）
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    print("Loading data...")
    train_df, val_df, _ = load_dataset() # 今回はテストデータは学習ループ内では使わない
    train_df = train_df
    val_df = val_df

    # データセットとデータローダーの作成
    train_dataset = AnimeDataset(train_df, transform=transform)
    val_dataset = AnimeDataset(val_df, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    print("Initializing model...")
    model = AnimeResNet(num_classes=len(GENRE_COLS)).to(device)
    if device.type != "mps":
        model = torch.compile(model)

    # 損失関数の定義
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # チェックポイントとログ記録用の変数
    best_val_loss = float('inf')
    metrics_history = []

    print("Starting training loop...")
    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

        # 学習フェーズ
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

        # 評価フェーズ
        val_loss, macro_f1, samples_f1, h_loss, m_ap = evaluate_model(model, val_loader, criterion, device)

        # 結果の出力
        print(f"Train Loss : {train_loss:.4f} | Val Loss : {val_loss:.4f}")
        print(f"Macro F1   : {macro_f1:.4f} | Samples F1: {samples_f1:.4f} | Hamming Loss: {h_loss:.4f} | mAP: {m_ap:.4f}")

        # ログの記録
        metrics_history.append({
            "Epoch": epoch + 1,
            "Train_Loss": train_loss,
            "Val_Loss": val_loss,
            "Macro_F1": macro_f1,
            "Samples_F1": samples_f1,
            "Hamming_Loss": h_loss,
            "mAP": m_ap
        })

        # モデル・チェックポイント（過去最高のVal Lossなら保存）
        if val_loss < best_val_loss:
            print(f">>> Val Loss improved ({best_val_loss:.4f} -> {val_loss:.4f}). Saving best model...")
            best_val_loss = val_loss
            raw_model = getattr(model, "_orig_mod", model)  # torch.compileでラップされたモデルから元のモデルを取得
            state = {k: v.cpu() for k, v in raw_model.state_dict().items()}  # CPUに移動して保存
            torch.save(state, "src/baseline_resnet/model/resnet18_best.pth")

    # 学習終了後にスコアをCSV形式で保存
    print("\nTraining complete. Saving all metrics to CSV...")
    df_history = pd.DataFrame(metrics_history)
    df_history.to_csv("src/baseline_resnet/model/baseline_full_metrics.csv", index=False)
    print("Done!")

if __name__ == "__main__":
    main()
