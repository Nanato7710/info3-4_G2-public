import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.preprocessing.dataset_utils import GENRE_COLS, load_dataset


class DistributionBalancedLoss(nn.Module):

    def __init__(
        self,
        class_counts: list[int] | np.ndarray,
        alpha: float = 6.0,
        beta: float = 0.999,
        gamma: float = 2.0,
        eps: float = 1e-6,
    ):
        """Distribution-Balanced Loss の実装

        Args:
            class_counts: 各クラス（ジャンル）のポジティブサンプル数が入ったリスト/配列
            alpha: Re-balancingの強度を制御するハイパーパラメータ
            beta: 有効サンプル数（Class-Balanced Loss）のハイパーパラメータ
            gamma: Focal Lossのフォーカスパラメータ
        """
        super().__init__()
        class_counts = torch.tensor(class_counts, dtype=torch.float32)

        # 1. Class-Balanced Weights (有効サンプル数に基づく重み)
        eff_num = 1.0 - torch.pow(beta, class_counts)
        weights = (1.0 - beta) / eff_num
        # 平均が1になるように正規化
        self.cb_weights = weights / torch.mean(weights)

        # 2. Negative-Balancingのための頻度ベースの閾値/重みの準備
        # 各クラスの出現確率
        p_c = class_counts / torch.sum(class_counts)
        # 論文に基づくスケーリング
        self.r_c = alpha * p_c

        self.gamma = gamma
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: モデルの未活性化出力 (batch_size, num_classes)
            targets: 正解マルチラベル (batch_size, num_classes) -> 0 or 1
        """
        device = logits.device
        self.cb_weights = self.cb_weights.to(device)
        self.r_c = self.r_c.to(device)

        # 確率の計算
        probs = torch.sigmoid(logits)

        # --- ① re-balancing (ポジティブとネガティブの勾配バランス調整) ---
        # ネガティブサンプルに対するペナルティをクラス頻度(r_c)に応じて緩和する
        # targets=1 のときはそのまま、targets=0 のときは r_c を用いてロジットを補正
        negative_logits_correction = torch.log(self.r_c + self.eps)
        adjusted_logits = logits - (1.0 - targets) * negative_logits_correction

        # --- ② re-mining (Focal Loss 的な難易度に基づくサンプリング) ---
        # 予測が難しいサンプル（正解なのに確率が低い、不正解なのに確率が高い）を重視
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_weights = torch.pow(1.0 - p_t, self.gamma)

        # 基礎となるBCEの計算 (安定性のために再補正ロジットを使用)
        bce_loss = F.binary_cross_entropy_with_logits(
            adjusted_logits, targets, reduction="none"
        )

        # すべての要素（Class-Balanced × Focal × 補正BCE）を掛け合わせる
        loss = self.cb_weights * focal_weights * bce_loss

        # バッチ全体で平均をとる
        return loss.mean()
    
def build_criterion() -> nn.Module:
    # 1. 訓練データから各ジャンルの件数を自動集計
    train_df, _, _ = load_dataset()
    # 各ジャンル列の合計（1の総数）を取得
    class_counts = train_df[GENRE_COLS].sum().values

    return DistributionBalancedLoss(
        class_counts=class_counts,
        alpha=6.0,
        beta=0.999,
        gamma=2.0,
    )