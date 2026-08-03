import torch
from tqdm import tqdm

def calculate_pos_weights(train_df, genre_cols, device):
    """
    学習データから各ジャンルの「負例 / 正例」の比率を計算し、
    BCEWithLogitsLossに渡すpos_weightテンソルを作成します。
    """
    pos_counts = train_df[genre_cols].sum().values
    # 各ジャンルの正例（1）の数を取得
    total_counts = len(train_df)
    # 負例（0）の数を計算
    neg_counts = total_counts - pos_counts
    
    # 重み = 負例 / 正例
    # ※正当なメカニズム：正例が0件の場合の「ゼロ除算エラー(NaN)」を物理的に防ぐため、
    # 分母に微小値(1e-7)を足す安全対策（アンダーフローガード）を入れています。
    weights = neg_counts / (pos_counts + 1e-7)
    
    # PyTorchのテンソルに変換してデバイス(GPU/CPU)に送る
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_one_epoch(model, dataloader, optimizer, criterion, device):
    """
    1エポック分の学習を行う関数
    """
    model.train()
    running_loss = 0.0
    
    # tqdmでプログレスバーを表示（leave=Falseでエポックごとにログを綺麗に保つ）
    for inputs, targets in tqdm(dataloader, desc="Training", leave=False):
        # 1. データをGPU(またはCPU)へ転送
        inputs = inputs.to(device)
        targets = targets.float().to(device) # BCEWithLogitsLossはfloat型の正解ラベルを要求します
        
        # 2. 勾配のリセット
        optimizer.zero_grad()
        
        # 3. 順伝播（Sigmoid無しのLogitsを出力）
        logits = model(inputs)
        
        # 4. 損失の計算（ここで自動的にLog-Sum-Expトリックが適用され安定計算される）
        loss = criterion(logits, targets)
        
        # 5. 逆伝播（勾配の計算）とパラメータの更新
        loss.backward()
        optimizer.step()
        
        # バッチサイズを掛けて全体の損失を累積
        running_loss += loss.item() * inputs.size(0)
        
    # エポック全体の平均損失を返す
    epoch_loss = running_loss / len(dataloader.dataset)
    return epoch_loss