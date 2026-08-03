import torch
from sklearn.metrics import average_precision_score, f1_score, hamming_loss
from tqdm import tqdm

def calculate_metrics_from_logits(logits, targets):
    """
    Logits（生データ）から直接評価指標を計算する関数。
    F1/Hamming LossはSigmoidを介さず「0より大きいか」で判定し、
    mAPは閾値化せずlogitの順位情報をそのまま使って計算します。
    """
    # 1. 離散化（0/1判定）
    # Logits > 0 は、Sigmoid(Logits) > 0.5 と数学的に同値です
    preds = (logits > 0).int()
    
    # CPU上のNumPy配列に変換（scikit-learnで処理するため）
    preds_np = preds.cpu().numpy()
    logits_np = logits.cpu().numpy()
    targets_np = targets.cpu().numpy()
    
    # 2. 評価指標の算出
    macro_f1 = f1_score(targets_np, preds_np, average='macro', zero_division=0)
    samples_f1 = f1_score(targets_np, preds_np, average='samples', zero_division=0)
    h_loss = hamming_loss(targets_np, preds_np)

    # 正例がないクラスのAPは定義できないため、mAPの平均から除外します。
    valid_class_indices = targets_np.sum(axis=0) > 0
    if valid_class_indices.any():
        mean_average_precision = average_precision_score(
            targets_np[:, valid_class_indices],
            logits_np[:, valid_class_indices],
            average='macro'
        )
    else:
        mean_average_precision = 0.0
    
    return macro_f1, samples_f1, h_loss, mean_average_precision

def evaluate_model(model, dataloader, criterion, device):
    """
    検証データ（Validation）またはテストデータ（Test）に対してモデルを評価するループ
    """
    # モデルを評価モードに切り替え（DropoutやBatchNormの挙動を固定）
    model.eval()
    running_loss = 0.0
    
    all_logits = []
    all_targets = []
    
    # 勾配計算を無効化（メモリ消費と計算コストを抑える）
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc="Evaluating", leave=False):
            inputs = inputs.to(device)
            targets = targets.float().to(device)
            
            # 順伝播
            logits = model(inputs)
            
            # 損失の計算
            loss = criterion(logits, targets)
            running_loss += loss.item() * inputs.size(0)
            
            # 指標計算用に結果をCPUに移して蓄積
            # ※GPUメモリを圧迫しないよう、この時点で.cpu()に移動させておくのが定石です
            all_logits.append(logits.cpu())
            all_targets.append(targets.cpu())
            
    # エポック全体の平均損失
    epoch_loss = running_loss / len(dataloader.dataset)
    
    # 蓄積したテンソルを行方向（データ方向）に結合
    all_logits = torch.cat(all_logits, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # 全データに対して一括で指標を計算
    macro_f1, samples_f1, h_loss, m_ap = calculate_metrics_from_logits(all_logits, all_targets)
    
    return epoch_loss, macro_f1, samples_f1, h_loss, m_ap
