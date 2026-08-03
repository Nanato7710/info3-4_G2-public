import time
import pandas as pd
from dataset_utils import load_image

# データ読み込み
df = pd.read_csv("data/preprocessed_anime_data.csv")

# 保存先はdataset_utils側と同じ想定
print("=== IMAGE DOWNLOAD START ===")

start_all = time.time()

success = 0
fail = 0

for i, anime_id in enumerate(df["ID"]):

    try:
        print(f"[{i}] ID: {anime_id}")

        img = load_image(df, anime_id)

        if img is not None:
            success += 1
        else:
            fail += 1

    except Exception as e:
        print(f"ERROR ID {anime_id}: {e}")
        fail += 1

end_all = time.time()

print("\n=== RESULT ===")
print(f"成功: {success}")
print(f"失敗: {fail}")
print(f"合計時間: {end_all - start_all:.2f} sec")