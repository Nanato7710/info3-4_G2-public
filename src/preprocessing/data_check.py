#ここでは基本的に欠損値、重複の確認、削除およびurlの確認も行う
import pandas as pd
import requests
import time

df = pd.read_csv("data/anime_data.csv")
before_df = df.copy()
valid_rows = []
genre_cols = [
    "Action","Adventure","Comedy",
    "Drama","Ecchi","Fantasy",
    "Hentai","Horror","Mahou Shoujo",
    "Mecha","Music","Mystery",
    "Psychological","Romance","Sci-Fi","Slice of Life",
    "Sports","Supernatural","Thriller"
]

df = df.dropna()
df = df[df[genre_cols].sum(axis=1) != 0]
df = df.reset_index(drop=True)

headers = {
    "User-Agent": "Mozilla/5.0"
}

for i, url in enumerate(df["ImageUrl"]):
    try:
        response = requests.get(
            url,
            headers=headers,
            timeout=5
        )
        if response.status_code == 200:
            valid_rows.append(i)
        if i % 50 == 0:
            print(f"{i}件確認")
        time.sleep(0.1)
    except Exception as e:
        print(e)

after_df = df.loc[valid_rows]

print("欠損値の確認\n", df.isnull().sum())
print("\nurlから重複の確認", df["ImageUrl"].duplicated().sum())
print("欠損処理前のデータ量", len(before_df))
print("欠損処理後のデータ量", len(df))
print("有効URL数", len(valid_rows))

after_df.to_csv('data/preprocessed_anime_data.csv',index=False)