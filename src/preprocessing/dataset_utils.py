import os
import pandas as pd
import requests
from PIL import Image
from io import BytesIO

IMAGE_DIR = "data/images"

# データセット内で使用するジャンル一覧
GENRE_COLS = [
    "Action", "Adventure", "Comedy",
    "Drama", "Ecchi", "Fantasy",
    "Hentai", "Horror", "Mahou Shoujo",
    "Mecha", "Music", "Mystery",
    "Psychological", "Romance", "Sci-Fi",
    "Slice of Life", "Sports",
    "Supernatural", "Thriller"
]


def load_dataset():
    """
    学習用(train)、検証用(validation)、テスト用(test)
    のCSVファイルを読み込む。

    Returns:
        train_df (DataFrame)
        val_df (DataFrame)
        test_df (DataFrame)
    """

    train_df = pd.read_csv("data/series_split_outputs/training_data_grouped.csv")
    val_df = pd.read_csv("data/series_split_outputs/validation_data_grouped.csv")
    test_df = pd.read_csv("data/series_split_outputs/test_data_grouped.csv")

    return train_df, val_df, test_df

def load_image(df, anime_id):
    """
    指定したアニメIDの画像を取得する。

    処理内容:
        1. ローカルに保存済みの画像があるか確認する
        2. 保存済みならローカル画像を読み込む
        3. 保存されていなければURLから画像を取得する
        4. RGB形式へ変換する
        5. ローカルに保存する
        6. 画像オブジェクトを返す

    Args:
        df (pandas.DataFrame):
            アニメ情報を格納したDataFrame。
            "ID"列と"ImageUrl"列を持つことを想定する。

        anime_id (int):
            取得したいアニメのID。

    Returns:
        PIL.Image.Image:
            取得した画像オブジェクト。

        None:
            指定したIDが存在しない場合。
    """

    # 画像保存用フォルダを作成
    os.makedirs(IMAGE_DIR, exist_ok=True)

    # 保存先パス
    save_path = f"{IMAGE_DIR}/{anime_id}.jpg"

    # ① 保存済み画像があればそれを返す
    if os.path.exists(save_path):
        return Image.open(save_path).convert("RGB")

    # ② IDからURLを取得
    row = df[df["ID"] == anime_id]

    if len(row) == 0:
        print(f"ID {anime_id} が見つかりません")
        return None

    url = row.iloc[0]["ImageUrl"]

    # ③ URLから画像を取得
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    # ④ PIL形式で読み込み、RGBへ変換
    image = Image.open(
        BytesIO(response.content)
    ).convert("RGB")

    # ⑤ ローカルへ保存
    image.save(save_path)

    return image
def get_genre_data(df, genre):
    """
    指定したジャンルを持つデータのみ抽出する。

    Args:
        df (DataFrame): 対象データ
        genre (str): ジャンル名

    Returns:
        DataFrame:
        genre列が1のデータのみ
    """

    if genre not in GENRE_COLS:
        raise ValueError(
            f"{genre} は存在しないジャンルです"
        )

    return df[df[genre] == 1]


def show_genre_distribution(df):
    """
    各ジャンルの件数と割合を表示する。

    Args:
        df (DataFrame): 対象データ
    """

    print("\n=== Genre Distribution ===")

    for genre in GENRE_COLS:
        count = df[genre].sum()
        ratio = count / len(df)

        print(
            f"{genre:<15}"
            f"{count:>5}件 "
            f"({ratio:.2%})"
        )


def show_dataset_summary():
    """
    train / validation / test の件数を表示する。

    Example:
        Train      : 7000
        Validation : 1500
        Test       : 1500
    """

    train_df, val_df, test_df = load_dataset()

    print("=== Dataset Summary ===")
    print(f"Train      : {len(train_df)}")
    print(f"Validation : {len(val_df)}")
    print(f"Test       : {len(test_df)}")