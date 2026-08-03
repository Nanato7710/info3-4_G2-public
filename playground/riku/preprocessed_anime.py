import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm


URL = "https://graphql.anilist.co"

INPUT_PATH = Path("data/preprocessed_anime_data.csv")
OUTPUT_PATH = Path("data/preprocessed_anime_data_year.csv")

QUERY = """
query ($id: Int) {
  Media(id: $id, type: ANIME) {
    id
    seasonYear
    startDate {
      year
    }
  }
}
"""


def fetch_year(anime_id: int) -> int | None:
    response = requests.post(
        URL,
        json={"query": QUERY, "variables": {"id": int(anime_id)}},
        timeout=30,
    )
    response.raise_for_status()

    media = response.json()["data"]["Media"]
    if media is None:
        return None

    return media["seasonYear"] or media["startDate"]["year"]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    years = {}
    for anime_id in tqdm(df["ID"].dropna().astype(int).unique()):
        years[anime_id] = fetch_year(anime_id)
        time.sleep(2.0)

    df["SeasonYear"] = df["ID"].map(years)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print(f"saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()