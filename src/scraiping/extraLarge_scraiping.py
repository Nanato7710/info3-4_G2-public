import time
from pathlib import Path

import pandas as pd
import requests
from requests import Response
from sklearn.preprocessing import MultiLabelBinarizer
from tqdm import tqdm


URL = "https://graphql.anilist.co"
SCRAIPING_MAX_PAGE = 50  # 一つの年に取り出す最大のページ数
ENTRIE_PER_PAGE = 50  # 一つのページに取り出す件数
API_REQUESTS_LIMIT = 20  # SCRAIPING_TIMEの時間の間に送るリクエストの数
SCRAIPING_TIME = 65  # 制限回数ごとの実行時間
SCRAIPING_MIN_YEAR = 2000  # 取り出す年の最初の年
SCRAIPING_MAX_YEAR = 2026  # 取り出す年の最後の年
MAX_RETRIES = 5
REQUEST_TIMEOUT = 30
MIN_REQUEST_INTERVAL = 2.2  # burst limitを避けるため、連続リクエストを少し空ける
OUTPUT_PATH = Path("data/anime_data.csv")
PARTIAL_OUTPUT_PATH = Path("data/anime_data.partial.csv")


PAGE_SEARCH_QUERY = """
query page_search($pageNum:Int,$scraipingYear:Int,$maxPage:Int){
  Page(page:$pageNum,perPage:$maxPage){
    media(type:ANIME,seasonYear:$scraipingYear){
      id
      title{
        native
      }
      genres
      coverImage{
        extraLarge
      }
    }
  }
}
"""


class AniListRequestError(Exception):
    pass


def wait_with_progress(seconds: float, desc: str) -> None:
    wait_seconds = max(0, int(seconds))
    for _ in tqdm(range(wait_seconds), desc=desc, leave=False):
        time.sleep(1)
    rest = seconds - wait_seconds
    if rest > 0:
        time.sleep(rest)


def parse_retry_after(response: Response) -> float | None:
    retry_after = response.headers.get("Retry-After")
    if retry_after is None:
        return None

    try:
        return float(retry_after)
    except ValueError:
        return None


def request_page(session: requests.Session, variables: dict) -> dict:
    for attempt in range(1, MAX_RETRIES + 1):
        response = session.post(
            URL,
            json={"query": PAGE_SEARCH_QUERY, "variables": variables},
            timeout=REQUEST_TIMEOUT,
        )

        if response.status_code == 429:
            retry_after = parse_retry_after(response)
            wait_seconds = retry_after if retry_after is not None else 60
            print(
                f"rate limited: wait {wait_seconds:.1f}s "
                f"(year={variables['scraipingYear']}, page={variables['pageNum']})"
            )
            wait_with_progress(wait_seconds, "rate limit待機中")
            continue

        try:
            body = response.json()
        except ValueError as exc:
            if attempt == MAX_RETRIES:
                raise AniListRequestError(
                    f"JSON decode failed: status={response.status_code}, "
                    f"year={variables['scraipingYear']}, page={variables['pageNum']}"
                ) from exc
            wait_with_progress(2**attempt, "JSON再試行待機中")
            continue

        errors = body.get("errors")
        if response.status_code >= 400 or body.get("data") is None:
            message = (
                f"AniList API error: status={response.status_code}, "
                f"year={variables['scraipingYear']}, page={variables['pageNum']}, "
                f"errors={errors}"
            )

            retry_after = parse_retry_after(response)
            if retry_after is not None and attempt < MAX_RETRIES:
                print(message)
                wait_with_progress(retry_after, "API再試行待機中")
                continue

            if attempt == MAX_RETRIES:
                raise AniListRequestError(message)

            print(message)
            wait_with_progress(2**attempt, "API再試行待機中")
            continue

        return body

    raise AniListRequestError(
        f"request failed after retries: "
        f"year={variables['scraipingYear']}, page={variables['pageNum']}"
    )


def build_dataframe(data: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(data, columns=["ID", "Title", "Genre", "ImageUrl"])

    mlb = MultiLabelBinarizer()
    genre_encoded = pd.DataFrame(
        mlb.fit_transform(df["Genre"]),
        columns=mlb.classes_,
        index=df.index,
    )

    df = pd.concat([df, genre_encoded], axis=1)
    df.drop("Genre", axis=1, inplace=True)
    return df


def save_csv(data: list[list], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_dataframe(data).to_csv(output_path, index=False)


def main() -> None:
    data = []
    request_count = 0
    window_start = time.perf_counter()
    last_request_at = 0.0

    try:
        with requests.Session() as session:
            for year in tqdm(range(SCRAIPING_MIN_YEAR, SCRAIPING_MAX_YEAR + 1), desc="year"):
                for page in tqdm(range(1, SCRAIPING_MAX_PAGE), desc="page", leave=False):
                    variables = {
                        "pageNum": page,
                        "scraipingYear": year,
                        "maxPage": ENTRIE_PER_PAGE,
                    }

                    if request_count >= API_REQUESTS_LIMIT:
                        runtime = time.perf_counter() - window_start
                        if SCRAIPING_TIME > runtime:
                            wait_with_progress(SCRAIPING_TIME - runtime, "待機中")
                        request_count = 0
                        window_start = time.perf_counter()

                    elapsed_since_last_request = time.perf_counter() - last_request_at
                    if elapsed_since_last_request < MIN_REQUEST_INTERVAL:
                        time.sleep(MIN_REQUEST_INTERVAL - elapsed_since_last_request)

                    response = request_page(session, variables)
                    request_count += 1
                    last_request_at = time.perf_counter()

                    media = response["data"]["Page"]["media"]
                    scraiping_line_number = len(media)
                    if scraiping_line_number == 0:
                        break

                    for item in tqdm(media, desc="line", leave=False):
                        title = item.get("title") or {}
                        cover_image = item.get("coverImage") or {}
                        data.append(
                            [
                                item.get("id"),
                                title.get("native"),
                                item.get("genres") or [],
                                cover_image.get("extraLarge"),
                            ]
                        )

        save_csv(data, OUTPUT_PATH)
    except Exception:
        if data:
            save_csv(data, PARTIAL_OUTPUT_PATH)
            print(f"partial data saved: {PARTIAL_OUTPUT_PATH}")
        raise


if __name__ == "__main__":
    main()
