import requests
import json
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
import time
from tqdm import tqdm
import sys

url = "https://graphql.anilist.co"
SCRAIPING_MAX_PAGE=50 #一つの年に取り出す最大のページ数
ENTRIE_PER_PAGE=50 #一つのページに取り出す件数
API_REQUESTS_LIMIT=20 #SCRAIPING_TIMEの時間の間に送るリクエストの数。規約は1分間に30リクエスト
SCRAIPING_TIME=65 #制限回数ごとの実行時間
SCRAIPING_MIN_YEAR=2000 #取り出す年の最初の年
SCRAIPING_MAX_YEAR=2026 #取り出す年の最後の年

page_search_query='''
query page_search($pageNum:Int,$scraipingYear:Int,$maxPage:Int){
  Page(page:$pageNum,perPage:$maxPage){
    media(type:ANIME,seasonYear:$scraipingYear){
      id
      title{
        native
      }
      genres
      coverImage{
        medium
      }
    }
  }
}
'''

data=[]
request_count=0
start=time.perf_counter()
try:
  for year in tqdm(range(SCRAIPING_MIN_YEAR,SCRAIPING_MAX_YEAR+1),desc="year"): 
    for page in tqdm(range(1,SCRAIPING_MAX_PAGE),desc="page",leave=False):
      
      # APIの引数を指定
      page_search_variables={
      "pageNum":page,
      "scraipingYear":year,
      "maxPage":ENTRIE_PER_PAGE
      }

      # 制限回数を超えた場合の実行の一時停止
      if request_count>API_REQUESTS_LIMIT:
        end=time.perf_counter()
        runtime=end-start
        if SCRAIPING_TIME>runtime:
          for i in tqdm(range(int(SCRAIPING_TIME-runtime)),desc="待機中",leave=False):
            time.sleep(1)
        request_count=0
        start=time.perf_counter()
      
      # APIリクエスト実行
      response=requests.post(url,json={'query':page_search_query,'variables': page_search_variables}).json()
      request_count+=1

      SCRAIPING_LINE_NUMBER=len(response["data"]["Page"]["media"])
      if SCRAIPING_LINE_NUMBER==0:break #データの中身がない場合はループを抜ける

      #結果から必要なデータを取り出す
      for line in tqdm(range(SCRAIPING_LINE_NUMBER),desc="line",leave=False):
          id=response["data"]["Page"]["media"][line]["id"]
          title=response["data"]["Page"]["media"][line]["title"]["native"]
          genres=response["data"]["Page"]["media"][line]["genres"]
          image_url=response["data"]["Page"]["media"][line]["coverImage"]["medium"]
          data.append([id,title,genres,image_url])
except:
  print(sys.exc_info())
df=pd.DataFrame(data,columns=["ID","Title","Genre","ImageUrl"])

#ジャンルをone-hotに変換
mlb = MultiLabelBinarizer()
genre_encoded = pd.DataFrame(
    mlb.fit_transform(df["Genre"]),
    columns=mlb.classes_,
    index=df.index
)
df = pd.concat([df, genre_encoded], axis=1)
df.drop("Genre",axis=1,inplace=True)

df.to_csv('data/anime_data.csv',index=False)