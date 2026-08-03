import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/preprocessed_anime_data.csv")

#IDとTitleは特徴量として使用しないため削除
df = df.drop(columns=["Title"])

# 80% train
train_df, temp_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

# 残り20%を10%/10%に分割
validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=42,
    shuffle=True
)

print("全体:", len(df))
print("train:", len(train_df))
print("validation:", len(validation_df))
print("test:", len(test_df))

train_df.to_csv("data/training_data.csv", index=False)
validation_df.to_csv("data/validation_data.csv", index=False)
test_df.to_csv("data/test_data.csv", index=False)