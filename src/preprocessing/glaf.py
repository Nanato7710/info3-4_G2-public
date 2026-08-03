import pandas as pd
import matplotlib.pyplot as plt

genre_cols = [
    "Action","Adventure","Comedy",
    "Drama","Ecchi","Fantasy",
    "Hentai","Horror","Mahou Shoujo",
    "Mecha","Music","Mystery",
    "Psychological","Romance","Sci-Fi",
    "Slice of Life","Sports",
    "Supernatural","Thriller"
]

train_df = pd.read_csv("data/training_data.csv")
val_df = pd.read_csv("data/validation_data.csv")
test_df = pd.read_csv("data/test_data.csv")

train_ratio = [train_df[g].mean() * 100 for g in genre_cols]
val_ratio = [val_df[g].mean() * 100 for g in genre_cols]
test_ratio = [test_df[g].mean() * 100 for g in genre_cols]

plt.figure(figsize=(14, 6))

plt.plot(genre_cols, train_ratio, marker="o", label="Train")
plt.plot(genre_cols, val_ratio, marker="o", label="Validation")
plt.plot(genre_cols, test_ratio, marker="o", label="Test")

plt.xticks(rotation=45)
plt.ylabel("Percentage (%)")
plt.title("Genre Distribution Comparison")
plt.legend()
plt.tight_layout()

plt.show()