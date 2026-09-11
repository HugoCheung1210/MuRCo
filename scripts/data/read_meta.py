import pandas as pd

tracks = pd.read_csv(
    "fma_metadata/tracks.csv",
    header=[0,1],
    index_col=0,
    nrows=5
)

print(tracks.columns)
print()
print(tracks[[('track','genres_all'),
              ('track','genre_top'),
              ('track','license'),
              ('track','duration'),
              ('set','subset')]])