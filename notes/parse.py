import sys
import pandas as pd


df = pd.read_csv(sys.argv[1])
# print(df[["agency", "auroc"]])
print(df)
