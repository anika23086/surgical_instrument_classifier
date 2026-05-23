import pandas as pd
from pathlib import Path

df = pd.read_csv("dataset/metadata.csv")
# Look for Tenaculum, Curved Artery, Straight Artery, Allis
queries = ["Tenaculum", "Artery Forceps (Curved)", "Artery Forceps (Straight)", "Allis Tissue", "Cheatle Forceps"]

print("Metadata inspect:")
for q in queries:
    matches = df[df['name'].str.contains(q, case=False, na=False)]
    print(f"\nQuery: {q}")
    for idx, row in matches.iterrows():
        print(f"  ID: {row['id']} | Name: {row['name']} | SKU: {row['sku']} | Path: {row['image_path']}")
