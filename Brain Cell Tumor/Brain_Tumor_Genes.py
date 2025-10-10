import pandas as pd
import requests
from io import StringIO

url = "https://soft.bioinfo-minzhao.org/bcgene/BCGene4download.txt"

# Pretend to be a normal browser
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/115.0.0.0 Safari/537.36"
}

# Get the file content
response = requests.get(url, headers=headers)
response.raise_for_status()  # raise error if download failed

# Read into DataFrame
df = pd.read_csv(StringIO(response.text), sep="\t")

# Show preview
print(df.head())
print(f"Downloaded {len(df)} rows")

# Save locally
df.to_csv("brain_tumor_genes.csv", index=False)
print("Saved as brain_tumor_genes.csv")
