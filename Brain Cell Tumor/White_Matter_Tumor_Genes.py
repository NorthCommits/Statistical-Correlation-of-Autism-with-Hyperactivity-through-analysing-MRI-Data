import pandas as pd
import re

# tu archivo de genes tumorales
src = "brain_tumor_genes.csv"
df = pd.read_csv(src)

# intentar adivinar la columna de símbolos
candidates = [c for c in df.columns if re.search(r"(symbol|gene|hugo)", c, re.I)]
if not candidates:
    raise ValueError(f"No encuentro columna de genes en {src}. Columnas: {list(df.columns)}")
col = candidates[0]

# normalizar símbolos
pos = (df[col].astype(str)
               .str.strip()
               .str.upper()
               .dropna()
               .unique())

# guardar como marcadores (POSITIVOS)
pd.DataFrame({"gene_symbol": pos}).to_csv("TCGA_LGG_markers.csv", index=False)

print(f"Marcadores positivos guardados en TCGA_LGG_markers.csv: {len(pos)} genes")
