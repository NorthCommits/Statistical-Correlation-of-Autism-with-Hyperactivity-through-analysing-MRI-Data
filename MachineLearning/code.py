import os
import re
import shutil
import pandas as pd
from pathlib import Path
from typing import List, Dict
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent
cha_folder = BASE_DIR / "Nadig"
features_folder = BASE_DIR / "Features Results"
features_folder.mkdir(exist_ok=True, parents=True)

if features_folder.exists():
    shutil.rmtree(features_folder)
    features_folder.mkdir()

cha_files = list(cha_folder.glob("*.cha"))
print(f"Found {len(cha_files)} .cha files")

FILLER_RE = re.compile(r"\b(uh|um|erm|er|mmm+|hmm+)\b", re.I)
REPEAT_RE = re.compile(r"\b(\w+)\s+\1\b", re.I)

def child_utts(chat_path: Path) -> List[str]:
    utts = []
    with chat_path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("*CHI:"):
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    utts.append(parts[1].strip())
    return utts

def pragmatic_counts(utts: List[str]) -> Dict[str, int | float]:
    c = {
        "filled_pauses": 0,
        "repetitions": 0,
        "coherent_turns": 0,
        "clear_turns": 0,
        "grammar_mistakes": 0,
        "total_utts": len(utts),
        "total_words": 0,
    }
    for u in utts:
        words = u.split()
        num_words = len(words)
        c["total_words"] += num_words
        if FILLER_RE.search(u): c["filled_pauses"] += 1
        if REPEAT_RE.search(u): c["repetitions"] += 1
        if num_words > 5: c["coherent_turns"] += 1
        if num_words >= 4 and u[0].isupper(): c["clear_turns"] += 1
        if num_words <= 2 and u.endswith("?"): c["grammar_mistakes"] += 1
    return c

def densities(c: Dict[str, int | float]) -> Dict[str, float]:
    w = c["total_words"] or 1
    u = c["total_utts"] or 1
    return {
        "filled_pauses_per100w": c["filled_pauses"] / w * 100,
        "repetitions_per100w": c["repetitions"] / w * 100,
        "coherent_turns_per100utts": c["coherent_turns"] / u * 100,
        "clear_turns_per100utts": c["clear_turns"] / u * 100,
        "grammar_mistakes_per100utts": c["grammar_mistakes"] / u * 100,
    }

def extract_features(file: Path) -> Dict[str, int | float | str]:
    utts = child_utts(file)
    counts = pragmatic_counts(utts)
    return {
        "file": file.name,
        **counts,
        **densities(counts),
    }

data = [extract_features(f) for f in cha_files]
df = pd.DataFrame(data)
out_csv = features_folder / "Features_Extracted.csv"
df.to_csv(out_csv, index=False)
print(f"\n Features saved to: {out_csv}\n")

features = [
    "filled_pauses_per100w",
    "repetitions_per100w",
    "coherent_turns_per100utts",
    "clear_turns_per100utts",
    "grammar_mistakes_per100utts"
]

df_clustering = df.dropna(subset=features).copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clustering[features])
agg_cluster = AgglomerativeClustering(n_clusters=2)
df_clustering["cluster"] = agg_cluster.fit_predict(X_scaled)
sil_score = silhouette_score(X_scaled, df_clustering["cluster"])
print(f" Silhouette Score (using validated features): {sil_score:.2f}")

linkage_matrix = linkage(X_scaled, method="ward")
plt.figure(figsize=(10, 6))
dendrogram(linkage_matrix, labels=df_clustering["file"].values, leaf_rotation=90)
plt.title("Dendrogram - Agglomerative Clustering")
plt.xlabel("Transcript File")
plt.ylabel("Distance")
plt.tight_layout()
plt.show()

ml_features_path = BASE_DIR / "Features Results" / "Features_Extracted.csv"
llm_folder = BASE_DIR / "LLM Result"
llm_percentages_path = BASE_DIR / "LLM ADHD%" / "llm_adhd_percentages.csv"
llm_percentages_path.parent.mkdir(parents=True, exist_ok=True)

llm_files = list(llm_folder.glob("*.csv"))
print(f"Found {len(llm_files)} LLM result files")

def extract_llm_adhd_percentage(file_path: Path) -> dict:
    try:
        df = pd.read_csv(file_path)
        if "Speaker" not in df or "Traits" not in df:
            return None
        chi_only = df[df["Speaker"] == "CHI"]
        if chi_only.empty:
            return None
        adhd_flag = chi_only["Traits"].str.upper() != "NONE"
        pct_adhd = adhd_flag.mean()
        return {
            "file": file_path.stem.replace("_analyzed", "") + ".cha",
            "adhd_llm_pct": round(pct_adhd * 100, 2),
            "n_chi_utts": len(chi_only)
        }
    except Exception as e:
        print(f"Error processing {file_path.name}: {e}")
        return None

llm_results = [extract_llm_adhd_percentage(f) for f in llm_files]
llm_results = [res for res in llm_results if res is not None]
llm_df = pd.DataFrame(llm_results)
print("\nExtracted ADHD % from LLM outputs:")
print(llm_df.head())
llm_df.to_csv(llm_percentages_path, index=False)
print(f" Saved LLM summary to: {llm_percentages_path}")

ml_df = pd.read_csv(ml_features_path)
merged_df = pd.merge(ml_df, llm_df, on="file")

features_to_test = [
    "filled_pauses_per100w",
    "repetitions_per100w",
    "coherent_turns_per100utts"
]

correlation_results = []
for feature in features_to_test:
    pearson_r, pearson_p = pearsonr(merged_df[feature], merged_df["adhd_llm_pct"])
    spearman_r, spearman_p = spearmanr(merged_df[feature], merged_df["adhd_llm_pct"])
    correlation_results.append({
        "Feature": feature,
        "Pearson r": round(pearson_r, 3),
        "Pearson p": round(pearson_p, 4),
        "Spearman r": round(spearman_r, 3),
        "Spearman p": round(spearman_p, 4)
    })

correlation_df = pd.DataFrame(correlation_results)
print("\n Correlation Between LLM and ML Features:")
print(correlation_df)

sns.set(style="whitegrid")
plt.figure(figsize=(15, 4))
for i, feature in enumerate(features_to_test, 1):
    plt.subplot(1, 3, i)
    sns.regplot(data=merged_df, x=feature, y="adhd_llm_pct",
                scatter_kws={'s': 60, 'alpha': 0.7},
                line_kws={"color": "red", "linestyle": "--"})
    plt.title(f"{feature} vs LLM ADHD %")
    plt.xlabel(feature.replace("_", " ").title())
    plt.ylabel("LLM ADHD-like %")

plt.tight_layout()
plt.suptitle("LLM vs ML Feature Comparison", fontsize=16, y=1.05)
plt.show()

correlation_out_path = BASE_DIR / "LLM ADHD%" / "llm_vs_ml_correlations.csv"
correlation_df.to_csv(correlation_out_path, index=False)
print(f"Correlation results saved to: {correlation_out_path}")

