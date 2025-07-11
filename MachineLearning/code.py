import os
from pathlib import Path
import re
from statistics import mean, median
from typing import Dict, List
import pandas as pd
from IPython.display import display
 
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
 
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
 
cha_folder = Path(r"C:/Users/danci/OneDrive - University of Birmingham/Documents/Final Project/Nadig")
cha_files = list(cha_folder.glob("*.cha"))
print(f"Found {len(cha_files)} .cha files")
 
FILLER_RE     = re.compile(r"\b(uh|um|erm|er|mmm+|hmm+)\b", re.I)
REPEAT_RE     = re.compile(r"\b(\w+)\s+\1\b", re.I)
REPAIR_RE     = re.compile(r"\b(i mean|i meant|no,? i|sorry)\b", re.I)
TOPICSHIFT_RE = re.compile(r"\b(anyway|whatever|let’s talk about|change the subject)\b", re.I)
INTERRUPT_RE  = re.compile(r"\.\.\.|//|\[/|\\]\\")
 
def child_utts(chat_path: Path) -> List[str]:
    utts: List[str] = []
    with chat_path.open(encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if line.startswith("*CHI:"):
                parts = line.split(":", 1)
                if len(parts) > 1 and parts[1].strip():
                    utts.append(parts[1].strip())
    return utts
 
def pragmatic_counts(utts: List[str]) -> Dict[str, int]:
    c = {
        "fillers": 0, "repairs": 0, "repetitions": 0,
        "topic_shifts": 0, "interruptions": 0, "short_utts": 0,
        "total_utts": len(utts), "total_words": 0,
    }
    for u in utts:
        words = u.split()
        c["total_words"] += len(words)
        if len(words) < 3:
            c["short_utts"] += 1
        if FILLER_RE.search(u):       c["fillers"]       += 1
        if REPEAT_RE.search(u):       c["repetitions"]   += 1
        if REPAIR_RE.search(u):       c["repairs"]       += 1
        if TOPICSHIFT_RE.search(u):   c["topic_shifts"]  += 1
        if INTERRUPT_RE.search(u):    c["interruptions"] += 1
    return c
 
def length_metrics(utts: List[str]) -> Dict[str, float]:
    lens = [len(u.split()) for u in utts]
    return {
        "mean_tokens_per_utt":   mean(lens) if lens else 0,
        "median_tokens_per_utt": median(lens) if lens else 0,
    }
 
def densities(c: Dict[str, int]) -> Dict[str, float]:
    w = c["total_words"] or 1
    u = c["total_utts"]  or 1
    return {
        "fillers_per100w"       : c["fillers"]       / w * 100,
        "repairs_per100w"       : c["repairs"]       / w * 100,
        "topicshifts_per100utts": c["topic_shifts"]  / u * 100,
        "interrupts_per100utts" : c["interruptions"] / u * 100,
    }
 
def extract_features(file: Path) -> Dict[str, int | float | str]:
    utts   = child_utts(file)
    counts = pragmatic_counts(utts)
    return {
        "file": file.name,
        **counts,
        **length_metrics(utts),
        **densities(counts),
    }
 
data = [extract_features(f) for f in cha_files]
df   = pd.DataFrame(data)
print("\nExtracted features for all files:")
display(df.head())
 
base_csv = Path("C:/Users/danci/Documents/features_all.csv")
df.to_csv(base_csv, index=False)
print(f"Features saved → {base_csv}\n")
 
pos = [
    "fillers_per100w", "repairs_per100w", "topicshifts_per100utts",
    "interrupts_per100utts", "short_utts"
]
neg = ["mean_tokens_per_utt", "median_tokens_per_utt"]
 
num_cols = df.select_dtypes("number").columns
scaler = StandardScaler()
Z = pd.DataFrame(scaler.fit_transform(df[num_cols]), columns=num_cols)
 
df["adhd_index"] = Z[pos].mean(axis=1) - Z[neg].mean(axis=1)
 
X_std = scaler.fit_transform(df.select_dtypes("number"))
km    = KMeans(n_clusters=2, random_state=42).fit(X_std)
df["cluster"] = km.labels_
 
print("Silhouette score:", silhouette_score(X_std, km.labels_))
print("\nCluster‑wise means (numeric):\n")
print(df.groupby("cluster").mean(numeric_only=True))
 
idx_means = df.groupby("cluster")["adhd_index"].mean()
adhd_like_cluster = idx_means.idxmax()
 
df["adhd_like"] = (df["cluster"] == adhd_like_cluster).astype(int)
print(f"\nCluster {adhd_like_cluster} labelled as ADHD‑like (adhd_like = 1)")
 
cluster_csv = Path("C:/Users/danci/Desktop/clustered.csv")
df.to_csv(cluster_csv, index=False)
print(f"Clustered results saved → {cluster_csv}")
 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import numpy as np
 
num_cols = df.select_dtypes("number").columns
X_std = StandardScaler().fit_transform(df[num_cols])
 
ks = range(1, 11)
inertias   = []
silhouettes = []
 
for k in ks:
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X_std)
    inertias.append(km.inertia_)
    if k > 1:
        silhouettes.append(silhouette_score(X_std, km.labels_))
    else:
        silhouettes.append(np.nan)
 
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
 
ax1.plot(ks, inertias, marker='o', label='Inertia (WCSS)', color='tab:blue')
ax2.plot(ks, silhouettes, marker='s', label='Silhouette', color='tab:orange')
 
ax1.set_xlabel("Number of clusters (k)")
ax1.set_ylabel("Inertia (lower is better)")
ax2.set_ylabel("Silhouette score (higher is better)")
 
ax1.set_xticks(ks)
ax1.grid(True, linestyle='--', alpha=0.4)
 
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines + lines2, labels + labels2, loc="upper right")
 
plt.title("Elbow & Silhouette analysis for K-Means")
plt.tight_layout()
plt.show()