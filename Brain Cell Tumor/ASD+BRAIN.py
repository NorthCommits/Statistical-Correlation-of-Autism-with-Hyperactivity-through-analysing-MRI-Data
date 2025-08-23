import re, csv
from math import comb
from textwrap import wrap
from typing import Optional, Set, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Paths (adjust if needed)
# -----------------------------
DATA_DIR  = Path(__file__).resolve().parent

SFARI_PATH   = DATA_DIR / "SFARI-Gene_genes_07-08-2025release_08-11-2025export.csv"   # ASD genes
TUMOR_MARKERS_PATH = DATA_DIR / "brain_tumor_genes.csv"                                # tumor marker list (positives)

GMT_PATH     = DATA_DIR / "ReactomePathways.gmt"                                       # Reactome GMT
MEASURED_UNIVERSE_PATH = DATA_DIR / "TCGA_LGG_measured_genes.txt"                      # one symbol per line

RESULTS_DIR = (DATA_DIR / "Results" / "ASD+TumorMarkers")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
UNIVERSE_MODE = "measured_if_available"  # or "reactome_only"
SFARI_FILTER_HIGH_CONF = True            # keep SFARI high-confidence tiers if detectable

# -----------------------------
# Gene column guessing
# -----------------------------
GENE_COL_GUESSES = [
    "gene-symbol","gene_symbol","gene symbol","gene","symbol","Gene","Symbol",
    "Gene Symbol","HGNC symbol","HGNC","Approved symbol","Approved Symbol",
    "ApprovedSymbol","GeneName","Name","GeneSymbol","Hugo_Symbol","HUGO_Symbol","Hugo Symbol",
    "SYMBOL","Gene_Symbol","gene_symbol"
]
AVOID_COL_HINTS = re.compile(r"(?:^|\b)(id|score|category|name|description|fullname|ensembl|pubmed|xref|number)(?:$|\b)", re.I)

def guess_gene_col(df: pd.DataFrame) -> Optional[str]:
    lower_map = {c.lower().strip(): c for c in df.columns}
    for cand in GENE_COL_GUESSES:
        if cand.lower().strip() in lower_map:
            return lower_map[cand.lower().strip()]
    for c in df.columns:
        if re.search(r"(symbol|hugo)", c, re.I): return c
    for c in df.columns:
        if re.search(r"gene", c, re.I) and not AVOID_COL_HINTS.search(c): return c
    for c in df.columns:
        if re.search(r"gene", c, re.I): return c
    return None

# -----------------------------
# Loading & cleaning
# -----------------------------
def split_clean_symbols(raw: str) -> Set[str]:
    out: Set[str] = set()
    if not isinstance(raw, str): return out
    s = raw.strip()
    if not s: return out
    s = re.sub(r"\(.*?\)", "", s)
    parts = re.split(r"[;,/|]", s)
    for p in parts:
        tok = p.strip().upper()
        if not tok: continue
        m = re.findall(r"[A-Z0-9\-\.]+", tok)
        if not m: continue
        sym = "".join(m)
        if len(sym) >= 2: out.add(sym)
    return out

def load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No file found at: {path}")
    if path.suffix.lower() in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str)
    try:
        return pd.read_csv(path, dtype=str, engine="python")
    except Exception:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            sample = fh.read(4096)
        try:
            sep = csv.Sniffer().sniff(sample).delimiter
        except Exception:
            sep = "," if path.suffix.lower()==".csv" else "\t"
        return pd.read_csv(path, sep=sep, dtype=str, engine="python")

def normalize_symbol_table(df: pd.DataFrame, tag: str) -> Tuple[pd.DataFrame, str]:
    col = guess_gene_col(df)
    if col is None:
        raise ValueError(f"Could not infer gene column. Columns: {list(df.columns)}")
    tmp = df.copy()
    tmp["__SYMBOL_LIST__"] = tmp[col].apply(lambda v: list(split_clean_symbols(v)))
    tmp = tmp.explode("__SYMBOL_LIST__", ignore_index=True).rename(columns={"__SYMBOL_LIST__":"SYMBOL"})
    tmp = tmp.dropna(subset=["SYMBOL"]).drop_duplicates(subset=["SYMBOL"], keep="first")
    tmp = tmp.rename(columns={c: f"{tag}_{c}" for c in tmp.columns if c!="SYMBOL"})
    cols = ["SYMBOL"] + [c for c in tmp.columns if c!="SYMBOL"]
    return tmp[cols], col

# -----------------------------
# Optional SFARI filter
# -----------------------------
def maybe_filter_sfari(df: pd.DataFrame) -> pd.DataFrame:
    if not SFARI_FILTER_HIGH_CONF: return df
    tier_col = None
    for c in df.columns:
        if re.search(r"(tier|score|category|evidence|rank)", c, re.I):
            tier_col = c; break
    if tier_col is None: return df
    mask = df[tier_col].astype(str).str.contains(r"\b(1|2|high|strong)\b", case=False, na=False)
    out = df[mask].copy()
    return out if not out.empty else df

# -----------------------------
# Simple Venn (ASD vs Tumor markers)
# -----------------------------
def show_venn2(a: set, b: set, labels=("A","B")) -> None:
    used_fallback = False
    try:
        from matplotlib_venn import venn2
        plt.figure(figsize=(5.5,5.5), dpi=140)
        venn2([a,b], set_labels=labels)
        plt.title(f"{labels[0]} ∩ {labels[1]}")
        plt.tight_layout()
    except Exception:
        used_fallback = True
        regions = [len(a-b), len(b-a), len(a & b)]
        labels2 = [f"{labels[0]} only", f"{labels[1]} only", "Overlap"]
        plt.figure(figsize=(6,4), dpi=140)
        plt.bar(labels2, regions)
        for i,v in enumerate(regions):
            plt.text(i, v + (max(regions)*0.05 if max(regions)>0 else 0.5), str(v), ha='center', va='bottom')
        plt.ylabel("Number of genes"); plt.title(f"{labels[0]} vs {labels[1]} (fallback)")
        plt.tight_layout()
    plt.show()
    if used_fallback:
        print("[note] matplotlib-venn not available → bar fallback was used. Install with: pip install matplotlib-venn")

# -----------------------------
# Reactome enrichment (ORA)
# -----------------------------
def load_reactome_gmt(gmt_path: Path) -> dict[str, set[str]]:
    pathways: dict[str,set[str]] = {}
    with gmt_path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3: continue
            name = parts[0].replace("REACTOME_", "").replace("_", " ")
            genes = {g.strip().upper() for g in parts[2:] if g.strip()}
            if genes: pathways[name] = genes
    return pathways

def bh_fdr(pvals: np.ndarray) -> np.ndarray:
    n = len(pvals)
    order = np.argsort(pvals)
    adj = np.empty(n, dtype=float)
    prev = 1.0
    for i, idx in enumerate(order[::-1], start=1):  # largest → smallest
        rank = n - i + 1
        val = min(prev, pvals[idx] * n / rank)
        adj[idx] = val
        prev = val
    return np.clip(adj, 0, 1)

def hypergeom_right_tail(M, K, n, k):
    top = 0
    max_i = min(K, n)
    for i in range(k, max_i + 1):
        top += comb(K, i) * comb(M - K, n - i)
    return top / comb(M, n)

def run_enrichment(gene_set: set[str], pathways: dict[str, set[str]], universe: set[str]) -> pd.DataFrame:
    q = {g for g in gene_set if g in universe}
    if not q: return pd.DataFrame()
    rows = []
    M, n = len(universe), len(q)
    for pname, pgenes in pathways.items():
        pg = pgenes & universe
        K = len(pg)
        if K < 5: continue
        k = len(q & pg)
        if k == 0: continue
        pval = hypergeom_right_tail(M, K, n, k)
        rows.append({
            "pathway": pname, "pathway_size": K, "query_size": n, "overlap": k,
            "p_value": pval, "overlap_genes": ",".join(sorted(q & pg))
        })
    if not rows: return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["fdr_bh"] = bh_fdr(df["p_value"].values.astype(float))
    df["enrichment_ratio"] = (df["overlap"] / df["pathway_size"]) / (df["query_size"] / M)
    df = df.sort_values(["fdr_bh","p_value","overlap"], ascending=[True,True,False]).reset_index(drop=True)
    return df

def plot_top_enrich(df: pd.DataFrame, title: str, topn: int = 12):
    if df is None or df.empty:
        print(f"[warn] No enrichment to plot for: {title}")
        return
    top = df.nsmallest(topn, "fdr_bh").copy().sort_values("fdr_bh")
    wrapped = ["\n".join(wrap(x, width=45)) for x in top["pathway"]]
    xvals = -np.log10(top["fdr_bh"].clip(lower=1e-300))
    plt.figure(figsize=(10, max(4, 0.55*len(top)+1)), dpi=140)
    plt.barh(wrapped, xvals)
    for i, (k, K, fdr) in enumerate(zip(top["overlap"], top["pathway_size"], top["fdr_bh"])):
        plt.text(x=0.02, y=i, s=f"{k}/{K}  (FDR={fdr:.2e})", va="center")
    plt.xlabel("-log10(FDR)")
    plt.title(title)
    plt.tight_layout()
    plt.show()

# -----------------------------
# Pairwise overlap significance (Fisher)
# -----------------------------
def fishers_overlap(a: set, b: set, universe: set) -> Tuple[float, dict]:
    a_, b_ = a & universe, b & universe
    k = len(a_ & b_)
    n = len(a_)
    K = len(b_)
    M = len(universe)
    p = hypergeom_right_tail(M, K, n, k)
    table = {
        "overlap": k,
        "a_only": len(a_ - b_),
        "b_only": len(b_ - a_),
        "neither": M - (k + len(a_-b_) + len(b_-a_))
    }
    return p, table

def print_overlap_stats(name_a: str, a: set, name_b: str, b: set, universe: set):
    p, tab = fishers_overlap(a, b, universe)
    print(f"[Overlap] {name_a} ∩ {name_b}: {tab['overlap']} genes "
          f"(Fisher right-tail p = {p:.3e})")
    print(f"          contingency: a_only={tab['a_only']}, b_only={tab['b_only']}, neither={tab['neither']}")

# -----------------------------
# Universe helpers
# -----------------------------
def load_measured_universe(path: Path) -> Optional[set[str]]:
    if not path.exists(): return None
    try:
        syms = []
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                syms.extend(list(split_clean_symbols(line)))
        return set(map(str.upper, syms))
    except Exception:
        return None

def choose_universe(reactome: dict[str,set[str]], measured: Optional[set[str]]) -> set[str]:
    rx = set().union(*reactome.values())
    if UNIVERSE_MODE == "reactome_only":
        return rx
    if UNIVERSE_MODE == "measured_if_available" and measured:
        return rx & measured
    return rx

# =============================
# FAST ML/AI ADD-ON (single fit, no CV)
# =============================
def build_gene_by_pathway_matrix(pathways: dict[str,set[str]], universe: set[str]) -> pd.DataFrame:
    genes = sorted(universe)
    cols = []
    for pname, pgenes in pathways.items():
        pg = pgenes & universe
        if len(pg) >= 5:
            cols.append((pname, pg))
    cols = sorted(cols, key=lambda x: x[0])
    data = np.zeros((len(genes), len(cols)), dtype=np.int8)
    g2i = {g:i for i,g in enumerate(genes)}
    for j,(pname,pg) in enumerate(cols):
        for g in pg:
            data[g2i[g], j] = 1
    X = pd.DataFrame(data, index=genes, columns=[p for p,_ in cols])
    X = X.loc[:, X.sum(0) > 0]
    return X

def train_quick_elastic_net(X: pd.DataFrame, tumor_genes: set[str]):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
    except Exception as e:
        print("[ML] scikit-learn required. Install with: pip install scikit-learn")
        raise

    y = pd.Series(0, index=X.index, dtype=int)
    y.loc[list(set(X.index) & set(tumor_genes))] = 1

    keep = X.sum(1) > 0
    Xf = X.loc[keep]
    yf = y.loc[keep]

    pipe = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            penalty="elasticnet",
            solver="saga",
            l1_ratio=0.5,
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=-1,
            random_state=7
        )
    )
    pipe.fit(Xf, yf)

    # Diagnostics
    y_score = pipe.predict_proba(Xf)[:, 1]
    fpr, tpr, _ = roc_curve(yf, y_score)
    roc_auc = auc(fpr, tpr)
    pr, rc, _ = precision_recall_curve(yf, y_score)
    ap = average_precision_score(yf, y_score)

    plt.figure(figsize=(5,4), dpi=140)
    plt.plot(fpr, tpr, lw=2); plt.plot([0,1],[0,1],'--', lw=1)
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"Tumor-likeness ROC (AUC={roc_auc:.3f})")
    plt.tight_layout(); plt.show()

    plt.figure(figsize=(5,4), dpi=140)
    plt.plot(rc, pr, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Tumor-likeness PR (AP={ap:.3f})")
    plt.tight_layout(); plt.show()

    return pipe, Xf

def extract_quick_pathway_weights(pipe, X_cols: list[str], top_k: int = 20) -> pd.DataFrame:
    lr = pipe.named_steps["logisticregression"]
    W = pd.Series(lr.coef_.ravel(), index=X_cols, dtype=float).sort_values(ascending=False)

    def _bar(series, title):
        if series.empty:
            print(f"[warn] No coefficients to plot for {title}")
            return
        labels = ["\n".join(wrap(x, width=45)) for x in series.index]
        plt.figure(figsize=(10, max(4, 0.5*len(series))), dpi=140)
        plt.barh(labels, series.values)
        plt.title(title); plt.tight_layout(); plt.show()

    _bar(W.head(top_k).iloc[::-1], "Top tumor-promoting pathways (positive weights)")
    _bar((-W.tail(top_k)).iloc[::-1], "Top anti-tumor pathways (negative weights)")
    return pd.DataFrame({"pathway": W.index, "weight": W.values})

def score_sets_with_model(pipe, X: pd.DataFrame, sets: dict[str,set[str]]) -> pd.DataFrame:
    scores = pipe.predict_proba(X)[:, 1]
    gene_scores = pd.Series(scores, index=X.index, name="tumor_like_score")
    rows = []
    plt.figure(figsize=(7,4), dpi=140)
    k = 0
    for name, s in sets.items():
        s_ = sorted(set(s) & set(X.index))
        if not s_: continue
        vals = gene_scores.loc[s_].values
        k += 1
        rows.append({"set": name, "n": len(vals), "mean": float(np.mean(vals)), "median": float(np.median(vals))})
        x = np.random.normal(loc=k, scale=0.03, size=len(vals))
        plt.scatter(x, vals, s=12, alpha=0.6)
    if rows:
        plt.xticks(range(1, k+1), [r["set"] for r in rows])
    plt.ylabel("Tumor-likeness probability")
    plt.title("Gene-set tumor-likeness scores")
    plt.tight_layout(); plt.show()
    return pd.DataFrame(rows).sort_values("mean", ascending=False)

def quick_permutation_p(pipe, X: pd.DataFrame, target_set: set[str], n=200, seed=11):
    import random
    rng = random.Random(seed)
    all_genes = list(X.index)
    S = list(set(target_set) & set(all_genes))
    if len(S) == 0:
        return np.nan, np.nan
    scores = pipe.predict_proba(X)[:, 1]
    gene_scores = pd.Series(scores, index=all_genes)
    obs = float(gene_scores.loc[S].mean())
    cnt = 0
    for _ in range(n):
        R = rng.sample(all_genes, len(S))
        if float(gene_scores.loc[R].mean()) >= obs:
            cnt += 1
    p = (cnt + 1) / (n + 1)
    return p, obs

# -----------------------------
# Main
# -----------------------------
def main():
    # 1) Load ASD
    asd_raw  = load_table(SFARI_PATH)
    asd_raw_f = maybe_filter_sfari(asd_raw)
    asd_df,  asd_col   = normalize_symbol_table(asd_raw_f, "ASD")
    asd_syms  = set(asd_df["SYMBOL"])

    # 2) Load tumor markers (positives)
    tumor_raw = load_table(TUMOR_MARKERS_PATH)
    tumor_df, tumor_col = normalize_symbol_table(tumor_raw, "Tumor")
    tumor_syms_all = set(tumor_df["SYMBOL"])

    print("Chosen columns:")
    sfari_note = " (filtered to high-confidence tiers)" if SFARI_FILTER_HIGH_CONF else ""
    print(f"  ASD   : {asd_col}{sfari_note}   (unique symbols: {len(asd_syms)})")
    print(f"  Tumor : {tumor_col}               (unique tumor markers: {len(tumor_syms_all)})")

    # 3) Overlap CSV (ASD ∩ Tumor markers)
    asd_tumor = pd.merge(asd_df, tumor_df, on="SYMBOL", how="inner")
    asd_tumor.to_csv(RESULTS_DIR / "ASD+TumorMarkers.csv", index=False)
    print("\nCSV outputs:")
    print(f"  - {RESULTS_DIR / 'ASD+TumorMarkers.csv'}")

    # 4) Venn (show-only)
    show_venn2(asd_syms, tumor_syms_all, labels=("ASD","Tumor markers"))

    # 5) Reactome enrichment (on ASD ∩ Tumor markers)
    if not GMT_PATH.exists():
        print(f"\n[warn] Reactome GMT not found at: {GMT_PATH}\n       Put ReactomePathways.gmt there to run enrichment.")
        return
    reactome = load_reactome_gmt(GMT_PATH)

    measured_universe = load_measured_universe(MEASURED_UNIVERSE_PATH)
    universe = choose_universe(reactome, measured_universe)

    print(f"\nUniverse mode: {UNIVERSE_MODE}")
    print(f"Universe size: {len(universe)} "
          f"({'measured∩Reactome' if (UNIVERSE_MODE!='reactome_only' and measured_universe) else 'Reactome-all'})")

    print("\nPairwise overlap significance (Fisher, right-tail within chosen universe):")
    print_overlap_stats("ASD", asd_syms, "Tumor markers", tumor_syms_all, universe)

    ENRICH_DIR = RESULTS_DIR / "enrichment"
    ENRICH_DIR.mkdir(parents=True, exist_ok=True)

    asd_tumor_syms = asd_syms & tumor_syms_all
    enr_asd_tumor = run_enrichment(asd_tumor_syms, reactome, universe)
    enr_asd_tumor.to_csv(ENRICH_DIR / "ASD_TumorMarkers_Reactome_enrichment.csv", index=False)
    plot_top_enrich(enr_asd_tumor, "Reactome: ASD ∩ Tumor markers")

    print("\nEnrichment CSVs:")
    print(f"  - {ENRICH_DIR / 'ASD_TumorMarkers_Reactome_enrichment.csv'}")

    # =============================
    # 6) FAST ML: pathway-based classifier (single fit)
    # =============================
    try:
        # Build matrix on the chosen universe
        X = build_gene_by_pathway_matrix(reactome, universe)

        # Train model: positives = tumor markers within the universe; negatives = other genes in universe
        model, Xf = train_quick_elastic_net(X, tumor_syms_all)

        # Pathway weights (interpretability)
        weights_df = extract_quick_pathway_weights(model, Xf.columns.tolist(), top_k=20)
        weights_df.to_csv(RESULTS_DIR / "ML_TopPathwayWeights.csv", index=False)

        # Score gene sets (only ASD here)
        sets_to_score = {
            "ASD": asd_syms,
            "ASD∩TumorMarkers": (asd_syms & tumor_syms_all),
        }
        scores_table = score_sets_with_model(model, Xf, sets_to_score)
        scores_table.to_csv(RESULTS_DIR / "ML_GeneSetScores.csv", index=False)
        print("\nGene-set tumor-likeness scores:\n", scores_table)

        # Quick permutation test (n=200) to gauge if ASD mean is higher than random same-size sets
        p, obs = quick_permutation_p(model, Xf, asd_syms, n=200, seed=11)
        print(f"ASD: mean tumor-likeness = {obs:.3f}, empirical p ≈ {p:.4f} (200 perms)")
    except Exception as e:
        print(f"[ML] Skipping ML due to error: {e}")

if __name__ == "__main__":
    main()

