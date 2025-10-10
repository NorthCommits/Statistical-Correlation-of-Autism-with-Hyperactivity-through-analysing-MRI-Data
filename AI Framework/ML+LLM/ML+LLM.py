from __future__ import annotations
from pathlib import Path
import argparse
import random
import re
from collections import Counter
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import pearsonr, spearmanr

# ===================== Paths =====================
BASE_DIR = Path(__file__).resolve().parent
TRANSCRIPTS_DIR = BASE_DIR / "Nadig"                     # .cha transcripts
LLM_RESULTS_DIR = BASE_DIR.parent / "LLM Agent" / "Outputs"     # LLM outputs (CSV)
OUTPUT_DIR = BASE_DIR / "Results" / "ML+LLM"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===================== Plot style =====================
PLOT_RC = {
    "axes.grid": False,
    "figure.dpi": 110,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
}
def use_plot_style():
    plt.rcParams.update(PLOT_RC)

# ===================== IDs & Regex =====================
TRAIT_COL_RE = re.compile(r"^T\d{2}_")
TRAIT_IDS = [f"T{t:02d}" for t in range(1, 51)]

T_LABELS = {
    "T01":"Talks Excessively","T02":"Rapid Speech","T03":"Interruptive Speech","T04":"Topic Switching",
    "T05":"Blurting Out Answers","T06":"Difficulty Waiting Turn","T07":"Filler Overuse","T08":"Disfluency",
    "T09":"Echolalia","T10":"Stilted/Pedantic","T11":"Emotional Reactivity","T12":"Inattentiveness Phrases",
    "T13":"Impulsive Responses","T14":"Working Memory Lapses","T15":"Motor Restlessness","T16":"Delay Aversion",
    "T17":"Overactive Gesturing","T18":"Sensory Distractibility","T19":"Reward-Seeking Urgency","T20":"Emotional Dysregulation",
    "T21":"Perseverative Speech","T22":"Monosyllabic Responses","T23":"Excessive Inquiries","T24":"Rapid Topic Overrun",
    "T25":"Sentence Fragments","T26":"Vocal Fillers Overuse","T27":"Rapid Laughter","T28":"Abrupt Volume Changes",
    "T29":"Garbled Articulation","T30":"Frequent Self-Correction","T31":"Auditory Overload Response","T32":"Sensory Hyperfocus",
    "T33":"Nonstop Oral Motor Noise","T34":"Peripheral Awareness Distraction","T35":"Rapid Topic Recap","T36":"Over-enthusiastic Tone",
    "T37":"Forced Humor","T38":"Distracted Eye Contact","T39":"Repetitive Questioning","T40":"Meta-Talk Overuse",
    "T41":"Peripheral Vocalizations","T42":"Segmented Speech","T43":"Excessive Qualifiers","T44":"Pseudo-Questions",
    "T45":"Immediate Repair","T46":"Spatial Drift","T47":"Content Overload","T48":"Rapid Ideation",
    "T49":"Externalization","T50":"Anticipatory Speech"
}

# ===================== Name / Token helpers =====================
def normalize_filename(name: str) -> str:
    s = name.lower()
    s = re.sub(r"(_|-|\s)analy(sed|zed)?$", "", s)
    s = re.sub(r"\.(cha|csv)$", "", s)
    m = re.search(r"\d+", s)
    return m.group(0) if m else re.sub(r"[^a-z0-9]+", "", s)

def tokenize(text: str) -> List[str]:
    text = text.lower()
    text = re.sub(r"\[[^\]]*\]", " ", text)
    text = re.sub(r"<[^>]*>", " ", text)
    text = re.sub(r"[^a-z0-9'\s-]", " ", text)
    return [t for t in re.split(r"\s+", text) if t]

def read_transcript(path: Path) -> Tuple[List[str], str]:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".cha":
        lines = [ln.strip() for ln in raw.splitlines()
                 if ln.strip().startswith(("*CHI:", "*PAR:", "*SPE:"))]
        chi = [ln for ln in lines if ln.startswith("*CHI:")]
        utterances = [re.sub(r"^\*(CHI|PAR|SPE):\s*", "", ln) for ln in (chi or lines)]
    else:
        utterances = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    return utterances, "\n".join(utterances)

# ===================== Phrase / word lists =====================
FILLERS = {"um", "uh", "er", "erm", "eh", "ah", "like", "you", "know", "mmm", "hmm"}
QUALIFIERS = {"very", "really", "super", "extremely", "totally", "so", "quite", "literally", "absolutely", "incredibly"}
FORMAL_WORDS = {"permit", "moreover", "therefore", "consequently", "furthermore", "precisely", "hence", "thus"}

# ===================== Basic counters =====================
def avg_utterance_length(utts): return (np.mean([len(tokenize(u)) for u in utts]) if utts else 0.0)

def long_utterance_rate(utts, t=20):
    L = [len(tokenize(u)) for u in utts]
    return sum(x >= t for x in L) / max(1, len(L))

def overrun_rate(utts):
    return sum((u.count(",") + u.count(";")) >= 3 and u.count(".") == 0 for u in utts) / max(1, len(utts))

def interruptions_per100u(raw, n):
    return 100 * len(re.findall(r"\[//\]|\[/\]|//|\[\/\]", raw)) / max(1, n)

def fillers_per1000w(tokens):
    c = i = 0
    while i < len(tokens):
        t = tokens[i]
        if t in FILLERS:
            c += 1
        elif t == "you" and i + 1 < len(tokens) and tokens[i + 1] == "know":
            c += 1
            i += 1
        i += 1
    return 1000 * c / max(1, len(tokens))

def disfluencies_per1000w(raw, tokens):
    rep = len(re.findall(r"\b(\w+)\s+\1\b", raw, re.I))
    dsh = raw.count("--")
    return 1000 * (rep + dsh) / max(1, len(tokens))

def jaccard_similarity(a, b):
    A = {t for t in tokenize(a) if len(t) > 2}
    B = {t for t in tokenize(b) if len(t) > 2}
    if not (A or B): return 0.0
    return len(A & B) / max(1, len(A | B))

def topic_switch_rate(utts, thr=0.1):
    if len(utts) < 2: return 0.0
    sw = sum(1 for i in range(1, len(utts)) if jaccard_similarity(utts[i - 1], utts[i]) <= thr)
    return sw / (len(utts) - 1)

def formal_words_per1000w(tokens): return 1000 * sum(t in FORMAL_WORDS for t in tokens) / max(1, len(tokens))
def exclamations_per_utt(raw, n): return raw.count("!") / max(1, n)

def impulsive_after_question(utts, max_len=3):
    c, tot = 0, 0
    for i in range(1, len(utts)):
        if utts[i - 1].strip().endswith("?"):
            tot += 1
            if len(tokenize(utts[i])) <= max_len:
                c += 1
    return c / max(1, tot)

def segmented_with_hyphen_per_utt(raw, n):
    pat = r"\b[a-z]{1,4}-[a-z]{1,4}\b"
    return len(re.findall(pat, raw.lower())) / max(1, n)

def long_glued_tokens_per1000w(tokens):
    return 1000 * sum(1 for t in tokens if len(t) >= 15) / max(1, len(tokens))

def listy_utterances(utts, conj="and", thr=3):
    return sum(u.lower().count(f" {conj} ") >= thr for u in utts) / max(1, len(utts))

def many_commas_or_semicolons(utts, thr=3):
    return sum((u.count(",") + u.count(";")) >= thr for u in utts) / max(1, len(utts))

def qualifiers_per1000w(tokens): return 1000 * sum(t in QUALIFIERS for t in tokens) / max(1, len(tokens))

def caps_rate_per_utt(utts):
    def has_caps(u):
        return any(len(w) >= 3 and w.isupper() for w in u.split())
    return sum(1 for u in utts if has_caps(u)) / max(1, len(utts))

# ===================== Severity weights =====================
SEVERITY = {
    "T01":3,"T02":3,"T03":3,"T04":2,"T05":2,"T06":2,"T07":2,"T08":2,"T09":1,"T10":1,
    "T11":2,"T12":2,"T13":2,"T14":1,"T15":2,"T16":2,"T17":1,"T18":1,"T19":1,"T20":2,
    "T21":2,"T22":1,"T23":2,"T24":2,"T25":1,"T26":2,"T27":1,"T28":2,"T29":1,"T30":1,
    "T31":1,"T32":1,"T33":1,"T34":1,"T35":1,"T36":1,"T37":1,"T38":1,"T39":1,"T40":1,
    "T41":1,"T42":1,"T43":1,"T44":1,"T45":1,"T46":1,"T47":2,"T48":2,"T49":1,"T50":1
}

# ===================== Full feature extraction =====================
def extract_features(utts, raw):
    tokens = [t for u in utts for t in tokenize(u)]
    n = len(utts)
    low = raw.lower()

    feats = {
        "T01_avg_len": avg_utterance_length(utts),
        "T01_long_rate": long_utterance_rate(utts),
        "T02_overrun": overrun_rate(utts),
        "T03_interruptions_per100u": interruptions_per100u(raw, n),
        "T04_topic_switch_rate": topic_switch_rate(utts),
        "T05_blurting_rate": impulsive_after_question(utts, max_len=2),
        "T06_overlap_rate": interruptions_per100u(raw, n),
        "T07_fillers_per100w": fillers_per1000w(tokens),         # (per 1000 words)
        "T08_disfluencies_per100w": disfluencies_per1000w(raw, tokens),
        "T10_formal_per100w": formal_words_per1000w(tokens),
        "T11_exclam_perutt": exclamations_per_utt(raw, n),
        "T12_inattentiveness": sum(ph in low for ph in ["lost my train of thought","what were we","distracted","forgot","i forgot"]) / max(1, n),
        "T13_impulsive": impulsive_after_question(utts, max_len=3),
        "T14_wm_lapse": sum(ph in low for ph in ["where did i","what were we doing","i forgot"]) / max(1, n),
        "T15_motor": sum(ph in low for ph in ["(fidgets)","(taps)","(moves)","(shifts)","(sways)","(bounces)"]) / max(1, n),
        "T16_delay": sum(ph in low for ph in ["move on","hurry","can't wait","can we just go on","skip"]) / max(1, n),
        "T17_gesturing": sum(ph in low for ph in ["(gestures)","(waves)","(points)","(gesticulates)"]) / max(1, n),
        "T18_sensory": sum(ph in low for ph in ["did you hear","honk","beep","sirens","noise","loud"]) / max(1, n),
        "T19_reward": sum(ph in low for ph in ["tell me now","do i get","when do we start","when will it start","now now"]) / max(1, n),
        "T20_emotional": (sum(ph in low for ph in ["not fair","angry","mad","upset"]) + raw.count("!")) / max(1, n),
        "T22_monosyll": sum(len(tokenize(u)) == 1 for u in utts) / max(1, len(utts)),
        "T23_qrate": sum(u.strip().endswith("?") for u in utts) / max(1, len(utts)),
        "T24_multiclause": many_commas_or_semicolons(utts) + listy_utterances(utts, "and", thr=3),
        "T25_fragments": sum((len(tokenize(u)) <= 3) or (not u.strip().endswith((".", "!", "?"))) for u in utts) / max(1, len(utts)),
        "T26_fillers_per100w": fillers_per1000w(tokens),
        "T27_laughter": sum(ph in low for ph in ["(laughs)","(laughing)","(giggles)"]) / max(1, n),
        "T28_caps_rate": caps_rate_per_utt(utts),
        "T29_garbled_per100w": long_glued_tokens_per1000w(tokens),
        "T30_selfrepair": sum((" i mean " in " " + u.lower() + " ") for u in utts) / max(1, len(utts)),
        "T31_auditory_overload": sum(ph in low for ph in ["too loud","cover my ears","covers ears","hurts my ears"]) / max(1, n),
        "T32_sensory_hyperfocus": sum(ph in low for ph in ["hum of the fan","texture","rhythm","pattern","the sound of","buzzing"]) / max(1, n),
        "T33_oral": sum(ph in low for ph in ["tsk","ts-ts","smack","mmm","hmm"]) / max(1, n),
        "T34_peripheral_awareness": sum(ph in low for ph in ["who is that over there","over there","look over there","who is that"]) / max(1, n),
        "T35_rapid_recap": sum(ph in low for ph in ["basically","in short","long story short","so basically"]) / max(1, n),
        "T36_over_enthusiasm": sum(ph in low for ph in ["so excited","amazing","awesome","great!","so happy"]) / max(1, n),
        "T37_forced_humor": sum(ph in low for ph in ["what did the","joke","haha","ha ha"]) / max(1, n),
        "T38_eye_contact": sum(ph in low for ph in ["(looks away)","(stares off)","(gazes away)"]) / max(1, n),
        "T40_meta_talk": sum(ph in low for ph in ["am i talking too much","turn-taking","talk about talking","let me talk about"]) / max(1, n),
        "T41_peripheral_voc": sum(ph in low for ph in ["sigh","whistle","whistling","sighs"]) / max(1, n),
        "T42_segmented": segmented_with_hyphen_per_utt(raw, n),
        "T43_qualifiers_per1000w": qualifiers_per1000w(tokens),
        "T44_pseudo_q": sum(any(u.lower().strip().endswith(tag) for tag in ["right?","isn't it?","innit?","okay?","ok?","yeah?","no?"]) for u in utts) / max(1, len(utts)),
        "T45_immediate_repair": len(re.findall(r"\b[a-z]{1,5}[—-]\s?[a-z]{2,}\b", raw.lower())) / max(1, n),
        "T46_spatial_drift": sum(ph in low for ph in ["(leans)","(leans forward)","(drifts)","(shifts weight)"]) / max(1, n),
        "T47_content_overload": sum((len(tokenize(u)) >= 30) and (sum(w in QUALIFIERS for w in tokenize(u)) >= 3) for u in utts) / max(1, len(utts)),
        "T48_rapid_ideation": listy_utterances(utts, "or", thr=3) + listy_utterances(utts, "and", thr=4),
        "T49_externalization": sum(ph in low for ph in ["hmm","let me think","i'm thinking","should i","i wonder"]) / max(1, n),
        "T50_anticipatory": sum(ph in low for ph in ["once we finish","after we","then we'll","we will","when we finish","later we'll"]) / max(1, n),
        "N_utts": n, "Total_tokens": len(tokens),
    }

    # T09, T21, T39 specialized:
    echo = sum(1 for i in range(1, len(utts))
               if jaccard_similarity(utts[i - 1], utts[i]) >= 0.6 and len(tokenize(utts[i])) >= 3)
    feats["T09_echolalia"] = echo / max(1, len(utts) - 1)

    norm = [re.sub(r"\s+", " ", u.strip().lower()) for u in utts]
    feats["T21_perseverative"] = sum(v - 1 for v in Counter(norm).values() if v > 1) / max(1, len(norm))

    qs = [re.sub(r"\s+", " ", u.strip().lower()) for u in utts if u.strip().endswith("?")]
    feats["T39_rep_questions"] = sum(v - 1 for v in Counter(qs).values() if v > 1) / max(1, len(qs))
    return feats

# ===================== ADHD-like index & confidence =====================
def adhd_index(df: pd.DataFrame) -> np.ndarray:
    cols = [c for c in df.columns if TRAIT_COL_RE.match(c)]
    if not cols:
        return np.zeros(len(df), dtype=float)
    Z = (df[cols] - df[cols].mean()) / (df[cols].std() + 1e-8)
    w = np.array([SEVERITY.get(re.match(r"^(T\d{2})_", c).group(1), 1.0) for c in cols], dtype=float)
    return (Z.values * w).sum(axis=1) / (w.sum() + 1e-8)

def soft_cluster_confidence(X, labels):
    K = labels.max() + 1
    centers = np.vstack([X[labels == k].mean(axis=0) for k in range(K)])
    sims = cosine_similarity(X, centers)
    e = np.exp(sims - sims.max(axis=1, keepdims=True))
    P = e / e.sum(axis=1, keepdims=True)
    return P[np.arange(X.shape[0]), labels], P

# ===================== LLM results loader =====================
def load_llm_summary(llm_dir: Path):
    if not llm_dir.exists():
        return None
    rows = []
    for csv in llm_dir.glob("*.csv"):
        try:
            df = pd.read_csv(csv)
        except Exception:
            continue
        base = csv.stem

        if "Traits" in df.columns:
            def parse(x):
                if pd.isna(x): return []
                s = str(x).strip()
                if s.upper() == "NONE" or s == "": return []
                return [p for p in re.split(r"[;,]\s*|\s+", s) if re.fullmatch(r"T\d{2}", p)]
            L = df["Traits"].apply(parse)
            any_frac = float(L.apply(lambda lst: 1 if lst else 0).mean()) if len(L) > 0 else 0.0
            row = {"basename": base, "adhd_llm_pct": any_frac}
            for t in TRAIT_IDS:
                row[f"LLM_{t}_pct"] = float(L.apply(lambda lst: 1 if t in lst else 0).mean()) if len(L) > 0 else np.nan
            rows.append(row)

        elif any(c in df.columns for c in TRAIT_IDS):
            trait_cols = [c for c in df.columns if c in TRAIT_IDS]
            any_frac = float((df[trait_cols].fillna(0).astype(float).sum(axis=1) > 0).mean()) if len(df) > 0 else 0.0
            row = {"basename": base, "adhd_llm_pct": any_frac}
            for t in TRAIT_IDS:
                row[f"LLM_{t}_pct"] = float(pd.to_numeric(df.get(t, 0), errors="coerce").fillna(0).astype(float).mean()) if t in df.columns else np.nan
            rows.append(row)

        elif set(["utterance_id", "trait_id", "present"]).issubset(df.columns):
            df["present"] = pd.to_numeric(df["present"], errors="coerce").fillna(0).astype(int)
            any_frac = float(df.groupby("utterance_id")["present"].max().mean())
            mat = df.pivot_table(index="utterance_id", columns="trait_id",
                                 values="present", aggfunc="max").fillna(0)
            row = {"basename": base, "adhd_llm_pct": any_frac}
            for t in TRAIT_IDS:
                row[f"LLM_{t}_pct"] = float(mat[t].mean()) if t in mat.columns else np.nan
            rows.append(row)

    return pd.DataFrame(rows) if rows else None

# ===================== Domain mapping & scores =====================
DOMAIN_MAP = {
    "Hyperactivity": ["T01","T02","T03","T04","T07","T08","T15","T17","T18","T26",
                      "T27","T28","T33","T34","T36","T41","T46","T47","T48"],
    "Impulsivity":   ["T05","T06","T11","T13","T16","T19","T20","T24","T30","T37","T40","T44","T45","T50"],
    "Inattention":   ["T09","T10","T12","T14","T21","T22","T23","T25","T29","T31","T32","T35","T38","T39","T42","T43","T49"],
}
def ml_domain_scores(feats_df: pd.DataFrame) -> pd.DataFrame:
    trait_cols = [c for c in feats_df.columns if TRAIT_COL_RE.match(c)]
    Z = (feats_df[trait_cols] - feats_df[trait_cols].mean()) / (feats_df[trait_cols].std() + 1e-8)
    out = {}
    for dom, tids in DOMAIN_MAP.items():
        cols = []
        for t in tids:
            cols.extend([c for c in Z.columns if c.startswith(f"{t}_")])
        out[f"ML_{dom}"] = Z[cols].mean(axis=1) if cols else pd.Series(0.0, index=feats_df.index)
    return pd.DataFrame(out, index=feats_df.index)

def llm_domain_scores(llm_summary: pd.DataFrame) -> pd.DataFrame:
    out = {}
    for dom, tids in DOMAIN_MAP.items():
        cols = [f"LLM_{t}_pct" for t in tids if f"LLM_{t}_pct" in llm_summary.columns]
        out[f"LLM_{dom}"] = llm_summary[cols].mean(axis=1) if cols else pd.Series(np.nan, index=llm_summary.index)
    return pd.DataFrame(out, index=llm_summary.index)

# ===================== Plot helpers =====================
def plot_dendrogram(Z_std, labels, k):
    L = linkage(Z_std, method="ward")
    cth = L[-(k - 1), 2] if k > 1 else None
    fig = plt.figure(figsize=(10, 4.8))
    dendrogram(L, labels=labels, orientation="top", distance_sort="ascending",
               leaf_rotation=90, color_threshold=cth)
    plt.title(f"Agglomerative (Ward) — Dendrogram (k={k})")
    plt.tight_layout()
    plt.show()

def plot_silhouette_curve(Z_std, k_min=2, k_max=8):
    ks = list(range(k_min, k_max + 1))
    vals = []
    for k in ks:
        labels = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean").fit_predict(Z_std)
        vals.append(silhouette_score(Z_std, labels, metric="euclidean"))
    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.gca()
    ax.plot(ks, vals, marker="o", linewidth=2)
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette coefficient")
    ax.set_title("Silhouette coefficient curve")
    fig.tight_layout()
    plt.show()
    print("Silhouette by k:", {k: round(v, 3) for k, v in zip(ks, vals)})

def heatmap_clean(matrix, row_labels, col_labels, title):
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    im = ax.imshow(matrix, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, "–" if np.isnan(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=11)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    plt.show()

def _single_scatter(ax, p):
    ax.scatter(p["x"], p["y"], alpha=0.8)
    a, b = np.polyfit(p["x"], p["y"], 1)
    xs = np.linspace(p["x"].min(), p["x"].max(), 100)
    ax.plot(xs, a * xs + b, ls="--")
    ax.set_title(p["feat"], fontsize=12, pad=6)
    stats = (f"Pearson r={p['pr']:.2f} (p={p['pp']:.3f})\n"
             f"Spearman r={p['sr']:.2f} (p={p['sp']:.3f})")
    ax.text(0.02, 0.98, stats, transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75, ec="none"))
    ax.set_xlabel(p["feat"])
    ax.set_ylabel("LLM ADHD-like (%)")
    ax.margins(x=0)

def plot_scatter_panels(panels):
    """
    Render two scatter panels stacked vertically in ONE figure,
    labeling them 'Fig 6.a' (top) and 'Fig 6.b' (bottom) at the top-left
    OUTSIDE each axes for readability. Falls back to horizontal layout if n != 2.
    """
    n = len(panels)
    if n == 0:
        return

    if n == 2:
        fig, axes = plt.subplots(2, 1, figsize=(7.2, 9.2),
                                 constrained_layout=True, sharex=False, sharey=False)
        letters = ["a", "b"]
        for ax, p, L in zip(axes, panels, letters):
            _single_scatter(ax, p)
            # panel label: top-left, slightly above the axes
            ax.text(0.0, 1.08, f"Fig 6.{L}",
                    transform=ax.transAxes,
                    ha="left", va="bottom",
                    fontsize=13, weight="bold",
                    clip_on=False)
        plt.show()
        return

    # Fallback if panel count differs
    fig, axes = plt.subplots(1, n, figsize=(6.8 * n, 4.2), constrained_layout=True)
    if n == 1:
        axes = [axes]
    for ax, p in zip(axes, panels):
        _single_scatter(ax, p)
    plt.show()


def visualize_all(Z_std, labels, df, k, merged, LLM, k_curve_min, k_curve_max):
    use_plot_style()
    plot_dendrogram(Z_std, df.index.tolist(), k)
    plot_silhouette_curve(Z_std, k_min=k_curve_min, k_max=k_curve_max)
    panels = []
    if "adhd_llm_pct" in merged.columns:
        y = pd.to_numeric(merged["adhd_llm_pct"], errors="coerce") * 100.0
        for feat in ["ADHD_index", "prob_ADHD_like"]:
            if feat not in merged.columns:
                continue
            x = pd.to_numeric(merged[feat], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() >= 3:
                pr, pp = pearsonr(x[m], y[m]); sr, sp = spearmanr(x[m], y[m])
                panels.append({"feat": feat, "x": x[m].values, "y": y[m].values,
                               "pr": pr, "pp": pp, "sr": sr, "sp": sp})
    # === now draw stacked panels in the SAME figure, labeled Fig 6.a / Fig 6.b
    plot_scatter_panels(panels)

    if (LLM is not None) and ("adhd_llm_pct" in merged.columns):
        ml_dom = ml_domain_scores(df)
        llm_index = "merge_key" if "merge_key" in LLM.columns else "basename"
        llm_dom = llm_domain_scores(LLM.set_index(llm_index))
        merged2 = merged.set_index("Transcript").join(ml_dom).reset_index()
        merged2 = merged2.join(llm_dom, on="merge_key")

        rows = []
        for dom in DOMAIN_MAP.keys():
            x = pd.to_numeric(merged2[f"ML_{dom}"], errors="coerce")
            y = pd.to_numeric(merged2[f"LLM_{dom}"], errors="coerce")
            m = x.notna() & y.notna()
            if m.sum() >= 3:
                r_p, p_p = pearsonr(x[m], y[m]); r_s, p_s = spearmanr(x[m], y[m])
            else:
                r_p = p_p = r_s = p_s = np.nan
            rows.append({"Domain": dom, "Pearson r": r_p, "Pearson p": p_p, "Spearman r": r_s, "Spearman p": p_s})
        dom_corr = pd.DataFrame(rows)
        matrix = dom_corr.set_index("Domain")[["Pearson r", "Spearman r"]].values
        heatmap_clean(matrix, list(DOMAIN_MAP.keys()), ["Pearson r", "Spearman r"], "ML vs LLM — Domain correlations")

# ===================== Per-utterance trait annotation =====================
def annotate_utterances(utts):
    rows = []
    seen_norm = Counter()
    seen_questions = Counter()

    def has_caps(u): return any(len(w) >= 3 and w.isupper() for w in u.split())
    def any_phrase(u_low, phrases): return any(ph in u_low for ph in phrases)

    for i, u in enumerate(utts):
        u_stripped = u.strip()
        u_low = u_stripped.lower()
        toks = tokenize(u_stripped)
        n_tok = len(toks)
        flags = {t: 0 for t in TRAIT_IDS}

        if n_tok >= 20: flags["T01"] = 1
        if (u_stripped.count(",") + u_stripped.count(";")) >= 3 and "." not in u_stripped: flags["T02"] = 1
        if re.search(r"\[//\]|\[/\]|//|\[\/\]", u_stripped): flags["T03"] = 1; flags["T06"] = 1
        if i > 0 and jaccard_similarity(utts[i-1], u_stripped) <= 0.1: flags["T04"] = 1
        if i > 0 and utts[i-1].strip().endswith("?") and n_tok <= 2: flags["T05"] = 1
        if any(t in FILLERS for t in toks) or ("you" in toks and "know" in toks): flags["T07"] = 1; flags["T26"] = 1
        if re.search(r"\b(\w+)\s+\1\b", u_low) or ("--" in u_stripped): flags["T08"] = 1
        if i > 0 and jaccard_similarity(utts[i-1], u_stripped) >= 0.6 and n_tok >= 3: flags["T09"] = 1
        if any(t in FORMAL_WORDS for t in toks): flags["T10"] = 1
        if "!" in u_stripped: flags["T11"] = 1; flags["T20"] = 1
        if any_phrase(u_low, ["lost my train of thought","what were we","distracted","forgot","i forgot"]): flags["T12"] = 1
        if i > 0 and utts[i-1].strip().endswith("?") and n_tok <= 3: flags["T13"] = 1
        if any_phrase(u_low, ["where did i","what were we doing","i forgot"]): flags["T14"] = 1
        if any_phrase(u_low, ["(fidgets)","(taps)","(moves)","(shifts)","(sways)","(bounces)"]): flags["T15"] = 1
        if any_phrase(u_low, ["move on","hurry","can't wait","can we just go on","skip"]): flags["T16"] = 1
        if any_phrase(u_low, ["(gestures)","(waves)","(points)","(gesticulates)"]): flags["T17"] = 1
        if any_phrase(u_low, ["did you hear","honk","beep","sirens","noise","loud"]): flags["T18"] = 1
        if any_phrase(u_low, ["tell me now","do i get","when do we start","when will it start","now now"]): flags["T19"] = 1
        if any_phrase(u_low, ["not fair","angry","mad","upset"]): flags["T20"] = 1
        norm = re.sub(r"\s+", " ", u_low)
        if seen_norm[norm] >= 1: flags["T21"] = 1
        seen_norm[norm] += 1
        if n_tok == 1: flags["T22"] = 1
        if u_stripped.endswith("?"): flags["T23"] = 1
        if (u_stripped.count(",") + u_stripped.count(";")) >= 3 or u_low.count(" and ") >= 3: flags["T24"] = 1
        if (n_tok <= 3) or (not u_stripped.endswith((".", "!", "?"))): flags["T25"] = 1
        if any_phrase(u_low, ["(laughs)","(laughing)","(giggles)"]): flags["T27"] = 1
        if has_caps(u_stripped): flags["T28"] = 1
        if any(len(t) >= 15 for t in toks): flags["T29"] = 1
        if " i mean " in f" {u_low} ": flags["T30"] = 1
        if any_phrase(u_low, ["too loud","cover my ears","covers ears","hurts my ears"]): flags["T31"] = 1
        if any_phrase(u_low, ["hum of the fan","texture","rhythm","pattern","the sound of","buzzing"]): flags["T32"] = 1
        if any_phrase(u_low, ["tsk","ts-ts","smack","mmm","hmm"]): flags["T33"] = 1
        if any_phrase(u_low, ["who is that over there","over there","look over there","who is that"]): flags["T34"] = 1
        if any_phrase(u_low, ["basically","in short","long story short","so basically"]): flags["T35"] = 1
        if any_phrase(u_low, ["so excited","amazing","awesome","great!","so happy"]): flags["T36"] = 1
        if any_phrase(u_low, ["what did the","joke","haha","ha ha"]): flags["T37"] = 1
        if any_phrase(u_low, ["(looks away)","(stares off)","(gazes away)"]): flags["T38"] = 1
        if u_stripped.endswith("?"):
            if seen_questions[norm] >= 1: flags["T39"] = 1
            seen_questions[norm] += 1
        if any_phrase(u_low, ["am i talking too much","turn-taking","talk about talking","let me talk about"]): flags["T40"] = 1
        if any_phrase(u_low, ["sigh","whistle","whistling","sighs"]): flags["T41"] = 1
        if re.search(r"\b[a-z]{1,4}-[a-z]{1,4}\b", u_low): flags["T42"] = 1
        if any(t in QUALIFIERS for t in toks): flags["T43"] = 1
        if any(u_low.endswith(tag) for tag in ["right?","isn't it?","innit?","okay?","ok?","yeah?","no?"]): flags["T44"] = 1
        if re.search(r"\b[a-z]{1,5}[—-]\s?[a-z]{2,}\b", u_low): flags["T45"] = 1
        if any_phrase(u_low, ["(leans)","(leans forward)","(drifts)","(shifts weight)"]): flags["T46"] = 1
        if (n_tok >= 30) and (sum(w in QUALIFIERS for w in toks) >= 3): flags["T47"] = 1
        if (u_low.count(" or ") >= 3) or (u_low.count(" and ") >= 4): flags["T48"] = 1
        if any_phrase(u_low, ["hmm","let me think","i'm thinking","should i","i wonder"]): flags["T49"] = 1
        if any_phrase(u_low, ["once we finish","after we","then we'll","we will","when we finish","later we'll"]): flags["T50"] = 1

        codes = [t for t, v in flags.items() if v == 1]
        rows.append({
            "utterance_id": i + 1,
            "utterance_text": u_stripped,
            "Traits_found_codes": ", ".join(codes),
            "Traits_found_names": ", ".join(T_LABELS.get(t, t) for t in codes),
            **{t: flags[t] for t in TRAIT_IDS},
        })
    return rows

# ===================== Per-child utterance CSV (present & used) =====================
def export_one_child_conversation_csv(
    df_clustering: pd.DataFrame,
    k: int,
    example_mode=("random", None),
    seed: int = 42,
) -> None:
    """
    Export per-utterance trait flags for a single child to
    Results/ML+LLM/one_child_utterance_traits.csv.
    """
    paths = sorted(TRANSCRIPTS_DIR.glob("*.cha"))
    if not paths:
        print("No .cha transcripts found in Nadig/. Skipping one-child CSV.")
        return

    # Choose which child
    random.seed(seed)
    if example_mode[0] == "name" and example_mode[1]:
        chosen = next((p for p in paths if p.name == example_mode[1]), paths[0])
    else:
        chosen = random.choice(paths)

    utts, raw = read_transcript(chosen)
    anno = annotate_utterances(utts)
    child_key = chosen.name

    # Pull cluster/meta if available for this child
    if child_key in df_clustering.index:
        row = df_clustering.loc[child_key]
        cluster = int(row.get("cluster", -1)) if not pd.isna(row.get("cluster", np.nan)) else -1
        cluster_conf = float(row.get("cluster_conf", np.nan)) if "cluster_conf" in df_clustering.columns else np.nan
        adhd_idx = float(row.get("ADHD_index", np.nan))
        prob_like = (
            float(row.get("prob_ADHD_like", np.nan))
            if ("prob_ADHD_like" in df_clustering.columns and k == 2)
            else np.nan
        )
    else:
        cluster = -1
        cluster_conf = np.nan
        adhd_idx = np.nan
        prob_like = np.nan

    # Attach metadata to each utterance row
    for r in anno:
        r["Transcript"] = child_key
        r["cluster"] = cluster
        r["cluster_conf"] = cluster_conf
        r["prob_ADHD_like"] = prob_like
        r["ADHD_index"] = adhd_idx

    out_df = pd.DataFrame(
        anno,
        columns=[
            "Transcript",
            "cluster",
            "cluster_conf",
            "prob_ADHD_like",
            "ADHD_index",
            "utterance_id",
            "utterance_text",
            "Traits_found_codes",
            "Traits_found_names",
            *TRAIT_IDS,
        ],
    )
    out_path = OUTPUT_DIR / "one_child_utterance_traits.csv"
    out_df.to_csv(out_path, index=False)
    print(f"Saved one_child_utterance_traits.csv (child: {child_key})")

# ===================== Cluster comparison table (CSV + LaTeX) =====================
def _fmt_mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.2f} ({sd:.2f})"

def _cohens_d_c1_minus_c0(c0: np.ndarray, c1: np.ndarray) -> float:
    """Cohen's d defined as (mean_cluster1 - mean_cluster0)/pooled_sd."""
    c0 = np.asarray(c0, dtype=float)
    c1 = np.asarray(c1, dtype=float)
    c0 = c0[~np.isnan(c0)]
    c1 = c1[~np.isnan(c1)]
    if len(c0) < 2 or len(c1) < 2:
        return np.nan
    m0, m1 = c0.mean(), c1.mean()
    s0, s1 = c0.std(ddof=1), c1.std(ddof=1)
    pooled = np.sqrt(((len(c0) - 1) * s0**2 + (len(c1) - 1) * s1**2) / (len(c0) + len(c1) - 2))
    if pooled == 0 or np.isnan(pooled):
        return np.nan
    return (m1 - m0) / pooled

def make_and_save_cluster_comparison(df: pd.DataFrame, k: int) -> None:
    """
    For k==2, compute the comparison table across selected metrics,
    save a tidy CSV with numeric values and a LaTeX table mirroring your example.
    """
    if "cluster" not in df.columns or k != 2:
        print("Cluster comparison table: only produced when k==2 and 'cluster' column exists.")
        return

    metrics = [
        ("Avg utterance length (tokens)", "T01_avg_len"),
        ("Fillers per 1000 words",       "T07_fillers_per100w"),
        ("Interruptions per 100 utts",   "T03_interruptions_per100u"),
        ("Topic-switch rate",            "T04_topic_switch_rate"),
        ("Fragment rate",                "T25_fragments"),
        ("ADHD index",                   "ADHD_index"),
    ]

    rows_num = []
    have_any = False
    for label, col in metrics:
        if col not in df.columns:
            print(f"[warn] Metric column missing: {col} — skipping in comparison table.")
            continue

        v0 = pd.to_numeric(df.loc[df["cluster"] == 0, col], errors="coerce")
        v1 = pd.to_numeric(df.loc[df["cluster"] == 1, col], errors="coerce")
        m0, s0 = float(np.nanmean(v0)), float(np.nanstd(v0, ddof=1))
        m1, s1 = float(np.nanmean(v1)), float(np.nanstd(v1, ddof=1))
        d = float(_cohens_d_c1_minus_c0(v0.values, v1.values))

        rows_num.append({
            "Metric": label,
            "cluster0_mean": m0, "cluster0_sd": s0,
            "cluster1_mean": m1, "cluster1_sd": s1,
            "cohens_d": d,
        })
        have_any = True

    if not have_any:
        print("No metrics available to build the cluster comparison table.")
        return

    out_csv = OUTPUT_DIR / "cluster_comparison.csv"
    pd.DataFrame(rows_num).to_csv(out_csv, index=False)
    print(f"Saved cluster comparison CSV: {out_csv}")

# ===================== Main pipeline =====================
def main(k=2, k_curve_min=2, k_curve_max=8, example_child=None, show_plots=True):
    use_plot_style()

    # Extract features from transcripts
    feats = []
    for p in sorted(TRANSCRIPTS_DIR.glob("*.cha")):
        utts, raw = read_transcript(p)
        d = extract_features(utts, raw)
        d["Transcript"] = p.name
        d["basename"] = p.stem
        d["merge_key"] = normalize_filename(p.name)
        feats.append(d)
    if not feats:
        print("No .cha transcripts found in Nadig/. Nothing to process.")
        return

    df = pd.DataFrame(feats).set_index("Transcript")

    # Order trait cols for readability
    def sort_trait_cols(cols):
        def key(c):
            m = re.match(r"^T(\d{2})_(.*)$", c)
            return (int(m.group(1)) if m else 999, c)
        return sorted(cols, key=key)
    trait_cols = sort_trait_cols([c for c in df.columns if TRAIT_COL_RE.match(c)])
    df = df[trait_cols + [c for c in df.columns if c not in trait_cols]]

    # Model matrix
    Z = StandardScaler().fit_transform(df[trait_cols].fillna(0.0).values)

    # Index + clustering
    df["ADHD_index"] = adhd_index(df)
    agg = AgglomerativeClustering(n_clusters=k, linkage="ward", metric="euclidean")
    labels = agg.fit_predict(Z)
    df["cluster"] = labels
    conf, P = soft_cluster_confidence(Z, labels)
    df["cluster_conf"] = conf
    if k == 2:
        means = [df.loc[df.cluster == c, "ADHD_index"].mean() for c in range(2)]
        adhd_like = int(np.argmax(means))
        df["prob_ADHD_like"] = P[:, adhd_like]

    # Save clustering results CSV
    df.to_csv(OUTPUT_DIR / "clustering_results.csv")
    print("Saved clustering_results.csv")

    # Per-utterance CSV for one child
    export_one_child_conversation_csv(
        df, k,
        example_mode=("name", example_child) if example_child else ("random", None),
        seed=42
    )

    # LLM merge + LLM summary CSV
    LLM = load_llm_summary(LLM_RESULTS_DIR)
    if LLM is None or LLM.empty:
        print(f"No LLM CSVs found in '{LLM_RESULTS_DIR}'. Skipping LLM merges.")
        merged = df.reset_index()
        pd.DataFrame(columns=["basename","merge_key","adhd_llm_frac","adhd_llm_pct"]).to_csv(
            OUTPUT_DIR / "llm_summary_adhd_pct.csv", index=False)
    else:
        LLM["merge_key"] = LLM["basename"].map(normalize_filename)
        llm_out_cols = ["basename", "merge_key", "adhd_llm_pct"]
        llm_trait_cols = [c for c in LLM.columns if c.startswith("LLM_T") and c.endswith("_pct")]
        llm_export = LLM[llm_out_cols + llm_trait_cols].copy()
        llm_export.rename(columns={"adhd_llm_pct": "adhd_llm_frac"}, inplace=True)
        llm_export["adhd_llm_pct"] = pd.to_numeric(llm_export["adhd_llm_frac"], errors="coerce") * 100.0
        llm_export.to_csv(OUTPUT_DIR / "llm_summary_adhd_pct.csv", index=False)
        print("Saved llm_summary_adhd_pct.csv")
        merged = df.reset_index().merge(LLM, on="merge_key", how="left")

    merged.to_csv(OUTPUT_DIR / "merged_with_llm.csv", index=False)
    print("Saved merged_with_llm.csv")

    # ===== Cluster comparison table =====
    make_and_save_cluster_comparison(df.reset_index(), k=k)

    # Visuals
    if show_plots:
        visualize_all(Z_std=Z, labels=labels, df=df, k=k,
                      merged=merged, LLM=LLM, k_curve_min=k_curve_min, k_curve_max=k_curve_max)

    try:
        sil = silhouette_score(Z, labels, metric="euclidean")
        print(f"Silhouette score (k={k}): {sil:.3f}")
    except Exception as e:
        print(f"(Silhouette computation skipped: {e})")

# ===================== Script entry =====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-k", type=int, default=2)
    parser.add_argument("--kmin", type=int, default=2)
    parser.add_argument("--kmax", type=int, default=8)
    parser.add_argument("--child", type=str, default=None, help="Specific .cha filename for detailed CSV")
    parser.add_argument("--no-plots", action="store_true", help="Disable plotting")
    args = parser.parse_args()

    main(k=max(2, args.k),
         k_curve_min=args.kmin,
         k_curve_max=args.kmax,
         example_child=args.child,
         show_plots=not args.no_plots)
