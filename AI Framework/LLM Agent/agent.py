import os
import logging
import time
import pandas as pd
import openai
import re
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple

from parser import parse_cha_file
from HyperactivityKnowledge import load_knowledge

# --- Logging ------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# --- OpenAI Setup --------------------------------------------------------------
load_dotenv()
logger.info("Environment variables loaded.")
key = os.getenv("OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
if not key:
    logger.error("Missing OPENAI_KEY/AZURE_OPENAI_API_KEY in .env")
    exit(1)
openai.api_key = key

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
logger.info("Using model: %s", model_name)

# --- Helper: Format traits for prompt ------------------------------------------
def format_traits_for_prompt(traits: list) -> str:
    lines = []
    for t in traits:
        example = t.get("examples", [""])[0]
        lines.append(f"- {t['id']}: {t['label']} — {t['description']} (e.g. \"{example}\")")
    return "\n".join(lines)

# --- Clinical Validation -------------------------------------------------------
def validate_classification(
    utterance: str,
    speaker: str,
    traits: List[str],
    justification: str
) -> Tuple[List[str], str]:
    normal_patterns = [
        r'\b(okay|yes|no|yeah|ya|hm|um|uh)\b',
        r'\b(can you|would you|do you want|are you)\b',
        r'\b(thank you|thanks)\b',
        r'\b(oh|wow|aw)\b',
        r'\b(what|where|when|how|why)\b',
        r'\b(ready|sure|fine|alright)\b'
    ]
    utterance_lower = utterance.lower()
    is_normal = any(re.search(p, utterance_lower) for p in normal_patterns)
    if is_normal and len(utterance.split()) <= 3:
        return ["NONE"], "normal conversational response"

    corrected = []
    for trait in traits:
        if trait == "NONE":
            corrected.append(trait)
            continue
        if trait == "T13" and "help" in utterance_lower and speaker == "MOT":
            continue
        if trait == "T05" and "?" in utterance and speaker == "MOT":
            continue
        if trait == "T11" and "wow" in utterance_lower:
            continue
        if trait == "T12" and len(utterance.split()) <= 2:
            continue
        corrected.append(trait)

    if not corrected:
        corrected = ["NONE"]
        justification = "normal conversational behavior"
    return corrected, justification

# --- LLM Response Parsing ------------------------------------------------------
def parse_llm_response(line: str, valid_ids: set) -> Optional[Dict]:
    line = line.strip()
    if not line:
        return None

    parsed = None
    # Strategy 1
    if "]: " in line and " – " in line:
        try:
            left, rest = line.split("]: ", 1)
            idx = int(left.lstrip("[").strip())
            ids_part, justification = rest.split(" – ", 1)
            trait_ids = [
                tid.strip()
                for tid in ids_part.split(",")
                if tid.strip() in valid_ids or tid.strip() == "NONE"
            ]
            parsed = {
                "index": idx,
                "traits": trait_ids,
                "justification": justification.strip()
            }
        except Exception:
            parsed = None

    if not parsed:
        logger.warning("Could not parse line: %s", line)
        return None

    # --- Extract Scores ---------------------------------------------------------
    scores = {}
    for key in ("clarity", "uniqueness",  "quality", "confidence"):
        m = re.search(rf"{key}\s*=\s*([0-9]*\.?[0-9]+)", line, re.IGNORECASE)
        scores[key] = float(m.group(1)) if m else 0.0
    parsed["scores"] = scores

    return parsed

# --- LLM Helper ----------------------------------------------------------------
def query_llm(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.0
) -> str:
    start = time.time()
    resp = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=500
    )
    logger.info("LLM call took %.2f sec", time.time() - start)
    content = resp.choices[0].message.content
    if content is None:
        logger.error("LLM returned None content")
        raise ValueError("LLM response content is None")
    return content

# --- Main Agent ----------------------------------------------------------------
def main():
    logger.info("Agent started.")
    cha_path = input("Enter the path to your .cha file: ").strip()

    # Parse input file
    try:
        df = parse_cha_file(cha_path)
        logger.info("Parsed %d utterances.", len(df))
    except Exception as e:
        logger.error("Parsing failed: %s", e, exc_info=True)
        return

    # Load trait knowledge
    try:
        traits = load_knowledge()
        total = len(traits)
        ids = {t["id"] for t in traits}
        logger.info("Loaded %d traits.", total)
    except Exception as e:
        logger.error("Trait loading failed: %s", e, exc_info=True)
        return

    # Build system prompt (classification + evaluation metrics)
    system_prompt = f"""
# You are a researcher who is finding the traits of Hyperactivity among the transcripts and have to proof your hypothesis.
#
# IMPORTANT RULES:
# 1. Classify if you believe that conversation exhibits hyperactivity traits.
# 2. If conversation do not exhibit any trait then make it 'No Traits Found'
# 3. Consider speaker role: CHI=child, MOT=mother where child have Autism.
# 4. Do not be conservative, think twice before choosing NONE
#
# Respond with exactly one line per utterance:
# [index]: TRAIT_ID – brief reason
#
# Available traits: {','.join(sorted(ids))}
#
# Examples:
# [0]: T03 – child interrupts mother mid-sentence
# [1]: NONE – normal question
# [2]: T01 – child dominates conversation with long monologue
# [3]: NONE – simple response

Available traits: {','.join(sorted(ids))}

Trait definitions:
{format_traits_for_prompt(traits)}

Finally, for each utterance, that has been marked as having Hyperactivity Trait, append numeric scores [0.0–1.0] for:
  • clarity: Check if how much the utterance marked as Hyperactivity Trait is clear.[Perfectly shows signs - score closer to 1.0, if ambiguous, weak signs - score closer to 0.0
  • uniqueness: Check if only only one trait can fit the utterance - should be higher score (closer to 1.0), if multiple traits could apply - score closer to 0.0
  • quality: How strong and specific is your justification?  Use:
      0.0  – no justification or just one short word (“yes”/“no”)
      0.25 – very generic (“interrupts”) or off-hand
      0.50 – somewhat specific but missing context (“child cut in”)
      0.75 – detailed with context (“child cut mother off mid-sentence, showing impatience”)
      1.0  – highly detailed, references multiple cues and context
 Compute overall confidence = mean of these three.
 Format exactly:
 [index]: TRAIT_IDS – justification; clarity=0.75; uniqueness=1.00; confidence=0.85
""".strip()
    logger.info("System prompt ready.")

    results = []
    processed = set()
    batch_size = 50

    # Iterate in batches
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start+batch_size]
        entries = [
            f"[{idx}] Speaker: {row['Speaker']}\nUtterance: {row['Utterance']}"
            for idx, row in batch.iterrows()
        ]
        user_prompt = (
            "Classify each of the following utterances:\n\n"
            + "\n---\n".join(entries)
        )

        try:
            reply = query_llm(system_prompt, user_prompt, temperature=0.0)
            for line in reply.splitlines():
                parsed = parse_llm_response(line, ids)
                if not parsed:
                    continue

                idx = parsed["index"]
                if idx in processed:
                    logger.warning("Duplicate index %d, skipping", idx)
                    continue
                processed.add(idx)

                utterance = df.at[idx, "Utterance"]
                speaker   = df.at[idx, "Speaker"]
                validated_traits, validated_justification = validate_classification(
                    utterance,
                    speaker,
                    parsed["traits"],
                    parsed["justification"]
                )

                # --- Print & Store Confidence & Scores ----------------------------
                scores = parsed["scores"]
                conf   = scores.get("confidence", 0.0)
                print(
                    f"[{idx}] Confidence: {conf:.2f} "
                    f"(clarity={scores['clarity']:.2f}, "
                    f"uniqueness={scores['uniqueness']:.2f}, "
                    f"quality={scores['quality']:.2f})"
                )

                results.append({
                    "Index": idx,
                    "Timestamp": df.at[idx, "Timestamp"],
                    "Speaker": speaker,
                    "Utterance": utterance,
                    "Traits": ",".join(validated_traits),
                    "Justification": validated_justification,
                    "Confidence": round(conf, 2),
                    "ClarityScore": round(scores["clarity"], 2),
                    "UniquenessScore": round(scores["uniqueness"], 2),
                    "QualityScore": round(scores["quality"], 2),
                })

        except Exception as e:
            logger.error("Batch %d-%d error: %s", start, start+batch_size-1, e)

    # Save results
    outputs_dir = "Outputs"
    os.makedirs(outputs_dir, exist_ok=True)
    fname = os.path.splitext(os.path.basename(cha_path))[0]
    out_csv = os.path.join(outputs_dir, f"{fname}_analyzed.csv")
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Saved analysis to %s", out_csv)

    # Coverage report
    found = set()
    counts = {}
    for r in results:
        for tid in r["Traits"].split(","):
            if tid != "NONE":
                found.add(tid)
                counts[tid] = counts.get(tid, 0) + 1
    pct = len(found) / total * 100 if total else 0
    logger.info("Detected %d/%d traits (%.2f%%)", len(found), total, pct)

    # Summary
    print("\n=== Analysis Summary ===")
    print(f"Total utterances processed: {len(results)}")
    print(f"Traits detected: {len(found)}/{total} ({pct:.2f}% coverage)")
    print(f"Normal utterances: {sum(1 for r in results if r['Traits']=='NONE')}")

    # --- Mean Confidence & Metrics (flagged utterances, safe div) ---------------
    flagged = [r for r in results if r["Confidence"] > 0.0]
    n = len(flagged)

    if n > 0:
        mean_conf = sum(r["Confidence"] for r in flagged) / n
        mean_clarity = sum(r["ClarityScore"] for r in flagged) / n
        mean_uniqueness = sum(r["UniquenessScore"] for r in flagged) / n
        mean_quality = sum(r["QualityScore"] for r in flagged) / n
    else:
        mean_conf = mean_clarity = mean_uniqueness = mean_quality = 0.0

    print(
        f"\nMean confidence (flagged only): {mean_conf:.2f} "
        f"(Mean clarity={mean_clarity:.2f}, "
        f"Mean uniqueness={mean_uniqueness:.2f}, "
        f"Mean quality={mean_quality:.2f})"
    )

    if counts:
        print("\nTrait frequency:")
        for tid, c in sorted(counts.items()):
            name = next((t['label'] for t in traits if t['id']==tid), tid)
            print(f"  {tid} ({name}): {c} occurrences")

if __name__ == "__main__":
    main()
