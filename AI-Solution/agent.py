import os
import logging
import time
import pandas as pd
import openai
from dotenv import load_dotenv

from parser import parse_cha_file
from HyperactivityKnowledge import load_knowledge

# ─── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# ─── OpenAI Setup ──────────────────────────────────────────────────────────────
load_dotenv()
logger.info("Environment variables loaded.")
key = os.getenv("OPENAI_KEY") or os.getenv("AZURE_OPENAI_API_KEY")
if not key:
    logger.error("Missing OPENAI_KEY/AZURE_OPENAI_API_KEY in .env")
    exit(1)
openai.api_key = key

model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
logger.info("Using model: %s", model_name)

# ─── LLM Helpers ───────────────────────────────────────────────────────────────
def query_llm(system_prompt: str, user_prompt: str) -> str:
    start = time.time()
    resp = openai.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        temperature=0.0,
        max_tokens=500
    )
    elapsed = time.time() - start
    logger.info("LLM call took %.2f sec", elapsed)
    
    content = resp.choices[0].message.content
    if content is None:
        logger.error("LLM returned None content")
        raise ValueError("LLM response content is None")
    
    return content

def format_traits_for_prompt(traits: list) -> str:
    lines = []
    for t in traits:
        # include the ID so model can reference it
        example = t.get("examples", [""])[0]
        lines.append(f"- {t['id']}: {t['label']} — {t['description']} (e.g. \"{example}\")")
    return "\n".join(lines)

# ─── Main Agent ─────────────────────────────────────────────────────────────────
def main():
    logger.info("Agent started.")
    cha_path = input("Enter the path to your .cha file: ").strip()
    logger.info("Parsing file: %s", cha_path)

    try:
        df = parse_cha_file(cha_path)
        logger.info("Parsed %d utterances.", len(df))
    except Exception as e:
        logger.error("Parsing failed: %s", e, exc_info=True)
        return

    try:
        traits = load_knowledge()  # loads list of dicts with 'id', 'label', etc.
        total = len(traits)
        ids = {t["id"] for t in traits}
        logger.info("Loaded %d traits.", total)
    except Exception as e:
        logger.error("Trait loading failed: %s", e, exc_info=True)
        return

    # Build a system prompt that strictly instructs the LLM on the output format
    system_prompt = f"""
You are a clinical language expert assessing conversational utterances for hyperactivity traits.
You MUST respond with exactly one line per utterance, in this precise format:

[index]: ID1,ID2 – very brief justification

• [index] is the zero-based index from the input list.
• IDs must be drawn from this list (only these, comma-separated, no spaces): {','.join(sorted(ids))}
• If no traits apply, return "[index]: NONE – no traits".

Here are two clear examples of correct versus incorrect:

CORRECT:
[0]: T03 – speaker cuts in mid-sentence
[1]: NONE – polite turn-taking

INCORRECT (you must NOT do this):
“Utterance 0 shows interruptive speech because they cut off…”
“1: Talks too much.”
“[2] T02,T05 – example justification.”

Trait definitions to use:
{format_traits_for_prompt(traits)}
""".strip()
    logger.info("System prompt ready.")

    results = []
    batch_size = 50  # break into chunks if >50
    for start in range(0, len(df), batch_size):
        batch = df.iloc[start:start+batch_size]
        entries = [
            f"[{idx}] Speaker: {row['Speaker']}\nUtterance: {row['Utterance']}"
            for idx, row in batch.iterrows()
        ]
        user_prompt = (
            "Classify each of the following utterances:\n\n" +
            "\n---\n".join(entries)
        )

        try:
            reply = query_llm(system_prompt, user_prompt)
            for line in reply.splitlines():
                line = line.strip()
                if not line:
                    continue
                # parse "[idx]: ID1,ID2 – justification"
                try:
                    left, rest = line.split("]:", 1)
                    idx = int(left.strip().lstrip("["))
                    ids_part, justification = rest.split("–", 1)
                    picked = [i.strip() for i in ids_part.split(",") if i.strip() in ids or i.strip()=="NONE"]
                    results.append({
                        "Index": idx,
                        "Timestamp": df.at[idx, "Timestamp"],
                        "Speaker":   df.at[idx, "Speaker"],
                        "Utterance": df.at[idx, "Utterance"],
                        "Traits": ",".join(picked),
                        "Justification": justification.strip()
                    })
                except Exception:
                    logger.warning("Couldn't parse line: %s", line)
        except Exception as e:
            logger.error("Batch %d-%d error: %s", start, start+batch_size-1, e)

    # save
    out_csv = os.path.splitext(cha_path)[0] + "_analyzed.csv"
    pd.DataFrame(results).to_csv(out_csv, index=False, encoding="utf-8")
    logger.info("Saved analysis to %s", out_csv)

    # coverage
    found = set()
    for r in results:
        for tid in r["Traits"].split(","):
            if tid != "NONE":
                found.add(tid)
    pct = len(found)/total*100 if total else 0
    logger.info("Detected %d/%d traits (%.2f%%)", len(found), total, pct)
    print(f"Detected {len(found)}/{total} traits — {pct:.2f}% covered.")

if __name__ == "__main__":
    main()
















































