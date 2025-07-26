import pandas as pd
import re
import os

def parse_cha_file(file_path):
    """
    Parses a .cha CHAT transcript file and returns a DataFrame
    with columns: Timestamp, Speaker, Utterance.
    """
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    conversations = []
    current_timestamp = None
    timestamp_pattern = re.compile(r"\x15(\d+:\d+:\d+_\d+:\d+:\d+)")

    for line in lines:
        line = line.strip()

        if line.startswith("*"):
            parts = line.split(":", 1)
            if len(parts) < 2:
                continue  # skip malformed lines
            speaker = parts[0][1:].strip()
            utterance = parts[1].strip()

            conversations.append({
                "Timestamp": current_timestamp,
                "Speaker": speaker,
                "Utterance": utterance
            })

        elif line.startswith("\x15"):
            match = timestamp_pattern.match(line)
            if match:
                current_timestamp = match.group(1)

    return pd.DataFrame(conversations)


if __name__ == "__main__":
    file_path = input("Enter the path to your .cha file: ")

    df = parse_cha_file(file_path)

    output_dir = r"C:\\Users\\swapn\\OneDrive\\Desktop\\AI-Agent\\ParserOutput"
    os.makedirs(output_dir, exist_ok=True)

    base_filename = os.path.basename(file_path).replace(".cha", "_parsed.csv")
    output_csv = os.path.join(output_dir, base_filename)

    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Parsing completed! File saved to: {output_csv}")
