import json
from json_repair import repair_json

INPUT_FILE = "data/nemotron_sft_all_final_98k.json"
OUTPUT_FILE = "data/nemotron_sft_cleaned.json"


def clean_dataset():
    valid_count = 0
    repaired_count = 0
    error_count = 0

    print(f"Starting cleanup of {INPUT_FILE}...")

    with open(INPUT_FILE, 'r', encoding='utf-8') as f_in, \
            open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:

        for i, line in enumerate(f_in):
            line = line.strip()
            if not line: continue

            try:
                # Try standard loading first
                json.loads(line)
                f_out.write(line + "\n")
                valid_count += 1
            except json.JSONDecodeError:
                # If it fails, attempt to repair the string
                repaired = repair_json(line)
                if repaired:
                    f_out.write(repaired + "\n")
                    repaired_count += 1
                else:
                    print(f"Line {i + 1} is too corrupted to fix. Skipping.")
                    error_count += 1

            if (i + 1) % 10000 == 0:
                print(f"Processed {i + 1} lines...")

    print("\n--- Cleanup Complete ---")
    print(f"Perfectly valid lines: {valid_count}")
    print(f"Repaired lines:        {repaired_count}")
    print(f"Discarded lines:       {error_count}")
    print(f"Cleaned file saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    clean_dataset()