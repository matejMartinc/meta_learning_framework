import json
from collections import defaultdict


def parse_best_model(best_model_data, model_mapping):
    """
    Handles 'best_model' whether it is a string or a list,
    and maps placeholders to real model IDs.
    """
    if not best_model_data:
        return []

    # If the judge returned a list, convert it to a string for keyword checking
    # or check the elements directly. To be safe, we'll normalize to a string.
    if isinstance(best_model_data, list):
        best_model_string = " ".join(map(str, best_model_data))
    else:
        best_model_string = str(best_model_data)

    lower_string = best_model_string.lower()
    tie_all_keywords = ['none', 'all models', 'equally', 'tie', 'draw']

    if any(keyword in lower_string for keyword in tie_all_keywords):
        return list(model_mapping.values())

    winners = []
    for placeholder, real_name in model_mapping.items():
        # Check if "Model 1" exists in the string/list output
        if placeholder in best_model_string:
            winners.append(real_name)

    return winners


def analyze_evaluations(file_path):
    model_scores = defaultdict(lambda: defaultdict(float))
    model_counts = defaultdict(lambda: defaultdict(int))
    win_counts = defaultdict(int)
    total_evaluations = 0

    all_real_models = set()
    criteria = ["grammar", "semantics", "flow", "completeness", "factuality"]

    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                data = json.loads(line)
                total_evaluations += 1

                mapping = data.get("model_mapping", {})
                all_real_models.update(mapping.values())

                best_model_data = data.get("best_model", "")
                winning_models = parse_best_model(best_model_data, mapping)
                for model in winning_models:
                    win_counts[model] += 1

                for evaluation in data.get("evaluations", []):
                    placeholder_name = evaluation.get("model")
                    real_model_name = mapping.get(placeholder_name)

                    if real_model_name:
                        scores_dict = evaluation.get("scores", {})
                        for criterion in criteria:
                            if criterion in scores_dict:
                                model_scores[real_model_name][criterion] += scores_dict[criterion]
                                model_counts[real_model_name][criterion] += 1

            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from a line in {file_path}")

    if total_evaluations == 0:
        return "The file is empty or invalid."

    target_models = sorted(list(all_real_models))

    # --- Calculations ---
    average_scores = defaultdict(dict)
    for model in target_models:
        for criterion in criteria:
            count = model_counts[model].get(criterion, 0)
            average_scores[model][criterion] = model_scores[model][criterion] / count if count > 0 else 0.0

    win_percentages = {model: (win_counts.get(model, 0) / total_evaluations) * 100 for model in target_models}

    # --- Generate LaTeX Table ---
    latex_safe_names = [m.replace('_', '\\_') for m in target_models]

    header = " & " + " & ".join([f"\\textbf{{{m}}}" for m in latex_safe_names]) + " \\\\"
    col_format = "|l|" + "c" * len(target_models) + "|"

    rows = []
    for criterion in criteria:
        row_name = criterion.replace('_', ' ').title()
        row_data = [f"{average_scores[model].get(criterion, 0):.2f}" for model in target_models]
        rows.append(f"\\textbf{{{row_name}}} & " + " & ".join(row_data) + " \\\\")

    win_count_data = [f"{win_counts.get(model, 0)}" for model in target_models]
    rows.append("\\hline\n\\textbf{Win Count} & " + " & ".join(win_count_data) + " \\\\")

    win_perc_data = [f"{win_percentages.get(model, 0):.2f}\\%" for model in target_models]
    rows.append("\\textbf{Win Rate} & " + " & ".join(win_perc_data) + " \\\\")

    rows_str = '\n\\hline\n'.join(rows)

    return f"""
        \\begin{{table}}[h!]
        \\centering
        \\small
        \\begin{{tabular}}{{{col_format}}}
        \\hline
        {header}
        \\hline
        {rows_str}
        \\hline
        \\end{{tabular}}
        \\caption{{Evaluation Results (n={total_evaluations})}}
        \\end{{table}}
        """.strip()


if __name__ == "__main__":
    file_name = "LLM_as_a_judge_scores.jsonl"
    print(analyze_evaluations(file_name))