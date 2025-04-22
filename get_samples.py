import os
import random
import json
from pathlib import Path

def process_folder(folder_names):

    folders = [Path(f"/data/group_data/cx_group/deepsearch_benchmark/reports/{f}") for f in folder_names]

    ids = [
        file.stem for file in folders[0].glob("*.q")
        if (folders[0] / f"{file.stem}.a").exists()
    ]
    if not ids:
        raise ValueError("No valid <id>.q and <id>.a pairs found.")

    # Pick a random ID
    selected_id = random.choice(ids)

    for folder in folders:
        q_path = folder / f"{selected_id}.q"
        a_path = folder / f"{selected_id}.a"

        with open(q_path, "r", encoding="utf-8") as f:
            question = f.read()

        with open(a_path, "r", encoding="utf-8") as f:
            answer = f.read()

        # Save to new txt files
        model_name = str(folder).split("/")[-1]
        os.makedirs(f".samples/{model_name}", exist_ok=True)
        with open(f".samples/{model_name}/{selected_id}_query.txt", "w", encoding="utf-8") as f:
            f.write(question)

        with open(f".samples/{model_name}/{selected_id}_report.txt", "w", encoding="utf-8") as f:
            f.write(answer)

        exit()
        # Load evaluations
        eval_path_quality = folder / "evaluation_results_detailed_gpt-4o-mini.json"
        eval_path_citation = folder / "evaluation_results_citation_gpt-4o-mini.json"
        assert eval_path_quality.exists(), "evaluation_results_detailed.json not found."
        assert eval_path_citation.exists(), "evaluation_results_citation.json not found."

        with open(eval_path_quality, "r", encoding="utf-8") as f:
            eval_data_quality = json.load(f)

        with open(eval_path_citation, "r", encoding="utf-8") as f:
            eval_data_citation = json.load(f)

        if selected_id not in eval_data_quality:
            raise KeyError(f"ID {selected_id} not found in evaluation_results_detailed.json")

        if selected_id not in eval_data_citation:
            raise KeyError(f"ID {selected_id} not found in evaluation_results_citation.json")

        eval_dict_quality = eval_data_quality[selected_id]

        eval_dict_citation = eval_data_citation[selected_id]

        # Save eval data to txt
        eval_txt_path = f".samples/{model_name}/{selected_id}_eval.txt"
        with open(eval_txt_path, "w", encoding="utf-8") as f:
            f.write(f"--------------\n")
            f.write(f"Answer Quality\n")
            f.write(f"--------------\n")
            for key, val in eval_dict_quality.items():
                if isinstance(val, dict):
                    f.write(f"{key}:\n")
                    for subkey, subval in val.items():
                        if isinstance(subval, list):
                            items = ", ".join(map(str, subval))
                            f.write(f"  {subkey}: {items}\n")
                        else:
                            f.write(f"  {subkey}: {subval}\n")
                else:
                    f.write(f"{key}: {val}\n")

            f.write(f"--------------\n")
            f.write(f"Citation Quality\n")
            f.write(f"--------------\n")

            f.write(f"Score: {eval_dict_citation.get('score', 'N/A')}\n")

            detailed = eval_dict_citation.get("detailed")
            if detailed:
                for claim_id, claim_info in detailed.items():
                    f.write(f"\n{claim_id}:\n")
                    f.write(f"  Claim: {claim_info.get('claim', '')}\n")
                    urls = claim_info.get("urls", [])
                    if urls:
                        f.write(f"  URLs:\n")
                        for url in urls:
                            f.write(f"    - {url}\n")
                    f.write(f"  Score: {claim_info.get('score', 'N/A')}\n")
                    f.write(f"  Justification: {claim_info.get('justification', '')}\n")
                

# Example usage
if __name__ == "__main__":
    #folder_names = ["gpt-4o-search-preview", "hf_deepsearch_o3mini", "sonar-deep-research"]
    folder_names = ["gpt-4o-search-preview"]
    process_folder(folder_names)