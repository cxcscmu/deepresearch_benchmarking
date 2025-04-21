import argparse
from pathlib import Path
import json

def evaluate_folder(subfolder_name, model):
    folder_path = Path("answers") / subfolder_name
    answer_quality_path = folder_path / f"evaluation_results_detailed_{model}.json"
    citation_quality_path = folder_path / f"evaluation_results_citation_{model}.json"

    per_query_global_scores = {}

    all_answer_quality_results = {}
    if answer_quality_path.exists():
        with open(answer_quality_path, "r", encoding="utf-8") as f:
            all_answer_quality_results = json.load(f)

    all_citation_quality_results = {}
    if citation_quality_path.exists():
        with open(citation_quality_path, "r", encoding="utf-8") as f:
            all_citation_quality_results = json.load(f)

    for query in all_answer_quality_results:
        citation_score = all_citation_quality_results[query]["score"]
        number_of_criteria = len(all_answer_quality_results[query]["scores"])
        score_without_citation = sum([x[0] for x in all_answer_quality_results[query]["scores"].values()])
        normalized_score_without_citations = score_without_citation * 100 / (number_of_criteria * 10)


        all_answer_quality_results[query]["scores"]["Citation Quality"] = [citation_score * 10, None]
        number_of_criteria = len(all_answer_quality_results[query]["scores"])
        score_with_citations = sum([x[0] for x in all_answer_quality_results[query]["scores"].values()])
        normalized_score_with_citations = score_with_citations * 100 / (number_of_criteria * 10)

        per_query_global_scores[query] = {"without_citations": normalized_score_without_citations, "with_citations": normalized_score_with_citations}

    
    avg_with_citations = sum([x["with_citations"] for x in per_query_global_scores.values()]) / len(per_query_global_scores)
    avg_without_citations = sum([x["without_citations"] for x in per_query_global_scores.values()]) / len(per_query_global_scores)
    return per_query_global_scores, avg_with_citations, avg_without_citations

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", help="Name of the subfolder inside 'answers/'")
    parser.add_argument("--open_ai_model", help="openai model name'")
    args = parser.parse_args()

    print(f"Mergig citation and quality results.")

    per_query, avg_with_citations, avg_without_citations = evaluate_folder(args.subfolder, args.open_ai_model)

    output_file = Path("answers") / args.subfolder / f"evaluation_results_per_query_score_{args.open_ai_model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)

    output_file = Path("answers") / args.subfolder / f"evaluation_results_single_score_{args.open_ai_model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"score_without_citations": avg_without_citations, "score_with_citations": avg_with_citations}, f, indent=2, ensure_ascii=False)
