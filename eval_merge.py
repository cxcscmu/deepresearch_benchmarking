import argparse
from pathlib import Path
import json

def evaluate_folder(subfolder_name, model, path_to_reports):
    folder_path = Path(path_to_reports) / subfolder_name
    answer_quality_path = folder_path / f"evaluation_results_detailed_{model}.json"
    citation_quality_path = folder_path / f"evaluation_results_citation_{model}.json"
    kpr_quality_path = folder_path / f"evaluation_results_kpr_{model}.json"

    per_query_global_scores = {}

    # calculate quality scores
    all_answer_quality_results = {}
    if answer_quality_path.exists():
        with open(answer_quality_path, "r", encoding="utf-8") as f:
            all_answer_quality_results = json.load(f)
    else:
        print(f"Warning: Missing answer quality file: {answer_quality_path}") 

    # calculate citation scores
    all_citation_quality_results = {}
    if citation_quality_path.exists():
        with open(citation_quality_path, "r", encoding="utf-8") as f:
            all_citation_quality_results = json.load(f)
    else:
        print(f"Warning: Missing citation quality file: {citation_quality_path}")

    # calculate KPR scores
    all_kpr_quality_results = {}
    if kpr_quality_path.exists():
        with open(kpr_quality_path, "r", encoding="utf-8") as f:
            all_kpr_quality_results = json.load(f)
    else:
        print(f"Warning: Missing KPR quality file: {kpr_quality_path}")

    
    for query in all_answer_quality_results:
        if query not in all_citation_quality_results:
            # should not happen, but catching just in case
            continue
        citation_score = all_citation_quality_results[query]["score"]
        number_of_criteria = len(all_answer_quality_results[query]["scores"])
        score_without_citation = sum([x[0] for x in all_answer_quality_results[query]["scores"].values()])
        normalized_score_without_citations = score_without_citation * 100 / (number_of_criteria * 10)


        all_answer_quality_results[query]["scores"]["Citation Quality"] = [citation_score * 10, None]
        number_of_criteria = len(all_answer_quality_results[query]["scores"])
        score_with_citations = sum([x[0] for x in all_answer_quality_results[query]["scores"].values()])
        normalized_score_with_citations = score_with_citations * 100 / (number_of_criteria * 10)

        per_query_global_scores[query] = {"without_citations": normalized_score_without_citations, "with_citations": normalized_score_with_citations}
   
    support_rates = []
    omitted_rates = []
    contradicted_rates = []

    for qid, kpr_result in all_kpr_quality_results.items():
        support_rate = kpr_result["support_rate"]
        omitted_rate = kpr_result["ommitted_rate"]
        contradicted_rate = kpr_result["contradicted_rate"]
        support_rates.append(support_rate)
        omitted_rates.append(omitted_rate)
        contradicted_rates.append(contradicted_rate)

    avg_support_rate = sum(support_rates) / len(support_rates) if support_rates else None
    avg_omitted_rate = sum(omitted_rates) / len(omitted_rates) if omitted_rates else None
    avg_contradicted_rate = sum(contradicted_rates) / len(contradicted_rates) if contradicted_rates else None

    
    avg_with_citations = sum([x["with_citations"] for x in per_query_global_scores.values()]) / len(per_query_global_scores)
    avg_without_citations = sum([x["without_citations"] for x in per_query_global_scores.values()]) / len(per_query_global_scores)
    return per_query_global_scores, avg_with_citations, avg_without_citations, avg_support_rate, avg_omitted_rate, avg_contradicted_rate

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--subfolder", help="Name of the subfolder inside 'answers/'")
    parser.add_argument("--open_ai_model", help="openai model name'")
    args = parser.parse_args()

    print(f"Mergig citation and quality results.")

    path_to_reports = "reports"
    path_to_results = "results"
    
    per_query, avg_with_citations, avg_without_citations, avg_support_rate, avg_omitted_rate, avg_contradicted_rate = evaluate_folder(args.subfolder, args.open_ai_model, path_to_results)

    result_dir = Path(path_to_results) / args.subfolder
    result_dir.mkdir(parents=True, exist_ok=True)

    output_file = result_dir / f"evaluation_results_per_query_score_{args.open_ai_model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(per_query, f, indent=2, ensure_ascii=False)

    output_file = result_dir / f"evaluation_results_single_score_{args.open_ai_model}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"score_without_citations": avg_without_citations, 
                   "score_with_citations": avg_with_citations, 
                   "score_support":avg_support_rate, 
                   "score_omitted": avg_omitted_rate,
                    "score_contradicted": avg_contradicted_rate
                    },
                   f, indent=2, ensure_ascii=False)
