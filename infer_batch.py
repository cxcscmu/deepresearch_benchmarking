import argparse
import json
import os
import requests
import torch
import transformers


def parse_args():
    parser = argparse.ArgumentParser(description="Batch inference with Search-R1 and SERPer")
    parser.add_argument(
        "--input_file", required=True,
        help="Path to JSONL of {id,query}"
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Path to JSONL with incremental query results"
    )
    parser.add_argument(
        "--search_url", default="http://127.0.0.1:8000",
        help="SERPer API endpoint URL"
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of top docs to fetch per query"
    )
    parser.add_argument(
        "--model_id", default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        help="HuggingFace model ID or local path"
    )
    return parser.parse_args()


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [tokenizer.encode(seq, add_special_tokens=False) for seq in target_sequences]
        self.target_lengths = [len(ids) for ids in self.target_ids]
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        if input_ids.shape[1] < min(self.target_lengths):
            return False
        for ids, length in zip(self.target_ids, self.target_lengths):
            if torch.equal(
                input_ids[0, -length:],
                torch.tensor(ids, device=input_ids.device)
            ):
                return True
        return False


def search(query: str, search_url: str, k: int) -> str:
    api_key = os.environ.get("SERPER_API_KEY")
    if not api_key:
        raise RuntimeError("SERPER_API_KEY not set in environment")
    headers = {"X-API-Key": api_key}
    params = {"q": query, "num": k}
    resp = requests.get(search_url, params=params, headers=headers)
    resp.raise_for_status()
    data = resp.json().get("organic", [])
    out = ""
    for i, item in enumerate(data):
        title = item.get("title", "").strip()
        snippet = item.get("snippet", "").strip()
        out += f"Doc {i+1} (Title: {title}) {snippet}\n"
    return out


def get_query(text: str) -> str:
    import re
    pattern = re.compile(r"<search>(.*?)</search>", re.DOTALL)
    matches = pattern.findall(text)
    return matches[-1] if matches else None


def answer_question(
    question: str,
    model,
    tokenizer,
    stopping_criteria,
    search_url: str,
    k: int
) -> (str, dict):
    search_count = 0
    token_count = 0

    question = question.strip()
    if not question.endswith('?'):
        question += '?'

    prompt = f"""Answer the given question. \
You must conduct reasoning inside <think> and </think> every time you get new information. \
After reasoning, if you find you lack some knowledge, you can call a search engine by <search>query</search> and it will return the top searched results between <information> and </information>. \
You can search as many times as you want. \
When no further external knowledge is needed, provide the answer inside <answer> and </answer> without detailed reasoning. \nQuestion: {question}\n"""
    curr_search_template = '\n\n{output_text}<information>{search_results}</information>\n\n'
    eos_ids = [151645, 151643]

    if tokenizer.chat_template:
        prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False
        )

    while True:
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        token_count += input_ids.numel()

        outputs = model.generate(
            input_ids,
            max_new_tokens=1024,
            stopping_criteria=stopping_criteria,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7
        )

        generated = outputs[0][input_ids.shape[1]:]
        token_count += generated.numel()
        text = tokenizer.decode(generated, skip_special_tokens=True)

        if outputs[0, -1].item() in eos_ids:
            return text.strip(), {"tokens": token_count, "search_calls": search_count}

        full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        query = get_query(full_text)
        if query:
            search_results = search(query, search_url, k)
            search_count += 1
        else:
            search_results = ''

        prompt += curr_search_template.format(
            output_text=text,
            search_results=search_results
        )


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    stop_sequences = ["</search>", " </search>", "</search>\n", " </search>\n"]
    stopping_criteria = transformers.StoppingCriteriaList([
        StopOnSequence(stop_sequences, tokenizer)
    ])

    with open(args.input_file) as f:
        queries = [json.loads(line) for line in f]

    open(args.output_file, 'w').close()

    for item in queries:
        qid = item.get("id")
        qry = item.get("query")
        print(f"Processing {qid}: {qry}")
        ans, usage = answer_question(
            qry,
            model,
            tokenizer,
            stopping_criteria,
            args.search_url,
            args.k
        )
        record = {
            f"{qid}.q": qry,
            f"{qid}.a": ans,
            f"{qid}.u": usage
        }
        with open(args.output_file, 'a') as out:
            out.write(json.dumps(record) + "\n")

if __name__ == "__main__":
    main()
