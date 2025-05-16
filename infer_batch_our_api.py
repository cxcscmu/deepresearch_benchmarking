import argparse
import json
import base64
import requests
import torch
import transformers


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch inference with Search-R1 and ClueWeb22 API"
    )
    parser.add_argument(
        "--input_file", required=True,
        help="Path to JSONL of {id,query}"
    )
    parser.add_argument(
        "--output_file", required=True,
        help="Path to JSONL with incremental query results"
    )
    parser.add_argument(
        "--search_url", default="",
        help="Base URL for ClueWeb22 API (will append /search?query=…&k=…)"
    )
    parser.add_argument(
        "--k", type=int, default=3,
        help="Number of top docs to fetch per query"
    )
    parser.add_argument(
        "--model_id",
        default="PeterJinGo/SearchR1-nq_hotpotqa_train-qwen2.5-7b-em-ppo",
        help="HuggingFace model ID or local path"
    )
    return parser.parse_args()


class StopOnSequence(transformers.StoppingCriteria):
    def __init__(self, target_sequences, tokenizer):
        self.target_ids = [
            tokenizer.encode(seq, add_special_tokens=False)
            for seq in target_sequences
        ]
        self.target_lengths = [len(ids) for ids in self.target_ids]

    def __call__(self, input_ids, scores, **kwargs):
        seq_len = input_ids.shape[1]
        if seq_len < min(self.target_lengths):
            return False
        for ids, length in zip(self.target_ids, self.target_lengths):
            if torch.equal(
                input_ids[0, -length:],
                torch.tensor(ids, device=input_ids.device)
            ):
                return True
        return False


def search(query: str, search_url: str, k: int) -> str:
    url = f"{search_url}/search?query={query}&k={k}"
    try:
        response = requests.get(url, timeout=10)
    except requests.RequestException as e:
        raise RuntimeError(f"Search API request failed for '{query}': {e}")
    try:
        data = response.json()
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON in response for '{query}': {e}")

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"Search API returned no results for '{query}'")

    out = ""
    for idx, item in enumerate(results):
        try:
            dec = base64.b64decode(item).decode("utf-8")
            doc = json.loads(dec)
        except Exception as e:
            raise RuntimeError(f"Error decoding result #{idx+1} for '{query}': {e}")
        u = doc.get("URL", "").strip()
        cid = doc.get("ClueWeb22-ID", "")
        txt = doc.get("Clean-Text", "")
        out += f"Doc {idx+1} (URL: {u}, ClueWebID: {cid}) {txt}\n"
    return out


def get_query(text: str) -> str:
    import re
    m = re.findall(r"<search>(.*?)</search>", text, re.DOTALL)
    return m[-1].strip() if m else None


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
    model_max   = tokenizer.model_max_length       
    gen_tokens  = 1024                              
    context_max = model_max - gen_tokens

    while True:
        # input_ids = tokenizer.encode(prompt, return_tensors='pt').to(model.device)
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(ids) > context_max:
            ids = ids[-context_max:]
        # 3) build the tensor
        input_ids = torch.tensor([ids], device=model.device)
        # 4) (optional) build attention mask
        attention_mask = torch.ones_like(input_ids)
        token_count += input_ids.numel()

        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
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
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_id)
    model     = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    stop_seqs = ["</search>", " </search>", "</search>\n", " </search>\n"]
    stopping  = transformers.StoppingCriteriaList([
        StopOnSequence(stop_seqs, tokenizer)
    ])

    with open(args.input_file) as f:
        queries = [json.loads(line) for line in f]
    open(args.output_file, 'w').close()

    for item in queries:
        qid, qry = item["id"], item["query"]
        print(f"Processing {qid}: {qry}")
        ans, usage = answer_question(
            qry, model, tokenizer,
            stopping, args.search_url, args.k
        )
        rec = {f"{qid}.q": qry, f"{qid}.a": ans, f"{qid}.u": usage}
        with open(args.output_file, 'a') as out:
            out.write(json.dumps(rec) + "\n")

if __name__ == "__main__":
    main()
