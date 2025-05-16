import os
import json
import time
import re
import argparse
import base64
import requests
import torch
from typing import Optional, List, Dict
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from prompts import (
    get_gpqa_search_o1_instruction,
    get_math_search_o1_instruction,
    get_code_search_o1_instruction,
    get_singleqa_search_o1_instruction,
    get_multiqa_search_o1_instruction,
    get_task_instruction_openqa,
    get_task_instruction_math,
    get_task_instruction_multi_choice,
    get_task_instruction_code,
)

# Special tokens
BEGIN_SEARCH_QUERY  = "<|begin_search_query|>"
END_SEARCH_QUERY    = "<|end_search_query|>"
BEGIN_SEARCH_RESULT = "<|begin_search_result|>"
END_SEARCH_RESULT   = "<|end_search_result|>"

# A regex to strip out the entire Search‐Result block when writing answers
SEARCH_RESULT_RE = re.compile(
    rf"{re.escape(BEGIN_SEARCH_RESULT)}.*?{re.escape(END_SEARCH_RESULT)}",
    flags=re.DOTALL
)

def custom_search(query: str, search_url: str, k: int) -> str:
    """
    Hits your API at {search_url}/search?query={query}&k={k},
    decodes each base64 result, and returns a single concatenated string.
    """
    url = f"{search_url}/search?query={query}&k={k}"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise RuntimeError(f"Search API request failed for '{query}': {e}")
    try:
        data = resp.json()
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON in response for '{query}': {e}")

    results = data.get("results", [])
    if not results:
        raise RuntimeError(f"Search API returned no results for '{query}'")

    out = ""
    for idx, item in enumerate(results):
        try:
            decoded = base64.b64decode(item).decode("utf-8")
            doc = json.loads(decoded)
        except Exception as e:
            raise RuntimeError(f"Error decoding result #{idx+1} for '{query}': {e}")
        u   = doc.get("URL", "").strip()
        cid = doc.get("ClueWeb22-ID", "")
        txt = doc.get("Clean-Text", "")
        out += f"Doc {idx+1} (URL: {u}, ClueWebID: {cid})\n{txt}\n\n"
    return out

def extract_between(text: str, start: str, end: str) -> Optional[str]:
    pattern = re.escape(start) + r"(.*?)" + re.escape(end)
    matches = re.findall(pattern, text, flags=re.DOTALL)
    return matches[-1].strip() if matches else None

def parse_args():
    p = argparse.ArgumentParser(
        description="Run Search-o1 on built-in datasets or a custom JSONL of queries."
    )
    p.add_argument('--input_file',   type=str,
                   help="Path to JSONL of {id,query}. If omitted, uses --dataset_name/--split.")
    p.add_argument('--search_url',    type=str, required=True,
                   help="Your custom search API base URL (e.g. https://my.api)")
    p.add_argument('--dataset_name',  type=str,
                   choices=['gpqa','math500','aime','amc','livecode',
                            'nq','triviaqa','hotpotqa','2wiki','musique','bamboogle'],
                   help="Built-in dataset name (ignored if --input_file is set).")
    p.add_argument('--split',         type=str, default='test',
                   choices=['test','diamond','main','extended'])
    p.add_argument('--subset_num',    type=int, default=-1,
                   help="Limit number of examples (built-in only).")
    p.add_argument('--max_search_limit', type=int, default=10)
    p.add_argument('--max_turn',      type=int, default=15)
    p.add_argument('--top_k',         type=int, default=10)
    p.add_argument('--max_doc_len',   type=int, default=3000)
    p.add_argument('--use_jina',      type=bool, default=True)
    p.add_argument('--jina_api_key',  type=str, default='None')
    p.add_argument('--model_path',    type=str, required=True)
    return p.parse_args()

def main():
    args = parse_args()

    # 1) Load data
    if args.input_file:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
        print(f"Loaded {len(data)} custom queries from {args.input_file}")
    else:
        if not args.dataset_name:
            raise ValueError("Either --input_file or --dataset_name must be provided.")
        if args.dataset_name == 'livecode':
            data_path = f'./data/LiveCodeBench/{args.split}.json'
        elif args.dataset_name in ['math500','gpqa','aime','amc']:
            data_path = f'./data/{args.dataset_name.upper()}/{args.split}.json'
        else:
            data_path = f'./data/QA_Datasets/{args.dataset_name}.json'
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if args.subset_num != -1:
            data = data[:args.subset_num]
        print(f"Loaded {len(data)} items from built-in {args.dataset_name}/{args.split}")

    # 2) Initialize model & tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=torch.cuda.device_count(),
        gpu_memory_utilization=0.95,
        max_model_len=30000,
    )

    # 3) Prepare prompts
    prompts: List[List[Dict]] = []
    for item in data:
        question = item.get('query') or item.get('Question')
        if args.input_file:
            instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
            user_prompt = get_task_instruction_openqa(question)
        else:
            dn = args.dataset_name
            if dn in ['nq','triviaqa']:
                instruction = get_singleqa_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_openqa(question)
            elif dn in ['hotpotqa','musique','bamboogle','2wiki']:
                instruction = get_multiqa_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_openqa(question)
            elif dn in ['math500','aime','amc']:
                instruction = get_math_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_math(question)
            elif dn == 'gpqa':
                instruction = get_gpqa_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_multi_choice(question)
            elif dn == 'livecode':
                instruction = get_code_search_o1_instruction(args.max_search_limit)
                user_prompt = get_task_instruction_code(
                    question,
                    question_title=item.get('question_title', '')
                )
            else:
                raise ValueError(f"Unsupported dataset: {dn}")
        p = [{"role":"user", "content": instruction + user_prompt}]
        prompts.append(
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
        )

    # 4) Initialize sequence state
    active = []
    for itm, pr in zip(data, prompts):
        active.append({
            'item': itm,
            'prompt': pr,
            'output': '',
            'history': [],
            'search_count': 0,
            'executed_search_queries': set(),
            'finished': False
        })

    # 5) Prepare streaming output file
    out_path = "outputs/searcho1_clueweb22_results.jsonl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fout = open(out_path, "a", buffering=1)
    written_ids = set()

    # 6) Main inference loop
    turn = 0
    start_t = time.time()
    while True:
        to_gen = [s for s in active if not s['finished']]
        if not to_gen or turn >= args.max_turn:
            break
        turn += 1
        print(f"\n--- Turn {turn} ({len(to_gen)} active) ---")

        sampling = SamplingParams(
            max_tokens=args.max_doc_len,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            repetition_penalty=1.05,
            stop=[END_SEARCH_QUERY, tokenizer.eos_token],
            include_stop_str_in_output=True,
        )
        outputs = llm.generate([s['prompt'] for s in to_gen], sampling)

        for seq, out in zip(to_gen, outputs):
            text = out.outputs[0].text
            seq['history'].append(text)
            seq['prompt'] += text
            seq['output'] += text

            q_ = extract_between(text, BEGIN_SEARCH_QUERY, END_SEARCH_QUERY)
            if q_ and seq['output'].rstrip().endswith(END_SEARCH_QUERY):
                if seq['search_count'] < args.max_search_limit and q_ not in seq['executed_search_queries']:
                    # sanitize query
                    sanitized = re.sub(r'<\|.*?\|>', '', q_, flags=re.DOTALL)
                    lines = [ln.strip() for ln in sanitized.splitlines() if ln.strip()]
                    if not lines:
                        seq['finished'] = True
                        continue
                    clean_q = lines[0]

                    # call your API
                    try:
                        docs = custom_search(clean_q, args.search_url, args.top_k)
                    except Exception as e:
                        print(f"Search API error for “{clean_q}”: {e}")
                        docs = ""

                    chunk = f"\n\n{BEGIN_SEARCH_RESULT}{docs}{END_SEARCH_RESULT}\n\n"
                    seq['prompt'] += chunk
                    seq['output'] += chunk
                    seq['search_count'] += 1
                    seq['executed_search_queries'].add(q_)
                else:
                    seq['finished'] = True
            else:
                seq['finished'] = True

            # as soon as we finish a sequence, write it
            if seq['finished']:
                _id = seq['item']['id']
                if _id not in written_ids:
                    qtxt = seq['item'].get('query') or seq['item'].get('Question')
                    raw_ans = seq['output'].strip()
                    # strip out the entire Search-Result block:
                    clean_ans = re.sub(SEARCH_RESULT_RE, "", raw_ans).strip()
                    fout.write(json.dumps({f"{_id}.q": qtxt, f"{_id}.a": clean_ans}) + "\n")
                    written_ids.add(_id)

    fout.close()
    print(f"\nAll done in {time.time()-start_t:.1f}s — results streaming to {out_path}")

if __name__ == "__main__":
    main()
