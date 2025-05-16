import json, re, random, pathlib, tqdm, gc
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ---------- file path ----------
BASE_MODEL = "you_<Qwen2.5-7B-Instruct>_model"
BASE = pathlib.Path("KBQA_CL_2stages")

SKELETON_JSON = {
    1: "task1_skeletons_with_elements.json",
    2: "task2_skeletons_with_elements.json",
    3: "task3_skeletons_with_elements.json",
    4: "task4_skeletons_with_elements.json",
}
OUT_JSON = {n: pathlib.Path(f"task{n}_pseudo_train.json") for n in range(1, 5)}
LORA_DIR = {
    n: f"<your_lora_path>" for n in range(1, 5)
}
PROMPT = ("Given a formal query and its relevant schema context, write a clear, fluent "
          "natural-language question that would be answered by executing the query.\n\n"
          "query:\n\n{query}\n\nschema:\n\n{schema}\n\n"
          "Only output the question, nothing else.")
INST = {
    1: "Given a list of knowledge graph schema items, write the question as an s-expression that can be used to find the answer. schema:\n\n{schema}\n\n\nquestion:\n\n{question}",
    2: "Convert the following natural language query to an API call in Task Oriented Parsing (TOP) representation using the following API specification. API specification:\n\n{schema}\n\n\nnatural language query:\n\n{question}",
    3: "Your task is to convert the following question to an SQL query using the following database schema. schema:\n\n{schema}\n\n\nquestion:\n\n{question}",
    4: "Given the list of schema items, write an SPARQL query that can be used to find the answer to the following question. schema:\n\n{schema}\n\n\nquestion:\n\n{question}",
}
NAME = {1: "grailqa", 2: "mtop", 3: "spider_with_cell", 4: "compwebq"}


def fill_skeleton(skeleton, elements):
    result = skeleton
    for k, v in elements.items():
        result = result.replace(f"[{k}]", v)
    return result

def format_task4_query(query):
    prefix = "PREFIX ns: <http://rdf.freebase.com/ns/>\nSELECT DISTINCT ?x\nWHERE "
    if query.startswith("PREFIX") or query.startswith("SELECT"):
        return query
    query = query.strip()
    if not query.startswith("{"):
        query = "{\n" + query
    if not query.endswith("}"):
        query = query + "\n}"
    return prefix + query

def gen_q(model, tokenizer, schema, query):
    prompt = PROMPT.format(schema=schema, query=query)
    chat_format = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(chat_format, return_tensors="pt").to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=128,
        temperature=0.1,
        top_p=0.95,
        top_k=50,
        repetition_penalty=1.1,
        do_sample=True,
        num_beams=3,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    text = response.strip()
    if "." in text:
        first_sent = text.split(".", 1)[0].strip() + "."
        if len(first_sent) > 10:
            text = first_sent
    text = re.sub(r'^["\']|["\']$', '', text)
    return text or "What is the answer?"


def build_tpl_pairs(file_path, max_pairs=20):
    with open(file_path, encoding="utf-8") as f:
        data = json.load(f)

    buckets = defaultdict(list)
    for ex in data:
        sig = tuple(sorted(ex["elements"].keys()))
        buckets[sig].append(ex)

    pairs = []
    for group in buckets.values():
        if len(group) < 2:
            continue
        tried = set()
        while len(pairs) < max_pairs and len(tried) < len(group)**2:
            a, b = random.sample(group, 2)
            if a["skeleton"] == b["skeleton"]:
                continue
            k = (a["skeleton"], b["skeleton"])
            if k in tried:
                continue
            tried.add(k)
            pairs.append((a["skeleton"], b["elements"], b["schema"]))
            pairs.append((b["skeleton"], a["elements"], a["schema"]))
    return pairs[:max_pairs]

def main():
    random.seed(3407)
    for tid in range(1, 5):
        print(f"\nTask {tid}: Generating pseudo samples...")
        tpl_pairs = build_tpl_pairs(SKELETON_JSON[tid])

        print("Loading Qwen2.5 + LoRA...")
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base_model, LORA_DIR[tid], device_map="auto")

        samples = []
        for skeleton, elements, schema in tqdm.tqdm(tpl_pairs, desc=f"task{tid}"):
            query = fill_skeleton(skeleton, elements)
            if tid == 4:
                query = format_task4_query(query)
            question = gen_q(model, tokenizer, schema, query)
            instruction = INST[tid].format(schema=schema, question=question)

            samples.append({
                "instruction": instruction,
                "input": "",
                "output": query,
                "system": "You are an AI assistant that specializes in analyzing and reasoning over structured information. You will be given a task, optionally with some structured knowledge input. Your answer must strictly adhere to the output format, if specified.",
                "is_truncated": False,
                "task_name": NAME[tid]
            })

        OUT_JSON[tid].write_text(json.dumps(samples, ensure_ascii=False, indent=4))
        print(f"✅ task{tid}: wrote {len(samples)} pseudo samples → {OUT_JSON[tid]}")

        del model, base_model
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except ImportError:
            pass

if __name__ == "__main__":
    main()
