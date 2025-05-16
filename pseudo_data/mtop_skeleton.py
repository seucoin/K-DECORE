import json
import re

# ============ file path ============
DATA_FILE = "KBQA_CL_2stages/task2/mtop_train2.json"
OUT_FILE = "task2_skeletons_with_elements.json"
# ==============================

# match IN: 和 SL:
IN_TAG_RE = re.compile(r"\[IN:([A-Z_]+)")
SL_TAG_RE = re.compile(r"\[SL:([A-Z_]+)")

def extract_schema_text(instruction: str) -> str:
    # extract API specification
    m = re.search(r"API specification:\s*\n+(.*?)\n+\n", instruction, re.S)
    return m.group(1).strip() if m else ""

def skeletonize_with_elements_mtop(output: str):
    tag_map = {}
    skeleton = output
    t_id, c_id = 1, 1

    # intent: IN:xxx → [T1]
    def repl_intent(m):
        nonlocal t_id
        raw = f"IN:{m.group(1)}"
        tag = f"T{t_id}"
        tag_map[tag] = raw
        t_id += 1
        return f"[{tag}"

    skeleton = IN_TAG_RE.sub(repl_intent, skeleton)

    # slot: SL:xxx → [C1], [C2] ...
    def repl_slot(m):
        nonlocal c_id
        raw = f"SL:{m.group(1)}"
        tag = f"C{c_id}"
        tag_map[tag] = raw
        c_id += 1
        return f"[{tag}"

    skeleton = SL_TAG_RE.sub(repl_slot, skeleton)

    return skeleton.strip(), tag_map

def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for ex in data:
        schema = extract_schema_text(ex["instruction"])
        skeleton, elements = skeletonize_with_elements_mtop(ex["output"])

        results.append({
            "skeleton": skeleton,
            "schema": schema,
            "elements": elements
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(results)} skeletons with elements → {OUT_FILE}")

if __name__ == "__main__":
    main()
