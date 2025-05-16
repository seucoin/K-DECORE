import json
import re

# ============ file path ============
DATA_FILE = "KBQA_CL_2stages/task1/grailqa_train2.json"
OUT_FILE = "task1_skeletons_with_elements.json"
# ==================================

SCHEMA_RE = re.compile(r"schema:\s*\n+(.*?)\n+question:", re.S | re.I)
TOKEN_RE = re.compile(r"[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*")
ENTITY_RE = re.compile(r"\b(?:m|g|d)\.[\w\d_]+\b")

def extract_schema_text(instr: str) -> str:
    m = SCHEMA_RE.search(instr)
    return m.group(1).strip() if m else ""

def extract_tables(instr: str) -> set[str]:
    schema_text = extract_schema_text(instr)
    tables = set()
    for seg in schema_text.split('|'):
        seg = seg.strip()
        if not seg:
            continue
        if ':' in seg:
            left = seg.split(':', 1)[0].strip()
            if left:
                tables.add(left)
        elif '.' in seg:
            tables.add(seg)
    return tables

def skeletonize_with_elements(query: str, tables: set[str]):
    ent_map, t_map, c_map = {}, {}, {}
    next_e, next_t, next_c = 1, 1, 1

    def ent_repl(m):
        nonlocal next_e
        eid = m.group(0)
        tag = f"E{next_e}"
        ent_map[tag] = eid
        next_e += 1
        return f"[{tag}]"

    query = ENTITY_RE.sub(ent_repl, query)

    def tok_repl(tok: str) -> str:
        nonlocal next_t, next_c
        if tok.isupper():
            return tok
        if tok in tables:
            tag = f"T{next_t}"
            t_map[tag] = tok
            next_t += 1
            return f"[{tag}]"
        parts = tok.split('.')
        if len(parts) > 1 and '.'.join(parts[:2]) in tables:
            tag = f"C{next_c}"
            c_map[tag] = tok
            next_c += 1
            return f"[{tag}]"
        tag = f"C{next_c}"
        c_map[tag] = tok
        next_c += 1
        return f"[{tag}]"

    query = TOKEN_RE.sub(lambda m: tok_repl(m.group(0)), query)
    skeleton = query.strip()

    elements = {}
    elements.update({k: v for k, v in t_map.items()})
    elements.update({k: v for k, v in c_map.items()})
    elements.update({k: v for k, v in ent_map.items()})

    return skeleton, elements

def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for ex in data:
        schema_text = extract_schema_text(ex["instruction"])
        tables = extract_tables(ex["instruction"])
        skeleton, elements = skeletonize_with_elements(ex["output"], tables)

        results.append({
            "skeleton": skeleton,
            "schema": schema_text,
            "elements": elements
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(results)} skeletons with elements → {OUT_FILE}")

if __name__ == "__main__":
    main()
