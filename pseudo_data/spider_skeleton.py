import json
import re

# ---------- file path----------
DATA_FILE = "KBQA_CL_2stages/task4/compwebq_train2.json"
OUT_FILE = "task4_skeletons_with_elements.json"
# --------------------------

SCHEMA_RE = re.compile(r"schema:\s*\n\n(.*?)\n\n\nquestion:", re.S | re.I)

def parse_schema(instr: str):
    m = SCHEMA_RE.search(instr)
    if not m:
        return set(), set(), ""
    schema_text = m.group(1).strip()
    tables, columns = set(), set()
    for seg in schema_text.split('|'):
        seg = seg.strip()
        if not seg:
            continue
        if ':' in seg:
            table, col_part = [s.strip() for s in seg.split(':', 1)]
            tables.add(table)
            for col in re.split(r'[,\s]+', col_part):
                col = col.strip()
                if col:
                    columns.add(f"{table}.{col}")
        else:
            tables.add(seg)
    return tables, columns, schema_text


PRED_RE   = re.compile(r"ns:([A-Za-z0-9_\.]+)")
ENT_RE    = re.compile(r"ns:(?:m|g|d)\.[A-Za-z0-9_]+")  # 实体 id

def skeletonize_with_elements(sparql: str, tables, columns):
    elements = {}
    e_map, t_map, c_map = {}, {}, {}
    idx_e = idx_t = idx_c = 1

    def ent_repl(m):
        nonlocal idx_e
        eid = m.group(0)[3:]  # remove 'ns:'
        tag = f"E{idx_e}"
        e_map[m.group(0)] = tag
        elements[tag] = eid
        idx_e += 1
        return f"[{tag}]"

    sparql = ENT_RE.sub(ent_repl, sparql)

    def pred_repl(m):
        nonlocal idx_t, idx_c
        core = m.group(1)
        if core in columns:
            if core not in c_map:
                tag = f"C{idx_c}"
                c_map[core] = tag
                elements[tag] = core
                idx_c += 1
            return f"[{c_map[core]}]"
        elif core in tables:
            if core not in t_map:
                tag = f"T{idx_t}"
                t_map[core] = tag
                elements[tag] = core
                idx_t += 1
            return f"[{t_map[core]}]"
        elif '.' in core and '.'.join(core.split('.')[:2]) in tables:
            if core not in c_map:
                tag = f"C{idx_c}"
                c_map[core] = tag
                elements[tag] = core
                idx_c += 1
            return f"[{c_map[core]}]"
        return m.group(0)

    sparql = PRED_RE.sub(pred_repl, sparql)

    skeleton = ' '.join(sparql.split()) 
    return skeleton.strip(), elements

def main():
    with open(DATA_FILE, encoding="utf-8") as f:
        samples = json.load(f)

    results = []
    for ex in samples:
        tables, columns, schema_text = parse_schema(ex["instruction"])
        skeleton, elements = skeletonize_with_elements(ex["output"], tables, columns)
        results.append({
            "skeleton": skeleton,
            "schema": schema_text,
            "elements": elements
        })

    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"✅ Extracted {len(results)} compwebq skeletons with elements → {OUT_FILE}")

if __name__ == "__main__":
    main()
