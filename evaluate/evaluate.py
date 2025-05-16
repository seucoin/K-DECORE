import argparse
import json
import os
import importlib
from SPARQLWrapper import SPARQLWrapper, JSON, SPARQLExceptions
from tqdm import tqdm
from utils.configure import Configure


def generic_evaluate(preds, test_data):
    cfgargs = Configure.Get("")
    evaluator = importlib.import_module("metrics.meta_tuning.evaluator").EvaluateTool(cfgargs)
    return evaluator.evaluate(preds, test_data, "test")


def parse_sparql_results(response):
    if "boolean" in response:
        return [response["boolean"]]
    bindings = response.get("results", {}).get("bindings", [])

    if len(bindings) > 0 and "callret-0" in bindings[0]:
        return [int(bindings[0]["callret-0"]["value"])]

    results = []
    for row in bindings:
        for v in row.values():
            results.append(v["value"].replace('http://rdf.freebase.com/ns/', ""))
    return results


def execute_sparql(query: str, kb_endpoint: str):
    clean_q = query.replace("FILTER(NOT EXISTS", "FILTER NOT EXISTS")
    clean_q = "\n".join(line.strip() for line in clean_q.splitlines() if line.strip())

    sparql = SPARQLWrapper(kb_endpoint)
    sparql.setReturnFormat(JSON)
    sparql.setQuery(clean_q)
    try:
        resp = sparql.query().convert()
        return parse_sparql_results(resp)
    except SPARQLExceptions.QueryBadFormed as e:
        print(f"[SPARQL Error] - Skip")
        return None
    except Exception as e:
        print(f"[SPARQL Error]")
        return None


def sparql_evaluate(preds, test_data, kb_endpoint):
    skipped = total = 0
    sum_p = sum_r = sum_f = 0.0

    for gold_item, pred_query in zip(test_data, preds):
        gold_q = gold_item.get("output")
        if not gold_q:
            skipped += 1
            continue

        gold_res = execute_sparql(gold_q, kb_endpoint)
        if not gold_res:
            skipped += 1
            continue

        pred_res = execute_sparql(pred_query, kb_endpoint)

        gold_set = set(gold_res)
        pred_set = set(pred_res) if pred_res else set()

        if not gold_set and not pred_set:
            p = r = f = 1.0
        else:
            tp = len(gold_set & pred_set)
            p = tp / len(pred_set) if pred_set else 0.0
            r = tp / len(gold_set) if gold_set else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

        sum_p += p; sum_r += r; sum_f += f
        total += 1

    avg_p = sum_p / total if total else 0.0
    avg_r = sum_r / total if total else 0.0
    avg_f = sum_f / total if total else 0.0

    return {
        "avg_precision": f"{avg_p:.4f}",
        "avg_recall":    f"{avg_r:.4f}",
        "avg_f1":        f"{avg_f:.4f}",
        "total_valid":   total,
        "skipped":       skipped
    }


def main():
    parser = argparse.ArgumentParser(description="evaluating all tasks")
    parser.add_argument("--pred_file",  required=True, help="pred JSON path")
    parser.add_argument("--test_file",  required=True, help="gold JSON path")
    parser.add_argument("--save_file",  required=True, help="save path")
    parser.add_argument("--kb_endpoint", default="http://10.201.38.151:3001/sparql",
                        help="for SPARQL endpoint")
    args = parser.parse_args()

    # 加载预测与测试数据
    preds = json.load(open(args.pred_file, 'r'))
    test_data = json.load(open(args.test_file, 'r'))
    assert len(preds) == len(test_data), "number of predictions and test data do not match"

    test_name = os.path.basename(args.test_file).lower()
    if 'compweb' in test_name:
        metrics = sparql_evaluate(preds, test_data, args.kb_endpoint)
    else:
        metrics = generic_evaluate(preds, test_data)

    os.makedirs(os.path.dirname(args.save_file), exist_ok=True)
    json.dump(metrics, open(args.save_file, 'w'), indent=4, ensure_ascii=False)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
