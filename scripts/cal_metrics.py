import json
import os
import numpy as np
from pprint import pprint

# 任务顺序配置
ORDERS = {
    "order1": ["grailqa", "mtop", "spider_with_cell", "comwebq"],
    "order2": ["comwebq", "spider_with_cell", "mtop", "grailqa"],
    "order3": ["mtop", "grailqa", "comwebq", "spider_with_cell"]
}

METRIC_PATHS = {
    "grailqa": "META_TUNING/grailqa.cfg/exact_match",
    "mtop": "META_TUNING/mtop.cfg/exact_match",
    "spider_with_cell": "META_TUNING/spider_with_cell.cfg/exact_match",
    "comwebq": "avg_f1"
}

def read_metric(filepath, metric_path, verbose=True):
    if not os.path.exists(filepath):
        if verbose:
            print(f"file not exists: {filepath}")
        return None
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
            
            if verbose:
                print(f"file contents: {content[:200]}..." if len(content) > 200 else content)
            
            data = json.loads(content)
        
        if metric_path in data:
            value = data[metric_path]
            if verbose:
                print(f"find metric - {metric_path}: {value}")
            return float(value)
        
        if verbose:
            print(f"file {filepath} has key: {list(data.keys())}")
            for key in data.keys():
                print(f"key: '{key}', type: {type(key)}, length: {len(key)}, ASCII: {[ord(c) for c in key]}")
        
        if '/' in metric_path:
            parts = metric_path.split('/')
            if verbose:
                print(f"trying to find: {metric_path}, split into: {parts}")
            
            for key in data.keys():
                if key.replace(' ', '').lower() == metric_path.replace(' ', '').lower():
                    return float(data[key])
            
            current = data
            for i, part in enumerate(parts):
                found = False
                if isinstance(current, dict):
                    if part in current:
                        current = current[part]
                        found = True
                    else:
                        for key in current.keys():
                            if key.replace(' ', '').lower() == part.replace(' ', '').lower():
                                current = current[key]
                                found = True
                                break
                
                if not found:
                    if verbose:
                        available_keys = list(current.keys()) if isinstance(current, dict) else "not dict"
                        print(f"no key found '{part}' (part {i+1}), current key: {available_keys}")
                    return None
            
            
            if isinstance(current, (int, float, str)):
                return float(current)
            else:
                if verbose:
                    print(f"not numbers: {type(current)}")
                return None
        
        else:
            for key in data.keys():
                if key.replace(' ', '').lower() == metric_path.replace(' ', '').lower():
                    return float(data[key])
            return None
            
    except json.JSONDecodeError as e:
        if verbose:
            print(f"loading JSON failed: {filepath}, -- {e}")
        return None
    except Exception as e:
        if verbose:
            print(f"loading {filepath} failed: {e}")
            import traceback
            traceback.print_exc()
        return None

def compute_metrics_for_order(order_name, verbose=True):
    if order_name not in ORDERS:
        raise ValueError(f"error order: {order_name}, available: {list(ORDERS.keys())}")
    
    task_order = ORDERS[order_name]
    T = len(task_order)
    
    single_base = f"../saves/KBQA_CL_2stages_{order_name}_single_llama/llama3"
    seq_base = f"../saves/5epochs/KBQA_CL_2stages_{order_name}_llama3/llama3/seq_lora/stage2"
    
    a = np.full((T+1, T+1), np.nan)
    
    # loading a0t
    for t in range(1, T+1):
        task_name = task_order[t-1]
        metric_path = METRIC_PATHS[task_name]
        filepath = f"{single_base}/task{t}/result/test{t}_stage2_metrics.json"
        
        if verbose:
            print(f"\nloading - Task{t} ({task_name}): {filepath}")
            print(f"PATH: {metric_path}")
        
        metric_value = read_metric(filepath, metric_path, verbose)
        if metric_value is not None:
            a[0, t] = metric_value
            if verbose:
                print(f"a[0,{t}] = {metric_value}")
    
    # loading a_i,j
    for i in range(1, T+1): 
        for j in range(1, i+1):
            task_name = task_order[j-1]
            metric_path = METRIC_PATHS[task_name]
            filepath = f"{seq_base}/task{i}/result/task{j}_test{i}_metrics.json"
            
            if verbose:
                print(f"\nTrained Task{i}，Evaluating Task{j} ({task_name}): {filepath}")
                print(f"PATH: {metric_path}")
            
            metric_value = read_metric(filepath, metric_path, verbose)
            if metric_value is not None:
                a[i, j] = metric_value
                if verbose:
                    print(f"a[{i},{j}] = {metric_value}")
    
    
    # 1. Average Performance (AP)
    valid_last_row = [a[T, t] for t in range(1, T+1) if not np.isnan(a[T, t])]
    AP = np.mean(valid_last_row) if valid_last_row else np.nan
    
    # 2. Forward Transfer (FWT)
    fwt_terms = []
    for t in range(1, T+1):
        if not np.isnan(a[t, t]) and not np.isnan(a[0, t]):
            fwt_terms.append(a[t, t] - a[0, t])
    FWT = np.mean(fwt_terms) if fwt_terms else np.nan
    
    # 3. Backward Transfer (BWT)
    bwt_terms = []
    for t in range(1, T):
        if not np.isnan(a[T, t]) and not np.isnan(a[t, t]):
            bwt_terms.append(a[T, t] - a[t, t])
    BWT = np.mean(bwt_terms) if bwt_terms else np.nan


    return {
        "order": order_name,
        "task_order": task_order,
        "a_matrix": a.tolist(),
        "AP": round(float(AP), 5) if not np.isnan(AP) else "N/A",
        "FWT": round(float(FWT), 5) if not np.isnan(FWT) else "N/A",
        "BWT": round(float(BWT), 5) if not np.isnan(BWT) else "N/A"
    }

def main():
    all_results = {}
    verbose = True
    
    for order_name in ORDERS.keys():
        try:
            print(f"\n{'='*50}")
            print(f"Calculating {order_name} ...")
            print(f"{'='*50}")
            
            results = compute_metrics_for_order(order_name, verbose)
            all_results[order_name] = results


            print(f"\n{order_name}:")
            print(f"Average Performance (AP): {results['AP']}")
            print(f"Forward Transfer (FWT): {results['FWT']}")
            print(f"Backward Transfer (BWT): {results['BWT']}")
            
        except Exception as e:
            print(f"{order_name} failed: {e}")
            import traceback
            traceback.print_exc()
    

    print("\n\n========== all orders ==========")
    print("order\t\tAP\tFWT\tBWT")
    for order_name, results in all_results.items():
        print(f"{order_name}\t\t{results['AP']}\t{results['FWT']}\t{results['BWT']}")
    

if __name__ == "__main__":
    main()
