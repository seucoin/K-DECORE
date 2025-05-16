#!/bin/bash

# a11
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M1/task1/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task1/grailqa_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M1/task1/result.json

# a21
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M2/task1/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task1/grailqa_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M2/task1/result.json

# a22
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M2/task2/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task2/mtop_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M2/task2/result.json

# a31
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M3/task1/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task1/grailqa_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M3/task1/result.json

# a32
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M3/task2/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task2/mtop_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M3/task2/result.json

# a33
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M3/task3/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task3/spider_with_cell_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M3/task3/result.json

# a41
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task1/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task1/grailqa_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task1/result.json

# a42
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task2/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task2/mtop_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task2/result.json

# a43
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task3/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task3/spider_with_cell_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task3/result.json

# a44：使用单独的 task4 评估脚本
python my_eval_json.py \
    --json_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task4/generated_predictions.json \
    --test_file /data1/lyc/Struct_CL/data/KBQA_CL/task4/compwebq_test.json \
    --save_file /data1/lyc/Struct_CL/saves/KBQA_CL/seq_lora/result/M4/task4/result.json
