#!/usr/bin/env bash
set -euo pipefail
IFS=$'\n\t'

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# —— model —— #
MODEL_NAME="llama3"
TEMPLATE="llama3"
TRAIN_GPUS="0,1"
INFER_GPU="0"

# —— path —— #
DATA_ROOT="../data/KBQA_CL_2stages_order3_llama3"
INFO_JSON="${DATA_ROOT}/dataset_info.json"
export INFO_JSON
BASE_MODEL_PATH="your_<Llama-3-8B-Instruct>_model_path"
EVAL_SCRIPT="../evaluate/evaluate.py"

SAVE_ROOT="../saves/KBQA_CL_2stages_order3_llama3/${MODEL_NAME}/seq_lora"
CFG_ROOT="../config/KBQA_CL_2stages_order3_llama3/${MODEL_NAME}/seq_lora"

# —— hyperparameters —— #
LORA_RANK=8; LORA_ALPHA=16; LORA_DROPOUT=0
LR=5.0e-05; EPOCHS=5; BATCH_SIZE=1; MAX_SAMPLES=100000
GRAD_ACCUM=16; SAVE_STEPS=1000

# —— tasks —— #
TASKS=(task1 task2 task3 task4)

# Stage-1
RUN_EVAL_STAGE1=false


get_gold_file () {
  local key="$1"                       
  local rel
  rel=$(python - "$key" <<'PY'
import json, os, sys, pathlib
info_file = os.environ["INFO_JSON"]     
key = sys.argv[1]                      
with open(info_file, 'r') as f:
    data = json.load(f)
if key not in data or "file_name" not in data[key]:
    sys.exit(1)                        
print(data[key]["file_name"])
PY
  ) || { echo "CAN NOT FIND ${key} in ${INFO_JSON}" >&2 ; exit 1; }

  echo "${DATA_ROOT}/${rel}"
}


make_yaml () {
cat > "$1" <<EOF
bf16: true
cutoff_len: 1024
dataset: $2
dataset_dir: ${DATA_ROOT}
do_train: true
finetuning_type: lora
flash_attn: auto
gradient_accumulation_steps: ${GRAD_ACCUM}
include_num_input_tokens_seen: true
learning_rate: ${LR}
lora_alpha: ${LORA_ALPHA}
lora_dropout: ${LORA_DROPOUT}
lora_rank: ${LORA_RANK}
lora_target: all
lr_scheduler_type: cosine
max_grad_norm: 1.0
max_samples: ${MAX_SAMPLES}
model_name_or_path: ${BASE_MODEL_PATH}
num_train_epochs: ${EPOCHS}
optim: adamw_torch
output_dir: $3
per_device_train_batch_size: ${BATCH_SIZE}
plot_loss: true
report_to: none
save_steps: ${SAVE_STEPS}
save_strategy: "no"
save_total_limit: 1
stage: sft
template: ${TEMPLATE}
warmup_steps: 0
EOF
}

run_train () {
  export CUDA_VISIBLE_DEVICES=${TRAIN_GPUS}
  python run.py "$1"
}
run_infer () {
  local adapter=$1 ds=$2 outfile=$3
  export CUDA_VISIBLE_DEVICES=${INFER_GPU}
  python vllm_infer.py \
      --model_name_or_path  ${BASE_MODEL_PATH} \
      --adapter_name_or_path "${adapter}" \
      --dataset "${ds}" --dataset_dir "${DATA_ROOT}" \
      --template ${TEMPLATE} --cutoff_len 1024 \
      --save_name "${outfile}" \
      --temperature 0.0 --top_p 0.7 --top_k 50 \
      --max_new_tokens 1024 --repetition_penalty 1.0
}
run_eval () {
  local pred=$1 gold=$2 out=$3
  pushd "$(dirname "${EVAL_SCRIPT}")" >/dev/null
  python "$(basename "${EVAL_SCRIPT}")" \
      --pred_file "${pred}" --test_file "${gold}" \
      --save_file "${out}" --kb_endpoint "http://10.201.234.130:3001/sparql"
  popd >/dev/null
}

################################################################################
# 2. Stage-1 Training / Inference
################################################################################
echo -e "\n===================== Stage-1 Training / Inference ====================="

STAGE1_CFG="${CFG_ROOT}/stage1_train"
STAGE1_SAVE="${SAVE_ROOT}/stage1"
mkdir -p "${STAGE1_CFG}" "${STAGE1_SAVE}"

for ((i=0;i<${#TASKS[@]};i++)); do
  t_idx=$((i+1))
  task="${TASKS[i]}"
  dataset="${task}_stage1"
  yaml="${STAGE1_CFG}/${task}.yaml"
  lora_dir="${STAGE1_SAVE}/${task}"
  result_dir="${lora_dir}/result"; mkdir -p "${result_dir}"

  echo -e "\n--- [Stage-1] Training ${dataset} ---"
  make_yaml "${yaml}" "${dataset}" "${lora_dir}"

  #loading pre adapter
  [[ ${t_idx} -gt 1 ]] && \
    sed -i "2i adapter_name_or_path: ${STAGE1_SAVE}/${TASKS[i-1]}" "${yaml}"

  [[ -s "${lora_dir}/adapter_config.json" ]] || run_train "${yaml}"

  for ((j=1;j<=t_idx;j++)); do
    test_ds="test${j}_stage1"
    pred="${result_dir}/${test_ds}.json"
    [[ -s "${pred}" ]] || run_infer "${lora_dir}" "${test_ds}" "${pred}"

    if ${RUN_EVAL_STAGE1}; then
      metrics="${result_dir}/${test_ds}_metrics.json"
      gold=$(get_gold_file "${test_ds}")
      [[ -s "${metrics}" ]] || run_eval "${pred}" "${gold}" "${metrics}"
    fi
  done
done

################################################################################
# 3. generating Stage-2 test files
################################################################################
echo -e "\n===================== generating Stage-2 test files ====================="
pushd "${DATA_ROOT}/task1" >/dev/null && python mtop_split_test.py   && popd >/dev/null
pushd "${DATA_ROOT}/task2" >/dev/null && python grailqa_split_test.py     && popd >/dev/null
pushd "${DATA_ROOT}/task3" >/dev/null && python compwebq_split_test.py   && popd >/dev/null
pushd "${DATA_ROOT}/task4" >/dev/null && python spider_split_test.py && popd >/dev/null

################################################################################
# 4. Stage-2 Training / Inference / Evaluate
################################################################################
echo -e "\n===================== Stage-2 Training / Inference / Evaluate ====================="

STAGE2_CFG="${CFG_ROOT}/stage2_train"
STAGE2_SAVE="${SAVE_ROOT}/stage2"
mkdir -p "${STAGE2_CFG}" "${STAGE2_SAVE}"

for ((i=0;i<${#TASKS[@]};i++)); do
  t_idx=$((i+1))
  task="${TASKS[i]}"
  dataset="${task}_stage2"
  yaml="${STAGE2_CFG}/${task}.yaml"
  lora_dir="${STAGE2_SAVE}/${task}"
  result_dir="${lora_dir}/result"; mkdir -p "${result_dir}"

  echo -e "\n--- [Stage-2] Training ${dataset} ---"
  make_yaml "${yaml}" "${dataset}" "${lora_dir}"
  [[ ${t_idx} -gt 1 ]] && \
    sed -i "2i adapter_name_or_path: ${STAGE2_SAVE}/${TASKS[i-1]}" "${yaml}"

  [[ -s "${lora_dir}/adapter_config.json" ]] || run_train "${yaml}"


  for ((j=1;j<=t_idx;j++)); do
    test_ds="task${j}_test${t_idx}"
    pred="${result_dir}/${test_ds}.json"
    metrics="${result_dir}/${test_ds}_metrics.json"

    [[ -s "${pred}" ]]    || run_infer "${lora_dir}" "${test_ds}" "${pred}"
    gold=$(get_gold_file "${test_ds}")
    [[ -s "${metrics}" ]] || run_eval  "${pred}" "${gold}" "${metrics}"
  done
done

echo -e "\n🎉  Finished both 2 stages! \nStage-1 LoRA saved in: ${STAGE1_SAVE}\nStage-2 LoRA saved in: ${STAGE2_SAVE}"
