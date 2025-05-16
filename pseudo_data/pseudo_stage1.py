import os
import json
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

# config
BASE_DATA_DIR        = "KBQA_CL_2stages"
STAGE_DIR            = os.path.join(BASE_DATA_DIR, "stage1")
EMBEDDING_MODEL_PATH = "your_<bge-large-en-v1.5_model>_path"
NUM_CLUSTERS         = 5
DEVICE               = torch.device("cuda" if torch.cuda.is_available() else "cpu")


os.makedirs(STAGE_DIR, exist_ok=True)
task_files = {
    "task1": "task1/grailqa_train1.json",
    "task2": "task2/mtop_train1.json",
    "task3": "task3/spider_train1.json",
    "task4": "task4/compwebq_train1.json",
}

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output.last_hidden_state  # [B, L, D]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    summed = torch.sum(token_embeddings * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts

# 1. loading embedding model
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
model     = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH).to(DEVICE).eval()

for task_name, rel_path in task_files.items():
    data_path = os.path.join(BASE_DATA_DIR, rel_path)
    samples   = load_json(data_path)
    schemas   = [s["output"] for s in samples]

    # 2. embedding all schemas
    all_embs = []
    batch_size = 32
    for i in tqdm(range(0, len(schemas), batch_size), desc=task_name):
        batch = schemas[i : i + batch_size]
        enc   = tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            out  = model(**enc)
            embs = mean_pooling(out, enc.attention_mask)
            embs = torch.nn.functional.normalize(embs, p=2, dim=1)
        all_embs.append(embs.cpu().numpy())
    all_embs = np.vstack(all_embs)  # shape = [N_samples, D]

    # 3. K-Means
    kmeans = KMeans(n_clusters=NUM_CLUSTERS, random_state=42).fit(all_embs)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    # 4. selecting samples
    memory = []
    for c in range(NUM_CLUSTERS):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        embs_c = all_embs[idxs]
        sims   = cosine_similarity(embs_c, centers[c][None])
        best   = idxs[sims.argmax()]
        memory.append(samples[int(best)])

    # 5. save
    out_path = os.path.join(STAGE_DIR, f"{task_name}.json")
    save_json(memory, out_path)
    print(f"[{task_name}] select {len(memory)} samples, save into {out_path}")
