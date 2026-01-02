import os
import pickle
import torch
import numpy as np

# ======================
# 설정
# ======================
task_name = "cms"
# model_name = "google/flan-t5-base"
model_name = "EleutherAI/pythia-70m"
data_dir = "data"

topic_emb_path = os.path.join(data_dir, task_name, "topic_emb")
query_clf_path = os.path.join(data_dir, task_name, "query_clf_logit")

save_path = os.path.join(
    data_dir,
    task_name,
    f"topic_knowledge_{model_name.split('/')[-1]}"
)

# ======================
# load
# ======================
with open(topic_emb_path, "rb") as f:
    topic_emb = pickle.load(f)

with open(query_clf_path, "rb") as f:
    query_clf_logit = pickle.load(f)

query_clf = torch.sigmoid(torch.tensor(query_clf_logit))

# ======================
# TopicK 핵심 아이디어:
# topic별 평균 반응값 계산
# ======================
topic_knowledge = query_clf.mean(dim=0).numpy()

# ======================
# 저장
# ======================
with open(save_path, "wb") as f:
    pickle.dump(topic_knowledge, f)

print("Saved topic knowledge:", save_path)
print("shape:", topic_knowledge.shape)

