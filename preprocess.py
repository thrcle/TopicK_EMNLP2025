"""
preprocess.py

이 파일의 역할:
1. 데이터 로드
2. 문장 / 토픽 임베딩 생성
3. 토픽 분류용 MLP 로드
4. query → topic score 계산
5. 결과를 pickle로 저장

⚠️ GPU 전용 코드였던 부분을 CPU/GPU 자동 분기되도록 수정함
"""

# =========================================================
# 기본 라이브러리 / 유틸
# =========================================================
import os
import json
import pickle
import math
import re
from pprint import pprint
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
from tqdm import tqdm

# HuggingFace / NLP
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

# openicl 구성요소 (retriever / evaluator 등)
from openicl import (
    PromptTemplate,
    DatasetReader,
    RandomRetriever,
    BM25Retriever,
    ConERetriever,
    TopkRetriever,
    PPLInferencer,
    AccEvaluator,
    DPPRetriever,
    MDLRetriever,
)

from utils import (
    templates,
    input_columns,
    output_columns,
    test_split,
    score_mat_2_rank_mat,
    omit_substrings,
)

# =========================================================
# device 설정 (CPU / GPU 자동 분기)
# =========================================================
# → CUDA 있으면 GPU 사용
# → Mac / CPU 환경이면 자동 CPU
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# CUDA 전용 옵션은 GPU 있을 때만 활성화
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)


# =========================================================
# 데이터 로딩
# =========================================================
print("loading dataset")

task = "cms"
task_name = task
data_dir = "data/"

train_path = data_dir + task_name + "/train.jsonl"
test_name = test_split[task_name]
test_path = data_dir + task_name + "/" + test_name + ".jsonl"

# HuggingFace datasets 로드
combined_dataset = load_dataset(
    "json",
    data_files={
        "train": train_path,
        "test": test_path
    }
)

train_dataset = combined_dataset["train"]
test_dataset = combined_dataset["test"]

# query → topic 매핑 정보
with open(data_dir + task_name + "/qid2tid_dic", "rb") as f:
    qid2tid_dic = pickle.load(f)

# 전체 topic 목록
with open(data_dir + task_name + "/topic_list", "rb") as f:
    topic_list = pickle.load(f)


# =========================================================
# Sentence Embedding 생성
# =========================================================
print("computing query/topic embeddings")

# 문장 임베딩 모델
model_id = "sentence-transformers/all-mpnet-base-v2"
model = SentenceTransformer(model_id)
model = model.to(device)
model.eval()

from torch.utils.data import DataLoader

# -------------------------
# Query embedding
# -------------------------
query_loader = DataLoader(train_dataset["text"], batch_size=1024)

query_embeddings = []
for batch in tqdm(query_loader):
    with torch.no_grad():
        emb = model.encode(batch)
    query_embeddings.extend(emb)

query_emb = np.array(query_embeddings)

# -------------------------
# Topic embedding
# -------------------------
topic_loader = DataLoader(topic_list, batch_size=1024)

topic_embeddings = []
for batch in tqdm(topic_loader):
    with torch.no_grad():
        emb = model.encode(batch)
    topic_embeddings.extend(emb)

topic_emb = np.array(topic_embeddings)

# -------------------------
# 저장
# -------------------------
with open(data_dir + task_name + "/query_emb", "wb") as fw:
    pickle.dump(query_emb, fw, protocol=pickle.HIGHEST_PROTOCOL)

with open(data_dir + task_name + "/topic_emb", "wb") as fw:
    pickle.dump(topic_emb, fw, protocol=pickle.HIGHEST_PROTOCOL)


# =========================================================
# Topic Predictor (MLP)
# =========================================================
"""
query embedding → topic embedding과의 점수 계산 모델

구조:
- 입력: query embedding (768)
- MLP (3-layer)
- topic embedding과 내적 → topic score
"""

class Topic_predictor(nn.Module):
    def __init__(self, topic_emb):
        super().__init__()

        # 학습하지 않는 고정 topic embedding
        self.topic_emb = nn.Parameter(topic_emb, requires_grad=False)

        # 간단한 MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )

    def forward(self, batch_X):
        # (B, 768) → (B, num_topic)
        return torch.mm(self.mlp(batch_X), self.topic_emb.T)


# 모델 생성
CLF = Topic_predictor(torch.FloatTensor(topic_emb)).to(device)

# 사전 학습된 weight 로드 (GPU/CPU 안전)
clf_path = data_dir + task_name + "/topic_predictor"
CLF.load_state_dict(
    torch.load(clf_path, map_location=device, weights_only=True)
)


# =========================================================
# Dataset wrapper (추론용)
# =========================================================
class CLF_dataset(data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, self.X[idx]

    def get_labels(self, batch_indices):
        return self.Y[batch_indices]


train_X = torch.FloatTensor(query_emb)
train_Y = None

clf_dataset = CLF_dataset(train_X, train_Y)
clf_loader = data.DataLoader(
    clf_dataset,
    batch_size=1024,
    shuffle=False,
)


# =========================================================
# Topic score 추론
# =========================================================
with torch.no_grad():
    CLF.eval()
    all_logits = []

    for _, batch in enumerate(clf_loader):
        batch_indices, batch_X = batch
        batch_X = batch_X.to(device)

        output = CLF(batch_X)
        all_logits.extend(output.cpu())

c_clf_logit = torch.stack(all_logits)

# 결과 저장
with open(data_dir + task_name + "/query_clf_logit", "wb") as fw:
    pickle.dump(c_clf_logit, fw, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ preprocess finished successfully.")
