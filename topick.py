import sys
# print(sys.path)

from openicl import PromptTemplate
from openicl import DatasetReader
from openicl import RandomRetriever, BM25Retriever, ConERetriever, TopkRetriever, PPLInferencer, AccEvaluator, DPPRetriever, MDLRetriever
from datasets import load_dataset, concatenate_datasets
from accelerate import Accelerator
import math
import os
import re
import json
from pprint import pprint
import numpy as np
import transformers
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data as data
import torch.optim as optim
from collections import Counter
from scipy.sparse import csr_matrix

from tqdm import tqdm
import pickle

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import MultiLabelBinarizer

from openai import OpenAI
import openai
import faiss

from utils import templates, input_columns, output_columns, test_split, score_mat_2_rank_mat, omit_substrings


# =========================================================
# 추가: device 자동 분기 (CUDA 강제 사용 제거 목적)
# =========================================================
if torch.cuda.is_available():
    device = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)


# =========================================================
# 원본 CUDA 강제 설정 (CPU 환경에서 혼선/에러 유발 가능 -> 주석 처리)
# =========================================================
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"


#################### load dataset
print("loading dataset")

task = 'cms'
task_name = task
data_dir = 'data/'

train_path = data_dir + task_name + '/train.jsonl'
test_name = test_split[task_name]
test_path = data_dir + task_name + '/' + test_name + '.jsonl'

combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})

train_dataset = combined_dataset["train"]
test_dataset = combined_dataset["test"]

with open(data_dir + task_name + "/qid2tid_dic", 'rb') as f:
    qid2tid_dic = pickle.load(f)

with open(data_dir + task_name + "/topic_list", 'rb') as f:
    topic_list = pickle.load(f)

#################### query / topic embedding
print("computing query/topic embeddings")

# =========================================================
# 원본 CUDA backend 설정 (CUDA 있을 때만 적용되도록 수정)
# =========================================================
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
if torch.cuda.is_available():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)

model_id = 'sentence-transformers/all-mpnet-base-v2'  # #"sentence-transformers/multi-qa-mpnet-base-cos-v1" #
model = SentenceTransformer(model_id)

# =========================================================
# 원본: model.to("cuda") -> device 기반으로 수정
# =========================================================
# model = model.to("cuda")
model = model.to(device)

model = model.eval()

from torch.utils.data import DataLoader
dataloader = DataLoader(train_dataset['text'], batch_size=1024)
emb_list = []
for _, entry in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        emb = model.encode(entry)
    emb_list.extend(emb)
query_emb = np.array(emb_list)

dataloader = DataLoader(topic_list, batch_size=1024)
emb_list = []
for _, entry in enumerate(tqdm(dataloader)):
    with torch.no_grad():
        emb = model.encode(entry)
    emb_list.extend(emb)
topic_emb = np.array(emb_list)

with open(data_dir + task_name + "/query_emb", 'wb') as fw:
    pickle.dump(query_emb, fw, protocol=pickle.HIGHEST_PROTOCOL)

with open(data_dir + task_name + "/topic_emb", 'wb') as fw:
    pickle.dump(topic_emb, fw, protocol=pickle.HIGHEST_PROTOCOL)

#################### covered topic prediction
print("computing required topics")
class Topic_predictor(nn.Module):
    def __init__(self, topic_emb):
        super(Topic_predictor, self).__init__()

        self.topic_emb = nn.Parameter(topic_emb, requires_grad=False)
        self.mlp = nn.Sequential(nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768), nn.ReLU(), nn.Linear(768, 768))
        
    def forward(self, batch_X):
        
        output = torch.mm(self.mlp(batch_X), self.topic_emb.T)
        return output

# =========================================================
# 원본: .to('cuda') -> device 기반으로 수정
# =========================================================
# CLF = Topic_predictor(torch.FloatTensor(topic_emb)).to('cuda')
CLF = Topic_predictor(torch.FloatTensor(topic_emb)).to(device)

# =========================================================
# 원본 torch.load는 GPU 저장본이면 CPU에서 터질 수 있음 -> map_location 적용
# =========================================================
# CLF.load_state_dict(torch.load(data_dir + task_name + "/topic_predictor", weights_only=True))
CLF.load_state_dict(
    torch.load(
        data_dir + task_name + "/topic_predictor",
        map_location=device,
        weights_only=True
    )
)

class CLF_dataset(data.Dataset):
    def __init__(self, X, Y):

        super(CLF_dataset, self).__init__()
        self.X = X
        self.Y = Y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return idx, self.X[idx]

    def get_labels(self, batch_indices):
        return self.Y[batch_indices]
    
num_topic = topic_emb.shape[0]
num_query = query_emb.shape[0]

train_X = torch.FloatTensor(query_emb)
train_Y = None

CLF_train_dataset = CLF_dataset(train_X, train_Y)
CLF_test_loader = data.DataLoader(CLF_train_dataset, batch_size=1024, shuffle=False)

with torch.no_grad():
    CLF_test = CLF.eval()
    c_clf_logit = []
    for _, mini_batch in enumerate(CLF_test_loader):
        batch_indices, batch_X = mini_batch

        # =====================================================
        # 원본: batch_X.to('cuda') -> device 기반으로 수정
        # =====================================================
        # batch_X = batch_X.to('cuda')
        batch_X = batch_X.to(device)

        output = CLF_test(batch_X)
        c_clf_logit.extend(output.cpu())

c_clf_logit = torch.stack(c_clf_logit)
with open(data_dir + task_name + "/query_clf_logit", 'wb') as fw:
    pickle.dump(c_clf_logit, fw, protocol=pickle.HIGHEST_PROTOCOL)

print("Done.")

################# 추가
# =========================
# 추가: TopicK inference 실행 + result 저장
# =========================

RUN_INFERENCE = True       # 필요할 때만 True
SKIP_PREPROCESS = False    # 전처리 파일 이미 있으면 True로 바꿔서 시간 절약

if SKIP_PREPROCESS:
    print("Skip preprocess (user flag).")

if RUN_INFERENCE:
    print("Start inference to generate result json...")

    # prediction.py가 읽는 경로와 동일하게
    # model_name = "meta-llama/Llama-3.2-3B-Instruct"
    # model_name = "google/flan-t5-base"
    model_name = "EleutherAI/pythia-70m"
    ppl_model_name = "EleutherAI/pythia-70m"

    seed = 1
    k_shot = 8

    # 1) result 폴더 생성
    output_json_filepath = os.path.join("result", model_name, task_name)
    os.makedirs(output_json_filepath, exist_ok=True)
    print("output_json_filepath:", output_json_filepath)

    # 2) 전처리 산출물 로드 (이미 위에서 만들었으면 그대로 써도 됨)
    #    - query_emb/topic_emb/query_clf_logit 는 위에서 생성됨
    #    - topic_knowledge는 레포에 저장 규칙이 있을 텐데, 없으면 먼저 생성해야 함
    #      (파일명이 다르면 여기만 맞춰줘야 함)
    model_short = model_name.split("/")[-1]
    topic_knowledge_path = os.path.join(data_dir, task_name, f"topic_knowledge_{model_short}")

    if not os.path.exists(topic_knowledge_path):
        raise FileNotFoundError(
            f"topic_knowledge 파일이 없어: {topic_knowledge_path}\n"
            f"repo에서 topic_knowledge 생성 단계가 따로 있을 수 있어. 파일명/생성 스크립트를 확인해야 함."
        )

    with open(topic_knowledge_path, "rb") as f:
        topic_knowledge = pickle.load(f)

    # 3) DatasetReader 구성
    data_reader = DatasetReader(
        combined_dataset,
        input_columns=input_columns[task_name],
        output_column=output_columns[task_name]
    )

    # 4) Retriever 생성 (TopicKRetriever)
    accelerator = Accelerator()

    # TopicKRetriever는 openicl.icl_retriever.__init__에 export되어 있음
    from openicl.icl_retriever import TopicKRetriever

    topick_retriever = TopicKRetriever(
        dataset_reader=data_reader,
        CLF=CLF,                      # 위에서 로드한 CLF
        query_clf_logit=c_clf_logit,  # 위에서 만든 query_clf_logit
        topic_knowledge=topic_knowledge,
        task_name=task_name,
        ice_num=k_shot,
        tokenizer_name=model_name,
        batch_size=1,
        accelerator=accelerator,
        seed=seed
    )

    # 5) Inferencer 생성 + inference
    # inferencer = PPLInferencer(
    #     model_name=model_name,
    #     tokenizer=model_name,
    #     output_json_filepath=output_json_filepath,
    #     batch_size=1,
    #     accelerator=accelerator
    # )
    inferencer = PPLInferencer(
    model_name=ppl_model_name,
    tokenizer=ppl_model_name,
    output_json_filepath=output_json_filepath,
    batch_size=1,
    accelerator=accelerator
)


    output_file = f"TopicK_seed{seed}_{k_shot}_shot"  # prediction.py가 '8_shot'을 찾으니까 이 포맷 유지
    preds = inferencer.inference(
        topick_retriever,
        ice_template=templates[task_name],
        output_json_filename=output_file
    )

    # 6) 저장 보장: inferencer 내부 저장이 안 되더라도 무조건 파일 생성
    save_path = os.path.join(output_json_filepath, output_file + ".json")
    with open(save_path, "w") as f:
        json.dump(preds, f, ensure_ascii=False, indent=2)
    print("Saved:", save_path)

    # CUDA에서만 캐시 비우기
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

