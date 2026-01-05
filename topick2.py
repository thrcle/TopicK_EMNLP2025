import os
import json
import pickle

# --- (0) segfault 디버깅: 파이썬 레벨에서라도 마지막 출력 잡기
import faulthandler
faulthandler.enable()

# --- (1) 스레드/OMP 충돌 완화 (맥/MPS에서 특히 도움이 됨)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn

from datasets import load_dataset
from accelerate import Accelerator

from openicl import DatasetReader
from openicl import PPLInferencer
from openicl.icl_retriever import TopicKRetriever

from utils import templates, input_columns, output_columns, test_split


# =========================================================
# device 정책 (안정성 우선)
# - Retriever(FAISS + CLF)는 무조건 CPU 고정
# - LLM도 일단 CPU로 돌려서 "무조건 실행"부터 확인
# =========================================================
RETRIEVER_DEVICE = torch.device("cpu")
LLM_DEVICE = torch.device("cpu")  # 일단 실행 우선. 안정화 후 mps/cuda로 바꿔도 됨.

print(f"[Device] RETRIEVER_DEVICE={RETRIEVER_DEVICE}, LLM_DEVICE={LLM_DEVICE}", flush=True)

torch.set_num_threads(1)


# =========================================================
# 설정
# =========================================================
data_dir = "data/"
task_name = "cms"

# model_name = "google/flan-t5-base"
model_name = "EleutherAI/pythia-70m"

seed = 1
# 프롬프트에 실제로 붙는 예시 수 
k_shot = 8


# =========================================================
# 1) dataset 로드
# =========================================================
train_path = os.path.join(data_dir, task_name, "train.jsonl")
test_name = test_split[task_name]
test_path = os.path.join(data_dir, task_name, f"{test_name}.jsonl")

combined_dataset = load_dataset("json", data_files={"train": train_path, "test": test_path})


# =========================================================
# 2) 전처리 산출물 로드
# =========================================================
with open(os.path.join(data_dir, task_name, "topic_emb"), "rb") as f:
    topic_emb = pickle.load(f)

with open(os.path.join(data_dir, task_name, "query_clf_logit"), "rb") as f:
    query_clf_logit = pickle.load(f)


# =========================================================
# 3) Topic_predictor 정의 + 가중치 로드 (Retriever는 CPU 고정)
# =========================================================
class Topic_predictor(nn.Module):
    def __init__(self, topic_emb_tensor):
        super().__init__()
        self.topic_emb = nn.Parameter(topic_emb_tensor, requires_grad=False)
        self.mlp = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 768),
        )

    def forward(self, batch_X):
        # batch_X: (B,768)
        return torch.mm(self.mlp(batch_X), self.topic_emb.T)


# topic_emb dtype/shape 강제 (segfault 방지용)
topic_emb_tensor = torch.tensor(topic_emb, dtype=torch.float32, device=RETRIEVER_DEVICE)
CLF = Topic_predictor(topic_emb_tensor).to(RETRIEVER_DEVICE)
CLF.eval()

# state_dict 로드도 CPU로 강제
state = torch.load(
    os.path.join(data_dir, task_name, "topic_predictor"),
    map_location=RETRIEVER_DEVICE,
    weights_only=True,
)
CLF.load_state_dict(state)


# =========================================================
# 4) topic_knowledge 로드
# =========================================================
print("topic_knowledge 로드", flush=True)
model_short = model_name.split("/")[-1]
topic_knowledge_path = os.path.join(data_dir, task_name, f"topic_knowledge_{model_short}")

if not os.path.exists(topic_knowledge_path):
    raise FileNotFoundError(
        f"topic_knowledge 파일이 없음: {topic_knowledge_path}\n"
        f"이 레포는 모델별로 topic_knowledge를 따로 만들어서 저장하는 구조."
    )

with open(topic_knowledge_path, "rb") as f:
    topic_knowledge = pickle.load(f)


# =========================================================
# 5) result 폴더 생성
# =========================================================
output_json_filepath = os.path.join("result", model_name, task_name)
os.makedirs(output_json_filepath, exist_ok=True)
print("output_json_filepath:", output_json_filepath, flush=True)


# =========================================================
# 6) DatasetReader / Retriever / Inferencer 구성
# =========================================================
print("DatasetReader / Retriever / Inferencer 구성", flush=True)

data_reader = DatasetReader(
    combined_dataset,
    input_columns=input_columns[task_name],
    output_column=output_columns[task_name],
)

# Accelerator는 device를 강하게 고정하기 어렵고 내부에서 섞일 수 있어서,
# "무조건 실행"을 목표로 할 때는 CPU만 쓰도록 강제하는 게 안전함.
# (accelerate가 mps/cuda를 잡는 환경이면 세그폴트 촉발하는 케이스가 있음)
accelerator = Accelerator(cpu=True)

# query_clf_logit / topic_knowledge도 내부에서 torch 변환될 수 있으니
# 최대한 CPU 기반 자료형으로 맞춰두는 게 안전
# (pickle 결과가 numpy/torch/리스트 등일 수 있어서 여기선 그대로 두되,
#  TopicKRetriever 내부에서 변환될 때 device가 CPU로 가도록 위에서 cpu=True 강제)
topick_retriever = TopicKRetriever(
    dataset_reader=data_reader,
    CLF=CLF,
    query_clf_logit=query_clf_logit,
    topic_knowledge=topic_knowledge,
    task_name=task_name,
    ice_num=k_shot,
    tokenizer_name=model_name,
    batch_size=1,
    accelerator=accelerator,
    seed=seed,
)

inferencer = PPLInferencer(
    model_name=model_name,
    tokenizer=model_name,
    output_json_filepath=output_json_filepath,
    batch_size=1,
    accelerator=accelerator,
)


# =========================================================
# 7) inference 실행 + 저장 보장
# =========================================================
output_file = f"TopicK_seed{seed}_{k_shot}_shot"

print("INFER START", flush=True)

# retriever가 넘겨준 ICL 예시 사용 
preds = inferencer.inference(
    topick_retriever,
    ice_template=templates[task_name],
    output_json_filename=output_file,
)
print("INFER END", flush=True)

save_path = os.path.join(output_json_filepath, output_file + ".json")
with open(save_path, "w") as f:
    json.dump(preds, f, ensure_ascii=False, indent=2)

print("Saved:", save_path, flush=True)


