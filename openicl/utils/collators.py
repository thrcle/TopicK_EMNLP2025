from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
from transformers import PreTrainedTokenizerBase, BatchEncoding
from transformers.file_utils import PaddingStrategy
import numpy as np


class ListWrapper:
    def __init__(self, data: List[Any]):
        self.data = data

    def to(self, device):
        return self.data


def ignore_pad_dict(features):
    res_dict = {}
    if "metadata" in features[0]:
        res_dict['metadata'] = ListWrapper([x.pop("metadata") for x in features])
    return res_dict


@dataclass
class DataCollatorWithPaddingAndCuda:
    tokenizer: PreTrainedTokenizerBase
    device: object = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = 3000
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> BatchEncoding:
        res_dict = ignore_pad_dict(features)

        has_labels = "labels" in features[0]
        if has_labels:
            labels = [{"input_ids": x.pop("labels")} for x in features]
            labels = self.tokenizer.pad(
                labels,
                padding=True,
                max_length=self.max_length,
                pad_to_multiple_of=self.pad_to_multiple_of,
                return_attention_mask=True,
                return_tensors="pt",
                verbose=False
            )

        # print(features)
        batch = self.tokenizer.pad(
            features,
            padding=True,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_attention_mask=True,
            return_tensors="pt",
            verbose=False
        )

        if has_labels:
            batch['labels'] = labels.input_ids
        batch.update(res_dict)

        # 원본:
        # if self.device:
        #     batch = batch.to(self.device)

        # 수정: MPS/CPU에서 BatchEncoding.to()가 non_blocking 인자로 깨지는 케이스 방지
        if self.device:
            for k, v in batch.items():
                # 텐서만 device로 이동 (ListWrapper/리스트/None 등은 그대로 둠)
                if torch.is_tensor(v):
                    batch[k] = v.to(self.device)


        return batch
