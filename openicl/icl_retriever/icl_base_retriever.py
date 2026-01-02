# """Basic Inferencer"""

# import os
# import torch
# from openicl import BaseRetriever, PromptTemplate
# # [FIX] circular import 방지: 패키지 루트(openicl)에서 다시 import하지 말 것 
# # from openicl.icl_prompt_template import PromptTemplate   # PromptTemplate 정의된 파일명에 맞춰 수정

# from openicl.utils.api_service import *
# from openicl.icl_evaluator import *
# from transformers import (
#     AutoTokenizer,
#     AutoModelForCausalLM,
#     PretrainedConfig,
#     GPT2Tokenizer,
#     AutoConfig,
#     T5ForConditionalGeneration
# )
# from typing import List, Union, Optional, Any
# from accelerate import Accelerator
# from accelerate import init_empty_weights, infer_auto_device_map


# class BaseInferencer:
#     model = None
#     tokenizer = None
#     call_api = False

#     def __init__(
#         self,
#         model_name: Optional[Union[str, Any]] = "gpt2-xl",
#         tokenizer_name: Optional[Union[str, Any]] = None,
#         max_model_token_num: Optional[int] = None,
#         model_config: Optional[PretrainedConfig] = None,
#         batch_size: Optional[int] = 1,
#         accelerator: Optional[Accelerator] = None,
#         output_json_filepath: Optional[str] = "./icl_inference_output",
#         output_json_filename: Optional[str] = "predictions",
#         api_name: Optional[str] = None,
#         model_parallel: Optional[bool] = False,
#         **kwargs
#     ) -> None:

#         self.model_name = model_name
#         self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
#         self.accelerator = accelerator
#         self.is_main_process = True if self.accelerator is None or self.accelerator.is_main_process else False
#         self.api_name = api_name

#         if "no_split_module_classes" not in kwargs:
#             kwargs["no_split_module_classes"] = []
#         if "device_map" not in kwargs:
#             kwargs["device_map"] = None

#         no_split_module_classes = kwargs["no_split_module_classes"]
#         device_map = kwargs["device_map"]

#         self.__init_api(**kwargs)

#         if not self.call_api:
#             self.__init_model(self.model_name, model_config, model_parallel, device_map, no_split_module_classes)
#             self.__init_tokenizer(self.tokenizer_name)
#         else:
#             if self.api_name == "opt-175b":
#                 self.__init_tokenizer(self.tokenizer_name)

#         self.device = "cuda" if torch.cuda.is_available() else "cpu"
#         if self.model is not None:
#             self.model.to(self.device)
#             self.model.eval()

#         self.max_model_token_num = max_model_token_num
#         self.batch_size = batch_size
#         self.output_json_filepath = output_json_filepath
#         self.output_json_filename = output_json_filename

#         if not os.path.exists(self.output_json_filepath):
#             os.makedirs(self.output_json_filepath)

#     # ============================
#     # MODEL INIT
#     # ============================
#     def __init_model(self, model_name, model_config, model_parallel, device_map, no_split_module_classes):
#         if not isinstance(model_name, str):
#             self.model = model_name
#             self.model_name = ""
#             return

#         if not model_parallel:
#             if model_config is not None:
#                 self.model = self.__get_hf_model_from_config(model_name, model_config)
#             else:
#                 self.model = self.__get_hf_model_from_name(model_name)
#         else:
#             if model_config is None:
#                 model_config = AutoConfig.from_pretrained(model_name)

#             with init_empty_weights():
#                 empty_model = AutoModelForCausalLM.from_config(model_config)

#             if device_map is None:
#                 device_map = infer_auto_device_map(
#                     empty_model,
#                     no_split_module_classes=no_split_module_classes,
#                     dtype="float16"
#                 )

#             self.model = AutoModelForCausalLM.from_pretrained(
#                 model_name,
#                 device_map=device_map,
#                 offload_folder="offload",
#                 offload_state_dict=True,
#                 torch_dtype=torch.float16,
#             )

#     def __get_hf_model_from_name(self, model_name):
#         # ❗ 핵심: decoder-only LM만 사용
#         if "t5" in model_name.lower():
#             return T5ForConditionalGeneration.from_pretrained(model_name)
#         else:
#             return AutoModelForCausalLM.from_pretrained(model_name)

#     def __get_hf_model_from_config(self, model_name, model_config):
#         if "t5" in model_name.lower():
#             raise TypeError("T5 does not support from_config for this PPL setup")
#         return AutoModelForCausalLM.from_config(model_config)

#     # ============================
#     # TOKENIZER INIT
#     # ============================
#     def __init_tokenizer(self, tokenizer_name):
#         if self.api_name == "opt-175b":
#             self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
#         else:
#             if not isinstance(tokenizer_name, str):
#                 self.tokenizer = tokenizer_name
#             else:
#                 self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

#         # 🔥 핵심 수정: pad_token 강제 지정
#         if self.tokenizer.pad_token is None:
#             self.tokenizer.pad_token = self.tokenizer.eos_token

#         self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
#         self.tokenizer.padding_side = "left"

#     def __init_api(self, **kwargs):
#         if self.api_name is None:
#             return
#         self.call_api = is_api_available(self.api_name)
#         if not self.call_api:
#             UserWarning(f"api_name '{self.api_name}' is not available")
#         else:
#             update_openicl_api_request_config(self.api_name, **kwargs)

#     def get_input_token_num(self, inputs):
#         return len(self.tokenizer(inputs, verbose=False)["input_ids"])

"""Basic Retriever"""

from datasets import Dataset, DatasetDict
from typing import List, Union, Optional, Tuple, Dict
from openicl import DatasetReader, PromptTemplate
from openicl.utils.check_type import _check_str
from accelerate import Accelerator


class BaseRetriever:
    """Basic In-context Learning Retriever Class
        Base class for In-context Learning Retriever, without any retrieval method.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
    """
    index_ds = None
    test_ds = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        self.dataset_reader = DatasetReader._check_dataset_reader(dataset_reader)
        self.ice_separator = ice_separator
        self.ice_eos_token = ice_eos_token
        self.prompt_eos_token = prompt_eos_token
        self.ice_num = ice_num
        self.index_split = index_split
        self.test_split = test_split
        self.accelerator = accelerator
        self.is_main_process = True if self.accelerator is None or self.accelerator.is_main_process else False
        if isinstance(self.dataset_reader.dataset, Dataset):
            self.index_ds = self.dataset_reader.dataset
            self.test_ds = self.dataset_reader.dataset
            if self.accelerator is not None:
                self.test_ds = self.test_ds.shard(
                    num_shards=self.accelerator.num_processes,
                    index=self.accelerator.process_index
                )
        else:
            self.index_ds = self.dataset_reader.dataset[self.index_split]
            self.test_ds = self.dataset_reader.dataset[self.test_split]

            if self.accelerator is not None:
                self.test_ds = self.test_ds.shard(
                    num_shards=self.accelerator.num_processes,
                    index=self.accelerator.process_index
                )

    def retrieve(self) -> List[List]:
        """
            Retrieve for each data in generation_ds.
            
        Returns:
            `List[List]`: the index list of in-context example for each data in `test_ds`.
        """
        raise NotImplementedError("Method hasn't been implemented yet")

    def get_labels(self, ice_template: Optional[PromptTemplate] = None,
                   prompt_template: Optional[PromptTemplate] = None):
        labels = []
        if prompt_template is not None and isinstance(prompt_template.template, Dict):
            labels = list(prompt_template.template.keys())[:]
        elif ice_template is not None and ice_template.ice_token is not None and isinstance(ice_template.template,
                                                                                            Dict):
            labels = list(ice_template.template.keys())[:]
        else:
            labels = list(set(self.test_ds[self.dataset_reader.output_column]))
        return labels

    def generate_ice(self, idx_list: List[int], ice_template: Optional[PromptTemplate] = None) -> str:
        generated_ice_list = []
        dr = self.dataset_reader
        for idx in idx_list:
            if ice_template is None:
                generated_ice_list.append(' '.join(list(map(str,
                                                            [self.index_ds[idx][ctx] for ctx in dr.input_columns] + [
                                                                self.index_ds[idx][dr.output_column]]))))
            else:
                generated_ice_list.append(
                    ice_template.generate_ice_item(self.index_ds[idx], self.index_ds[idx][dr.output_column]))
        generated_ice = self.ice_separator.join(generated_ice_list) + self.ice_eos_token
        return generated_ice

    def generate_prompt(self, idx: int, ice: str, ice_template: Optional[PromptTemplate] = None,
                        prompt_template: Optional[PromptTemplate] = None) -> Tuple[List[str], List]:
        prompt_list = []
        labels = []
        if prompt_template is not None and isinstance(prompt_template.template, Dict):
            labels = list(prompt_template.template.keys())[:]
        elif ice_template is not None and isinstance(ice_template.template,
                                                     Dict) and ice_template.ice_token is not None:
            labels = list(ice_template.template.keys())[:]
        else:
            labels = list(set(self.test_ds[self.dataset_reader.output_column]))
        for label in labels:
            prompt_list.append(self.generate_label_prompt(idx, ice, label))
        return prompt_list, labels

    def generate_label_prompt(self, idx: int, ice: str, label, ice_template: Optional[PromptTemplate] = None,
                              prompt_template: Optional[PromptTemplate] = None, remain_sep: Optional[bool] = False) -> str:
        if prompt_template is not None:
            return prompt_template.generate_label_prompt_item(self.test_ds[idx], ice, label, remain_sep) + self.prompt_eos_token
        elif ice_template is not None and ice_template.ice_token is not None:
            return ice_template.generate_label_prompt_item(self.test_ds[idx], ice, label, remain_sep) + self.prompt_eos_token
        else:
            prefix_prompt = ' '.join(
                list(map(str, [self.test_ds[idx][ctx] for ctx in self.dataset_reader.input_columns])))
            return ice + prefix_prompt + ' ' + str(label) + self.prompt_eos_token

    def generate_prompt_for_generate_task(self, idx, ice, gen_field_replace_token='',
                                          ice_template: Optional[PromptTemplate] = None,
                                          prompt_template: Optional[PromptTemplate] = None):
        if prompt_template is not None:
            return prompt_template.generate_item(self.test_ds[idx], output_field=self.dataset_reader.output_column,
                                                 output_field_replace_token=gen_field_replace_token,
                                                 ice_field_replace_token=ice) + self.prompt_eos_token
        elif ice_template is not None and ice_template.ice_token is not None:
            return ice_template.generate_item(self.test_ds[idx], output_field=self.dataset_reader.output_column,
                                              output_field_replace_token=gen_field_replace_token,
                                              ice_field_replace_token=ice) + self.prompt_eos_token
        else:
            prefix_prompt = ' '.join(
                list(map(str, [self.test_ds[idx][ctx] for ctx in self.dataset_reader.input_columns])))
            return ice + prefix_prompt + gen_field_replace_token + self.prompt_eos_token
