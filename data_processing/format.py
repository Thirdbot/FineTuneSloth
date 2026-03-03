from pathlib import Path
from string import Formatter
from datasets import DatasetDict, Dataset, load_from_disk
from typing import Union
from transformers import PreTrainedTokenizerFast

class SlothDatasetBuilder:
    def __init__(self,
                 dataset:Union[Dataset,DatasetDict],
                 selected_columns:list[str] = None,
                 selected_tokenizer:PreTrainedTokenizerFast=None,
                 prompt:str = None):
        self.dataset = dataset
        self.selected_columns = selected_columns
        self.prompt = prompt
        self.formatted_dataset = None
        self.tokenizer = selected_tokenizer
        self.built_dataset = None

    def get_formatted_dataset(self):
        return self.formatted_dataset

    def format(self):

        if self.dataset is None:
            print('Dataset not found')
            return None
        self.formatted_dataset =self.dataset.map(self._format_prompt,batched=True,remove_columns = self.dataset.column_names)
        return self.formatted_dataset

    def build(self):
        if self.formatted_dataset is None:
            print('Dataset not formatted')
            return None
        if self.tokenizer is None:
            print('Tokenizer not found')
            return None

        try:
            def tokenize_func(examples):
                return self.tokenizer(examples["text"], truncation=True)

            self.built_dataset = self.formatted_dataset.map(tokenize_func, batched=True)
            return self.built_dataset
        except Exception as e:
            print(f"Error Tokenizing: {e}")
            return None

    def _format_prompt(self,examples):
        if self.prompt is None:
            print('Prompt Template not found')
            return None

        if not self._is_formattable():
            return None

        try:
            if self.dataset is None:
                print('Dataset not found')
                return None
            column_data = [examples[col] for col in self.selected_columns]

            texts = []
            for row_values in zip(*column_data):
                texts.append((self.prompt.format(*row_values) + self.tokenizer.eos_token).strip())

            return {"text": texts}

        except Exception as e:
            print(f"Error formatting prompt: {e}")


    def _is_formattable(self):
        try:
            # replacement field
            fields = [f[1] for f in Formatter().parse(self.prompt) if f[1] is not None]

            positional_fields = [f for f in fields if f == "" or f.isdigit()]
            # Validation Logic
            if len(positional_fields) != len(self.selected_columns):
                print( f"Expected {len(positional_fields)} positional args, got {len(self.selected_columns)}")
                return False

            return True
        except Exception as e:
            print(f"Error check parsing format string: {e}")
            return False

    def save_format(self,path:str):
        if self.formatted_dataset is None:
            print("Dataset not found")
            return None

        try:
            self.formatted_dataset.save_to_disk(path)
            print(f"Dataset Saved to {path} Successfully")
        except Exception as e:
            print(f"Error Saving Dataset: {e}")

    def save_build(self,path:str):
        if self.built_dataset is None:
            print("Build not found")
            return None

        try:
            self.built_dataset.save_to_disk(path)
            print(f"Dataset Saved to {path} Successfully")
        except Exception as e:
            print(f"Error Saving Dataset: {e}")

    def check_max_tokens(self):
        if self.formatted_dataset is None:
            print("Dataset not found")
            return None
        if self.tokenizer is None:
            print("Tokenizer not found")
            return None

        token_counts = [len(self.tokenizer.encode(example["text"])) for example in self.formatted_dataset]

        max_tokens = max(token_counts)

        return max_tokens


# HomePath = Path(__file__).parent.parent.absolute()
# dataset_path = HomePath / 'dataset' / 'raw_cleaned_dataset'
# save_path = HomePath / 'dataset' / 'formatted_cleaned_dataset'
# build_path = HomePath / 'dataset' / 'build_dataset'
#
# prompt_template = """ข้อความข่าวมีดังนี้:
# {}
#
# จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
# ประเภท 1: ข่าวจริง
# ประเภท 2: ข่าวปลอม
#
# คำตอบ
# คำตอบที่ถูกต้องคือ: ประเภท {}"""
#
# dataset = load_from_disk(dataset_path)
# builder = SlothDatasetBuilder(dataset=dataset,
#                               selected_columns=["Title","Verification_Status"],
#                               prompt=prompt_template,
#                               selected_tokenizer=None)
# formatted = builder.format()
# print(formatted[0])
# builder.save_format(save_path.as_posix())
# build = builder.build()
# builder.save_build(build_path.as_posix())
