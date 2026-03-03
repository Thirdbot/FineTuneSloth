from pathlib import Path
from typing import Union
from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerFast, TrainingArguments
from unsloth import FastLanguageModel, is_bfloat16_supported
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM

class UnslothTrainer:
    def __init__(self,model:FastLanguageModel=None,
                 tokenizer:PreTrainedTokenizerFast=None,
                 train_dataset:Union[Dataset,DatasetDict]=None,
                 eval_dataset:Union[Dataset,DatasetDict]=None,
                 max_seq_length:int=256):

        self.output_dir = Path(__file__).parent.parent.absolute() / "output"
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.max_seq_length = max_seq_length

        self.args = TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_ratio=0.05,
            num_train_epochs=1.0,
            learning_rate=5e-5,
            fp16= not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.05,
            lr_scheduler_type='cosine',
            output_dir='./results',
            eval_strategy="steps",
            eval_steps=50,
            save_strategy="steps",
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            load_best_model_at_end=True,
            save_total_limit=2,

        )
        self.response_template = "### Response:\n"
        self.collator = DataCollatorForCompletionOnlyLM(
            response_template=self.response_template,
            tokenizer=tokenizer,
            mlm=False,
        )
        self.results_path = Path(__file__).parent.parent.absolute() / "results"

    def train(self):
        if self.model is None:
            print("Model not found")
            return None
        if self.tokenizer is None:
            print("Tokenizer not found")
            return None
        if self.train_dataset is None:
            print("Train Dataset not found")
            return None

        if self.eval_dataset is None:
            print("Eval Dataset not found")
            return None
        try:
            trainer = SFTTrainer(
                model=self.model, #type: ignore
                train_dataset=self.train_dataset,
                eval_dataset=self.eval_dataset,
                tokenizer=self.tokenizer, #type: ignore
                dataset_text_field="text", #type: ignore
                max_seq_length=self.max_seq_length, #type: ignore
                args=self.args,
                data_collator=self.collator,
            )
            trainer.train(resume_from_checkpoint= False)

            return trainer.save_model(self.output_dir.as_posix())
        except Exception as e:
            print(f"Train Error:{e}")

