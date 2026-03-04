import json
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

        self.model = model
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.max_seq_length = max_seq_length
        self.push_private=False

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
            load_best_model_at_end=False,
            save_total_limit=2,
            hub_private_repo=self.push_private

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
            trainer.train(resume_from_checkpoint= True)

            return trainer
        except Exception as e:
            print(f"Train Error:{e}")

    def save_push(self, repo_id: str = None, output_dir: str = None):
        if self.model is None:
            print("Model not found")
            return None
        if self.tokenizer is None:
            print("Tokenizer not found")
            return None

        try:
            # Save merged full model locally (dequantizes LoRA into base weights)
            print(f"Saving merged model to {output_dir} ...")
            self.model.save_pretrained_merged(output_dir, self.tokenizer, save_method="merged_16bit")

            # Strip bitsandbytes quantization_config so inference endpoints don't require it
            config_path = Path(output_dir) / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    cfg = json.load(f)
                cfg.pop("quantization_config", None)
                with open(config_path, "w") as f:
                    json.dump(cfg, f, indent=2)

            print("Local save complete.")
        except Exception as e:
            print(f"Save Error (local): {e}")
            return None

        try:
            # Push merged full model to hub so inference providers can load it
            print(f"Pushing merged model to hub: {repo_id} ...")
            self.model.push_to_hub_merged(repo_id, self.tokenizer, save_method="merged_16bit", private=self.push_private)
            print("Hub push complete.")
        except Exception as e:
            print(f"Save Error (hub push): {e}")

