from pathlib import Path

from datasets import load_from_disk

from data_processing.format import SlothDatasetBuilder
from model_processing.load import LoadModel
from train_processing.train import UnslothTrainer

HomePath = Path(__file__).parent.absolute()
dataset_path = HomePath / 'dataset' / 'raw_cleaned_dataset'
save_form_path = HomePath / 'dataset' / 'formatted_cleaned_dataset'
save_build_path = HomePath / 'dataset' / 'build_dataset'

prompt_template = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
คุณเป็น AI ผู้เชี่ยวชาญด้านการตรวจสอบข่าวภาษาไทย กรุณาวิเคราะห์หัวข้อข่าวต่อไปนี้และตอบว่าเป็น "ข่าวจริง" หรือ "ข่าวปลอม" เท่านั้น

### Input:
{}

### Response:
{}"""

dataset = load_from_disk(dataset_path)
model_loader = LoadModel("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
model,tokenizer = model_loader.get_model()

builder = SlothDatasetBuilder(dataset=dataset,
                              selected_columns=["Title","Verification_Status"],
                              prompt=prompt_template,
                              selected_tokenizer=tokenizer)
formatted = builder.format()
print(formatted[0])
max_token = builder.check_max_tokens()
max_token = 256

print(f"max token: {max_token}")

model_loader.max_length = max_token

builder.save_format(save_form_path.as_posix())

build = builder.build()
# print(build[0])
builder.save_build(save_build_path.as_posix())

# load full lora or Qlora if not Peft then create One else use one
model,tokenizer = model_loader.get_model_lora() if model_loader.is_peft(model) else model_loader.get_model_Qlora()

dataset = load_from_disk(save_form_path)
dataset_split = dataset.train_test_split(test_size=0.2)

train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']


trainer_runner = UnslothTrainer(model=model,
               tokenizer=tokenizer,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               )
trainer_runner.train()

#TODO
# might need datacollation for train classication
# might use an unsloth's dataset gpt type conversation later.
# find best parameter with
# new feature
