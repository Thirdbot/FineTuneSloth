from pathlib import Path

from datasets import load_from_disk

from data_processing.format import SlothDatasetBuilder
from model_processing.load import LoadModel
from train_processing.train import UnslothTrainer

HomePath = Path(__file__).parent.absolute()
dataset_path = HomePath / 'dataset' / 'raw_cleaned_dataset'
save_form_path = HomePath / 'dataset' / 'formatted_cleaned_dataset'
save_build_path = HomePath / 'dataset' / 'build_dataset'

prompt_template = """ข้อความข่าวมีดังนี้:
{}

จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
ประเภท 1: ข่าวจริง
ประเภท 2: ข่าวปลอม

คำตอบ
คำตอบที่ถูกต้องคือ: ประเภท {}"""

dataset = load_from_disk(dataset_path)
model_loader = LoadModel("jojo-ai-mst/thai-opt350m-instruct")
model,tokenizer = model_loader.get_model()

builder = SlothDatasetBuilder(dataset=dataset,
                              selected_columns=["Title","Verification_Status"],
                              prompt=prompt_template,
                              selected_tokenizer=tokenizer)
formatted = builder.format()
print(formatted[0])
builder.save_format(save_form_path.as_posix())

build = builder.build()
print(build[0])
builder.save_build(save_build_path.as_posix())

# load full lora or Qlora if not Peft then create One else use one
model,tokenizer = model_loader.get_model_lora() if model_loader.is_peft(model) else model_loader.get_model_Qlora()

dataset = load_from_disk(save_form_path)
dataset_split = dataset.train_test_split(test_size=0.1)

train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']


UnslothTrainer(model=model,tokenizer=tokenizer,train_dataset=train_dataset,eval_dataset=eval_dataset).train()

#TODO
# might need datacollation for train classication
# might use an unsloth's dataset gpt type conversation later.
# find best parameter with
# new feature
