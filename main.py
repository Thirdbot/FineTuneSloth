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
#small thai model jojo-ai-mst/thai-opt350m-instruct
model_loader = LoadModel("unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit")
tokenizer = model_loader.get_tokenizer()

builder = SlothDatasetBuilder(dataset=dataset,
                              selected_columns=["Title","Verification_Status"],
                              prompt=prompt_template,
                              selected_tokenizer=tokenizer)
formatted = builder.format()
print(formatted[0])
max_token = 256 if builder.check_max_tokens() <= 256 else 1024

print(f"max token: {max_token}")

model_loader.max_length = max_token

builder.save_format(save_form_path.as_posix())

build = builder.build()
# print(build[0])
builder.save_build(save_build_path.as_posix())

# load full lora or Qlora if not Peft then create One else use one
model,tokenizer = model_loader.get_modelLora() if model_loader.is_peft() else model_loader.get_modelQlora()

dataset = load_from_disk(save_form_path)
dataset_split = dataset.train_test_split(test_size=0.2)

train_dataset = dataset_split['train']
eval_dataset = dataset_split['test']

merge_model = Path(__file__).parent.absolute() / "merged_adapter_weights"
full_model = Path(__file__).parent.absolute() / "full_model_weights"

trainer_runner = UnslothTrainer(model=model,
               tokenizer=tokenizer,
               train_dataset=train_dataset,
               eval_dataset=eval_dataset,
               max_seq_length=max_token,
               )

# training-only quantized model and save only quantized
# trainer = trainer_runner.train()
# print(trainer.evaluate())

# trainer.save_model(merge_model.as_posix()) # model local save for merge with inference pretrained


#save and push full model for hub
trainer_runner.save_push(repo_id="thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit",output_dir=full_model.as_posix())

# need full model for inference on hub, Run this if only pushing from save_push is not successful
# model_loader.push_hub(full_model.as_posix(),'thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit','model')

#TODO might use an unsloth's dataset gpt type conversation later.
#TODO find best parameter with wandb?
#TODO new feature
