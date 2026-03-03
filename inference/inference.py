from transformers import TextStreamer
from unsloth import FastLanguageModel
from pathlib import Path

model,tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                                          load_in_4bit=True,)
HomePath = Path(__file__).parent.parent.absolute()

output_dir = HomePath / "output"

model.load_adapter(output_dir)
FastLanguageModel.for_inference(model)

news_to_check = "ครม. ตรึงค่าน้ำมันดีเซลที่ 33 บ./ลิตร ถึง 31 ต.ค. 67"

prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
คุณเป็น AI ผู้เชี่ยวชาญด้านการตรวจสอบข่าวภาษาไทย กรุณาวิเคราะห์หัวข้อข่าวต่อไปนี้และตอบว่าเป็น "ข่าวจริง" หรือ "ข่าวปลอม" เท่านั้น

### Input:
{}

### Response:
""".format(news_to_check)

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)


print("--- ผลการวิเคราะห์ข่าว ---")
# print(tokenizer.batch_decode(
#  , skip_special_tokens=True)[0])

model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 5,
    use_cache = True,
    repetition_penalty = 1.2,
    temperature=0.1,
    top_p = 0.9,

)