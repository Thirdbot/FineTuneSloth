from transformers import TextStreamer
from unsloth import FastLanguageModel
from pathlib import Path

model,tokenizer = FastLanguageModel.from_pretrained(model_name="jojo-ai-mst/thai-opt350m-instruct",
                                          load_in_4bit=True,)
HomePath = Path(__file__).parent.parent.absolute()

output_dir = HomePath / "output"

model.load_adapter(output_dir)
FastLanguageModel.for_inference(model)

news_to_check = "กรมขนส่งฯ เปิดทำใบขับขี่ออนไลน์ผ่านเพจเฟซบุ๊กโดยไม่ต้องไปสอบที่สำนักงาน เพียงจ่ายค่าธรรมเนียมพิเศษ"

prompt = """ข้อความข่าวมีดังนี้:
{}

จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
ประเภท 1: ข่าวจริง
ประเภท 2: ข่าวปลอม
คำตอบที่ถูกต้องคือ: ประเภท """.format(news_to_check)

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