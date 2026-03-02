from transformers import TextStreamer
from unsloth import FastLanguageModel
from pathlib import Path

model,tokenizer = FastLanguageModel.from_pretrained(model_name="jojo-ai-mst/thai-opt350m-instruct",
                                          load_in_4bit=True,)
HomePath = Path(__file__).parent.parent.absolute()

output_dir = HomePath / "output"

model.load_adapter(output_dir)
FastLanguageModel.for_inference(model)

news_to_check = "กระทรวงการคลังเตรียมแจกเงินดิจิทัลวอลเล็ต 10,000 บาท รอบเก็บตกในสัปดาห์หน้า..."

prompt = """ข้อความข่าวมีดังนี้:
{}

จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
ประเภท 1: ข่าวจริง
ประเภท 2: ข่าวปลอม

คำตอบ
คำตอบที่ถูกต้องคือ: ประเภท """.format(news_to_check)

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)


print("--- ผลการวิเคราะห์ข่าว ---")
_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 10, # งานนี้ต้องการแค่ตัวเลข 1 หรือ 2 เท่านั้น 10 token ก็พอครับ
    use_cache = True
)