from transformers import TextStreamer
from unsloth import FastLanguageModel
from pathlib import Path

model,tokenizer = FastLanguageModel.from_pretrained(model_name="jojo-ai-mst/thai-opt350m-instruct",
                                          load_in_4bit=True,)
HomePath = Path(__file__).parent.parent.absolute()

output_dir = HomePath / "output"

model.load_adapter(output_dir)
FastLanguageModel.for_inference(model)

news_to_check = "ข่าวดี! รัฐบาลเตรียมโอนเงินเยียวยาพิเศษ 5,000 บาท เข้าบัตรสวัสดิการแห่งรัฐทุกคนในวันศุกร์นี้ เพื่อช่วยลดภาระค่าครองชีพ ใครยังไม่ได้ลงทะเบียนให้รีบคลิกลิงก์ด้านล่างด่วนก่อนสิทธิ์เต็ม"

prompt = """ข้อความข่าวมีดังนี้:
{}

จงจำแนกข่าวนี้ออกเป็นประเภทใดประเภทหนึ่งต่อไปนี้:
ประเภท 1: ข่าวจริง
ประเภท 2: ข่าวปลอม

คำตอบ
คำตอบที่ถูกต้องคือ: ประเภท  """.format(news_to_check)

inputs = tokenizer([prompt], return_tensors = "pt").to("cuda")

text_streamer = TextStreamer(tokenizer)


print("--- ผลการวิเคราะห์ข่าว ---")
_ = model.generate(
    **inputs,
    streamer = text_streamer,
    max_new_tokens = 5,
    use_cache = True,
    repetition_penalty = 1.2,
    temperature=0.1,
    top_p = 0.9
)