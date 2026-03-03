from unsloth import FastLanguageModel
from pathlib import Path
import torch

model,tokenizer = FastLanguageModel.from_pretrained(model_name="unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit",
                                          load_in_4bit=True,)
HomePath = Path(__file__).parent.parent.absolute()

output_dir = HomePath / "output"

model.load_adapter(output_dir)
FastLanguageModel.for_inference(model)

news_to_check = "ครม. ตรึงค่าน้ำมันดีเซลที่ 33 บ./ลิตร ถึง 31 ต.ค. 67"




def predict_label(title: str) -> str:
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    คุณเป็น AI ผู้เชี่ยวชาญด้านการตรวจสอบข่าวภาษาไทย กรุณาวิเคราะห์หัวข้อข่าวต่อไปนี้และตอบว่าเป็น "ข่าวจริง" หรือ "ข่าวปลอม" เท่านั้น

    ### Input:
    {}

    ### Response:
    """

    prompt = prompt.format(title)   # ไม่ใส่ label (ให้ model ทาย)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens = 10,
            temperature    = 0.1,
            do_sample      = True,
            pad_token_id   = tokenizer.eos_token_id,
        )

    # Decode เฉพาะ token ที่ generate ใหม่
    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

    # Map output → clean label
    if "จริง" in generated:
        return "ข่าวจริง"
    elif "ปลอม" in generated:
        return "ข่าวปลอม"
    else:
        return generated

student_questions = [
    ("รัฐบาลแจกเงิน 10,000 บาทให้ทุกคนฟรีผ่าน LINE",              "ข่าวปลอม"),
    ("กรมอุตุฯ เตือนพายุเข้าไทย 3 จังหวัดใต้รับมือ",              "ข่าวจริง"),
    ("ดื่มน้ำขิงเช้าเย็นรักษามะเร็งได้ 100% แพทย์ยืนยัน",        "ข่าวปลอม"),
    ("ธนาคารแห่งประเทศไทยขึ้นอัตราดอกเบี้ยนโยบาย 0.25%",         "ข่าวจริง"),
    ("วัคซีนโควิดทำให้ DNA เปลี่ยนและควบคุมจิตใจ",               "ข่าวปลอม"),
]

print("Student Q&A Results")
print("="*80)
for i, (question, expected) in enumerate(student_questions, 1):
    prediction = predict_label(question)
    status     = "ถูก" if prediction == expected else "ผิด"
    print(f"\nQ{i}: {question}")
    print(f"   Expected  : {expected}")
    print(f"   Predicted : {prediction}  {status}")
print("\n" + "="*80)

# formatted_text = prompt.format(news_to_check)
#
# inputs = tokenizer([formatted_text], return_tensors = "pt").to("cuda")
#
# text_streamer = TextStreamer(tokenizer)

#
# print("--- ผลการวิเคราะห์ข่าว ---")
# # print(tokenizer.batch_decode(
# #  , skip_special_tokens=True)[0])
#
# model.generate(
#     **inputs,
#     streamer = text_streamer,
#     max_new_tokens = 5,
#     use_cache = True,
#     repetition_penalty = 1.2,
#     temperature=0.1,
#     top_p = 0.9,
#
# )