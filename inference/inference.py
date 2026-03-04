from unsloth import FastLanguageModel
import torch

HUB_MODEL_ID = "thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit"

# If the hub repo contains adapter_config.json it is a PEFT-only repo
def _load_model():

    print(f"Loading model from {HUB_MODEL_ID}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        HUB_MODEL_ID,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


model, tokenizer = _load_model()


def predict_label(title: str) -> str:
    prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    คุณเป็น AI ผู้เชี่ยวชาญด้านการตรวจสอบข่าวภาษาไทย กรุณาวิเคราะห์หัวข้อข่าวต่อไปนี้และตอบว่าเป็น "ข่าวจริง" หรือ "ข่าวปลอม" เท่านั้น

    ### Input:
    {}

    ### Response:
    """

    prompt = prompt.format(title)
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()

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