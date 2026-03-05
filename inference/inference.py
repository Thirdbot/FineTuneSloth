import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from handler import EndpointHandler

load_dotenv()

ENDPOINT_URL = os.getenv("ENDPOINT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(base_url=ENDPOINT_URL, token=HF_TOKEN)


def predict_label(title: str) -> str:
    response = client.text_generation(
        title,
        max_new_tokens=10,
        temperature=0.1,
        do_sample=True,
    )

    generated = response.split("### Response:")[-1].strip()

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

# test endpoint
my_handler = EndpointHandler(path="/home/third/Desktop/FinetuneSloth/full_model_weights")
data = {"inputs": "ธนาคารแห่งประเทศไทยขึ้นอัตราดอกเบี้ยนโยบาย 0.25%"}
print(my_handler(data))
