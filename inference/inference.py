import os

from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from handler import EndpointHandler

from pathlib import Path

load_dotenv()

ENDPOINT_URL = os.getenv("ENDPOINT_URL")
HF_TOKEN = os.getenv("HF_TOKEN")

client = InferenceClient(base_url=ENDPOINT_URL, token=HF_TOKEN)

# predict from endpoints
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

# test endpoint from local
Home_path = Path(__file__).parent.parent.absolute()
full_model_path = Home_path / "full_model_weights"
my_handler = EndpointHandler(path=full_model_path.as_posix())
data = {"inputs": "ดื่มน้ำขิงเช้าเย็นรักษามะเร็งได้ 100% แพทย์ยืนยัน"}
print(my_handler(data))

# Load model directly from huggingface
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit")

#single prediction
messages = [{"role": "user", "content": "ดื่มน้ำขิงเช้าเย็นรักษามะเร็งได้ 100% แพทย์ยืนยัน"}]

inputs = tokenizer.apply_chat_template(
		messages,
		add_generation_prompt=True,
		tokenize=True,
		return_dict=True,
		return_tensors="pt",
	).to(model.device)
outputs = model.generate(**inputs, max_new_tokens=40)
print(tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:],skip_special_tokens=True))