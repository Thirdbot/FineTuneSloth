# evaluation from local
from pathlib import Path
from tqdm.auto import tqdm
from datasets import load_from_disk
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

from transformers import AutoTokenizer, AutoModelForCausalLM

from handler import EndpointHandler
import matplotlib.pyplot as plt
import seaborn as sns


#load from huggingface
tokenizer = AutoTokenizer.from_pretrained("thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit")
model = AutoModelForCausalLM.from_pretrained("thirdExec/Qwen2.5-1.5B-Instruct-ThaiFakeNews-bnb-4bit")

#load model from local
Home_path = Path(__file__).parent.parent.absolute()
full_model_path = Home_path / "full_model_weights"
my_handler = EndpointHandler(path=full_model_path.as_posix())
# data = {"inputs": "ดื่มน้ำขิงเช้าเย็นรักษามะเร็งได้ 100% แพทย์ยืนยัน"}
# print(my_handler(data))

def predict_label_local(title: str):
    response = my_handler({"inputs": title})
    generated = response["generated_text"].split("### Response:")[-1].strip()
    return generated

def predict_label_from_huggingface(title: str):

    messages = [{"role": "user", "content": title}]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    outputs = model.generate(**inputs, max_new_tokens=10)
    generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return generated



HomePath = Path(__file__).parent.parent.absolute()
dataset_path = HomePath / 'dataset' / 'raw_cleaned_dataset'
dataset = load_from_disk(dataset_path)

# 70 / 30
split_dataset = dataset.train_test_split(test_size=0.2,seed=42)
holder_dataset = split_dataset['train']
test_dataset = split_dataset['test']


print("🔄 Running inference on test set...\n")
y_true = test_dataset["Verification_Status"]
y_pred = []

for title in tqdm(test_dataset["Title"], desc="Predicting"):
    y_pred.append(predict_label_local(title))

# คำนวณ metrics
accuracy = accuracy_score(y_true, y_pred)
f1       = f1_score(y_true, y_pred, average="weighted",
                    labels=["ข่าวจริง", "ข่าวปลอม"])

print(f"\n{'='*55}")
print("📊 EVALUATION RESULTS")
print(f"{'='*55}")
print(f"✅ Accuracy  : {accuracy*100:.2f}%")
print(f"✅ F1 Score  : {f1*100:.2f}%  (weighted)")
print(f"\n📋 Classification Report:")
print(classification_report(y_true, y_pred,
                             labels=["ข่าวจริง", "ข่าวปลอม"],
                             target_names=["ข่าวจริง", "ข่าวปลอม"],
                             zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=["ข่าวจริง", "ข่าวปลอม"])

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["real news", "fake news"],
            yticklabels=["real news", "fake news"],
            linewidths=0.5, annot_kws={"size": 14})
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.title("Confusion Matrix — Thai Fake News Detector", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=120, bbox_inches="tight")
plt.show()
print("✅ Confusion matrix saved!")