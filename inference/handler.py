from typing import Dict,List,Any
from transformers import pipeline

class EndpointHandler:
    def __init__(self, path:str=""):
        self.model = pipeline('text-generation',model=path)
        self.PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

            ### Instruction:
            คุณเป็น AI ผู้เชี่ยวชาญด้านการตรวจสอบข่าวภาษาไทย กรุณาวิเคราะห์หัวข้อข่าวต่อไปนี้และตอบว่าเป็น "ข่าวจริง" หรือ "ข่าวปลอม" เท่านั้น

            ### Input:
            {}

            ### Response:
            """
    def __call__(self, data:Dict[str,Any]):

        inputs = data.pop("inputs",data)

        prompt = self.PROMPT_TEMPLATE.format(inputs)

        prediction = self.model(prompt)
        prediction = prediction[0]
        return prediction


