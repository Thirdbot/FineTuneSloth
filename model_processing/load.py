from accelerate.utils import is_peft_model
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from sympy import false
from unsloth import FastLanguageModel

class LoadModel:
    def __init__(self,repo_id:str,max_token:int=2048):
        self.repo_id = repo_id
        self.model,self.tokenizer = self._load_model()
        self.max_length = max_token

        # default lora config
        self.r = 32
        self.alpha = 64
        self.target_mod = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
        self.dropout = 0.1

    #load base model
    def _load_model(self):
        try:
            return FastLanguageModel.from_pretrained(
                    model_name=self.repo_id,
                    load_in_4bit=False,
                   )

        except RepositoryNotFoundError:
            print(f"Error:Repository not found")
            return None
        except HfHubHTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Error Loading Model: {e}")
            return None


    #check and set peft config
    def is_peft(self,model):

        if model is None:
            print("Model not found")
            return None
        try:
            if is_peft_model(model):
                config = model.peft_config['default']
                self.r,self.alpha,self.target_mod,self.dropout =  config.r,config.lora_alpha,config.target_modules,config.lora_dropout
                return True
            else:
                return False

        except Exception as e:
            print(f"Error Getting Peft Config{e}")
            return false

    def get_model(self):
        return self.model,self.tokenizer

    def get_model_Qlora(self):

        if self.model is None:
            print("Model not found")
            return None
        try:

            model,tokenizer = FastLanguageModel.from_pretrained(
                self.repo_id,
                max_seq_length=self.max_length,
                load_in_4bit=True
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=self.r,
                lora_alpha=self.alpha,
                target_modules=self.target_mod,
                bias="none",
                lora_dropout=self.dropout,
                use_gradient_checkpointing="unsloth"
            )
            return model,tokenizer

        except RepositoryNotFoundError:
            print(f"Error:Repository not found")
            return None
        except HfHubHTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Error Loading Model: {e}")
            return None

    #laod with lora
    def get_model_lora(self):

        if self.model is None:
            print("Model not found")
            return None
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                self.repo_id,
                max_seq_length=self.max_length,
                load_in_4bit=True
            )

            model = FastLanguageModel.get_peft_model(
                model,
                r=self.r,
                lora_alpha=self.alpha,
                target_modules=self.target_mod,
                bias="none",
                lora_dropout=self.dropout,
                use_gradient_checkpointing="unsloth"
            )
            return model, tokenizer

        except RepositoryNotFoundError:
            print(f"Error:Repository not found")
            return None
        except HfHubHTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Error Loading Model: {e}")
            return None



# single file loading example
#typhoon-ai/typhoon-s-thaillm-8b-instruct-research-preview
# loader = LoadModel("jojo-ai-mst/thai-opt350m-instruct")
# model,tokenizer = loader.get_model()
