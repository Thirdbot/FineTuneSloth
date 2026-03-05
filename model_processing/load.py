from huggingface_hub import file_exists,upload_folder
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
from transformers import AutoTokenizer
from unsloth import FastLanguageModel

class LoadModel:
    def __init__(self,repo_id:str,max_token:int=2048):
        self.repo_id = repo_id
        self.tokenizer = self._load_tokenizer()
        self.model = None
        self.max_length = max_token

        # default lora config
        self.r = 32
        self.alpha = 64
        self.target_mod = ["q_proj", "k_proj", "v_proj", "o_proj","gate_proj", "up_proj", "down_proj"]
        self.dropout = 0

    def _load_tokenizer(self):
        try:
            return AutoTokenizer.from_pretrained(self.repo_id)

        except RepositoryNotFoundError:
            print(f"Error:Repository not found")
            return None
        except HfHubHTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Error Loading Tokenizer: {e}")
            return None


    # check if repo has PEFT adapter without loading the model
    def is_peft(self):
        try:
            return file_exists(self.repo_id, "adapter_config.json")
        except Exception as e:
            print(f"Error Checking Peft: {e}")
            return False

    def get_tokenizer(self):
        return self.tokenizer

    def get_modelQlora(self):


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

            self.model = model

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

    # load existing PEFT adapter from repo (no new LoRA created)
    def get_modelLora(self):
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                self.repo_id,
                max_seq_length=self.max_length,
                load_in_4bit=True
            )

            self.model = model

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
