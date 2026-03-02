from datasets import load_dataset
from huggingface_hub.errors import HfHubHTTPError
from huggingface_hub.utils import RepositoryNotFoundError

class LoadDataset:
    def __init__(self,repo_id:str,split:str='default'):
        self.repo_id = repo_id
        self.split = split
        self.ds = self._load_data()

    def _load_data(self):
        try:
            return load_dataset(self.repo_id,self.split)

        except KeyError:
            print(f"Error: {self.split} not found")
            return None
        except RepositoryNotFoundError:
            print(f"Error:Repository not found")
            return None
        except HfHubHTTPError as e:
            print(f"HTTP error occurred: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error: {e}")
            return None

    def get_dataset(self):
        return self.ds

# single file loading example
# loader = LoadDataset("EXt1/Thai-True-Fake-News")
# loader.get_dataset()