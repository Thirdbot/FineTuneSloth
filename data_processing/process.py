from pathlib import Path

from datasets import DatasetDict,Dataset
from collections import Counter

from collections.abc import Iterable

from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

import pandas as pd

from typing import Union

from load import LoadDataset

class ProcessDataset:
    def __init__(self,dataset:Union[DatasetDict,Dataset]):

        self.dataset = dataset

    def get_dataset(self):
        return self.dataset

    def select_split(self,default_split='train'):

        if self.dataset is None:
            print("Dataset not found")
            return None

        # the dataset has only one split
        if isinstance(self.dataset,Dataset):
            #might be potentials bug since no check
            print(f'set dataset to use split:{default_split}')
            self.dataset = self.dataset[default_split]
            return self.dataset

        elif isinstance(self.dataset,DatasetDict):

            if hasattr(self.dataset,'column_names'):
                print(f'avaliable columns:{self.dataset.column_names}')

            if hasattr(self.dataset, 'keys'):
                keys = list(self.dataset.keys())


                # multiple split
                if len(keys) > 1:

                    split = default_split if isinstance(keys,
                                                    Iterable) and default_split in keys else keys.pop()
                    print(f'Selected split:{split}')
                    # set new dataset as dataset selected split column
                    try:
                        print(f'set dataset to use split:{split}')
                        self.dataset = self.dataset[split]
                    except KeyError:
                        print(f"Error Splitting: {keys} not found")
                        return None

                    return self.dataset

                else:
                    # default dataset
                    print(f'set dataset to use split:{keys[0]}')
                    self.dataset = self.dataset[keys[0]]
                    return self.dataset
            else:
                print("Dataset not have any column_names")
                return None

    def clean_text(self,column_names:list[str] = None):
        if self.dataset is None:
            print("Dataset not found")
            return None
        try:
            pd_data = self.dataset.to_pandas()

            if column_names is None:
                for column in pd_data.columns:
                   pd_data[column_names]=pd_data[column].astype(str).str.strip()
                self.dataset = Dataset.from_pandas(pd_data, preserve_index=False)

            else:

                try:
                    for column in column_names:
                        pd_data[column] = pd_data[column].astype(str).str.strip()

                    self.dataset = Dataset.from_pandas(pd_data, preserve_index=False)

                except KeyError:
                    print(f"Error Cleaning: {column_names} not found")

                return None
        except Exception as e:
            print(f"Error Cleaning: {e}")

    def drop_null(self):
        if self.dataset is None:
            print("Dataset not found")
            return None
        try:
            #dataset has empty
            pd_data = self.dataset.to_pandas().dropna()
            self.dataset = Dataset.from_pandas(pd_data,preserve_index=False)
            return self.dataset
        except Exception as e:
            print(f"Error Drop Null: {e}")

    def drop_dupe(self,column_names:list[str]=None):

        if self.dataset is None:
            print("Dataset not found")
            return None
        try:
            # dataset contains duplicate
            pd_data = self.dataset.to_pandas()

            if column_names is None:
                pd_data = pd_data.drop_duplicates()
            else:
                pd_data = pd_data.drop_duplicates(subset=column_names)

            self.dataset = Dataset.from_pandas(pd_data,preserve_index=False)
            return self.dataset
        except Exception as e:
            print(f"Error Drop Duplicate: {e}")

    def drop_selected(self,selected:list[str] = None):

        if selected is None:
            return self.dataset
        try:
            pd_data = self.dataset.to_pandas()
            pd_data = pd_data.drop(selected,axis=1)
            self.dataset = Dataset.from_pandas(pd_data,preserve_index=False)
            return self.dataset
        except Exception as e:
            print(f"Error Drop Selected: {e}")

    def check_balance_native(self, column_name:str=None):
        # check target column balance
        if self.dataset is None:
            print("Dataset not found")
            return None
        if column_name is None:
            print("Error Checking Balance: column_name is None")
            return None

        try:
            pd_data = self.dataset.to_pandas()

            counts = Counter(pd_data[column_name])
            total = len(pd_data)

            print(f"--- Native Balance Report ---")
            for label, count in counts.items():
                percent = (count / total) * 100
                print(f"{label}: {count} ({percent:.2f}%)")
        except Exception as e:
            print(f"Error Checking Balance: {e}")

    def balance_dataset(self,selected_column_x:str,selected_column_y:str,balance_method:str='under'):

        if self.dataset is None:
            print("Dataset not found")
            return None

        if selected_column_x is None or selected_column_y is None:
            print("Error Balancing: selected_column_x or selected_column_y is None")
            return None

        pd_data = self.dataset.to_pandas()
        X = pd_data[[selected_column_x]]
        y = pd_data[selected_column_y]

        sampler = None
        match balance_method:
            case 'under':
                try:
                    sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)
                except Exception as e:
                    print(f"Error Balancing: {e}")
            case 'over':
                try:
                    sampler = RandomOverSampler(sampling_strategy='auto', random_state=42)
                except Exception as e:
                    print(f"Error Balancing: {e}")
            case _:
                print("Error Balancing: balance_method is not under or over")
                return None

        try:
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            resampled_df = pd.DataFrame(X_resampled, columns=[selected_column_x])
            resampled_df[selected_column_y] = y_resampled

            self.dataset = Dataset.from_pandas(resampled_df,preserve_index=False)
            print(f"Dataset Balanced({balance_method})")
        except Exception as e:
            print(f"Error Balancing: {e}")

    def save_dataset(self,path:str):
        if self.dataset is None:
            print("Dataset not found")
            return None
        try:
            self.dataset.save_to_disk(path)
            print(f"Dataset Saved to {path} Successfully")
        except Exception as e:
            print(f"Error Saving Dataset: {e}")

HomePath = Path(__file__).parent.parent.absolute()
save_path = HomePath / 'dataset' / 'raw_cleaned_dataset'

load_dataset = LoadDataset("EXt1/Thai-True-Fake-News").get_dataset()
process_dataset = ProcessDataset(load_dataset)
process_dataset.select_split('train')

dataset = process_dataset.get_dataset()
print(f'before cleaning: {len(dataset)}')
process_dataset.check_balance_native('Verification_Status')

process_dataset.drop_selected(['Unnamed: 0'])
process_dataset.clean_text(['Title','Verification_Status'])
process_dataset.drop_null()
process_dataset.drop_dupe(['Title'])
cleaned_dataset = process_dataset.get_dataset()
print(f'after cleaning: {len(cleaned_dataset)}')
process_dataset.check_balance_native('Verification_Status')

process_dataset.balance_dataset('Title','Verification_Status',
                                'under')
balance_dataset = process_dataset.get_dataset()
print(f'after balancing: {len(balance_dataset)}')
process_dataset.check_balance_native('Verification_Status')

process_dataset.save_dataset(save_path.as_posix())