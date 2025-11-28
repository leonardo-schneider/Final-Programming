#%%
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd
from dotenv import load_dotenv
import os


load_dotenv()



try:
    df = kagglehub.dataset_load(
        KaggleDatasetAdapter.PANDAS,
        "miroslavsabo/young-people-survey",
        "responses.csv", 
    )


    print("\n--- SHAPE ---")
    print(f"Linhas: {df.shape[0]}")
    print(f"Colunas: {df.shape[1]}")

except Exception as e:
    print(f"\nErro: {e}")

#%%
df
#%%
