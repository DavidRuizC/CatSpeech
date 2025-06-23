import os
from huggingface_hub import login
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("HUGGINGFACE_API_KEY")

login(key)

dataset = load_dataset("mozilla-foundation/common_voice_16_0", "ca")

dataset['train'].to_parquet("./data/cv_ca_train.parquet")
dataset['validation'].to_parquet("./data/cv_ca_validation.parquet")
dataset['test'].to_parquet("./data/cv_ca_test.parquet")
