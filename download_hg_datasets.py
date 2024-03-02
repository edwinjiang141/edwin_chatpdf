# from datasets import load_dataset,load_from_disk,load_dataset_builder

# # dataset = load_dataset("yelp_review_full")
# # dataset.save_to_disk('D:\edwin\other\AIGC\AIsGC-Class\大模型微调\datasets')
# # dataset = load_from_disk("D:\edwin\other\AIGC\AIGC-Class\大模型微调\datasets")
# # ds_builder = load_dataset_builder("D:\edwin\other\AIGC\AIGC-Class\大模型微调\datasets")
# # print(ds_builder.info.features)
# import numpy as np
# import evaluate

# metric = evaluate.load("accuracy")

# import transformers

from transformers import pipeline

# nlp = pipeline("sentiment-analysis")

# from pathlib import Path
# import torch

# output_dir = Path('D:\edwin\other\AIGC\hugginggface\pipeline\sentiment-analysis')
# output_dir.mkdir(exist_ok=True)

# torch.save(nlp.model.state_dict(), str(output_dir / 'model.bin'))
# nlp.tokenizer.save_pretrained(str(output_dir))
# nlp.config.save_pretrained(str(output_dir))

from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification,AutoModel

# nlp = pipeline("sentiment-analysis") # 加载任何模型

# model_path = "D:\edwin\other\AIGC\hugginggface\pipeline\sentiment-analysis"
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForSequenceClassification.from_pretrained(model_path)
# nlp.model.set_state_dict(torch.load(f"{model_path}/model.bin"))

# nlp.tokenizer = tokenizer
# nlp.model.to("cpu")
# model_name ="D:\edwin\other\AIGC\hugginggface\pipeline\sentiment-analysis\model"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# print(model)
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# classifier = pipeline('sentiment-analysis',model=model,tokenizer=tokenizer)
# print(classifier("ShangHai is realy cold today"))

# pipe = pipeline("sentiment-analysis")
# pipe("今儿上海可真冷啊")


#C4 数据集
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTQConfig
import torch
from datasets import load_dataset,load_from_disk
import tqdm as notebook_tqdm

# model_id = "D:\edwin\other\AIGC\AIGC-Class\大模型微调\opt27bb"
# 

from datasets import load_dataset,load_metric
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["HF_DATASETS_CACHE"] = "D:\edwin\other\AIGC\hugginggface\huggingface\datasets"
os.environ["HF_HOME"] = "D:\edwin\other\AIGC\hugginggface\huggingface\datasets"
os.environ["HUGGINGFACE_HUB_CACHE"] = "D:\edwin\other\AIGC\hugginggface\huggingface\datasets"
os.environ["TRANSFORMERS_CACHE"] = "D:\edwin\other\AIGC\hugginggface\huggingface\datasets"

# dataset = load_dataset("mit-han-lab/pile-val-backup")
# squad_v2 = False
# dataset = load_dataset("squad_v2" if squad_v2 else "squad")
# metric = load_metric("squad")
# print(metric)
# conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

dataset = load_dataset("yelp_review_full")
dataset.save_to_disk("D:\edwin\other\AIGC\hugginggface\datasets")
# print(datasets)
# dataset.save_to_disk("D:\edwin\other\AIGC\hugginggface\pile-val-backup")
#export HF_ENDPOINT=https://hf-mirror.com
#export HUGGINGFACE_CO_RESOLVE_ENDPOINT=https://hf-mirror.com

# huggingface-cli download --resume-download --local-dir-use-symlinks False bigscience/bloom-560m --local-dir bloom-560m
# huggingface-cli download --token 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 --resume-download --local-dir-use-symlinks False openai/whisper-large-v3 --local-dir whisper-large-v3
