# 在线语音实时知识检索
## 采用 whisper-large-v2 大模型 + Common Voice 11.0数据集进基于 PEFT+LORA进行模型微调，增强对中文的理解
## 采用 LangChain+OpenAI API+FASIS向量数据库，构建基于本地的知识库
## 在模型加载时，采用BitsAndBytes技术，加载后的微调模型占用4.5GB左右的GPU显存

