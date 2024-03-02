### 在线语音实时知识检索
#### 1、采用 whisper-large-v2 大模型 + Common Voice 11.0数据集进基于 PEFT+LORA进行模型微调，增强对中文的理解
#### 2、采用 LangChain+OpenAI API+FASIS向量数据库，构建基于本地的知识库
#### 3、在模型加载时，采用BitsAndBytes技术，加载后的微调模型占用4.5GB左右的GPU显存
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/8363a68b-3cdc-459a-950d-ce7f9a045837)
#### 4、演示地址：https://47.100.203.133:7860/
#### 5、演示例子：
#### 1）、问题：请给出表空间扩容的详细步骤包括前面的检查以及各种类型表空间扩容的命令
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/af9c8202-53b7-4143-96c5-18c47eebff03)
#### 文档内容：
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/895cb32b-9a29-41b4-95f1-78b40d3175c1)
#### 语音检索结果：
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/9e516b27-9828-4509-8854-7f6a1daea12b)

#### 2）、请给出归档日满了以后应该如何进行处理和检查
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/3225ab8a-9120-42e6-990a-62ed66dfce36)
#### 文档内容
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/98536d87-e8b0-4582-aa77-400a24a77b5e)
#### 语音检索结果：
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/328f56e7-0661-45f0-bb0f-e692e34ba81d)

#### 3）、数据库用户的密码周期如何改为无限制的？
#### 语音检索内容：
![图片](https://github.com/edwinjiang141/edwin_chatpdf/assets/152252397/c7000234-95fd-4e04-a6f2-43d583f78d30)








