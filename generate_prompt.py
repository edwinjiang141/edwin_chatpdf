from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from langchain.schema import HumanMessage,SystemMessage,AIMessage

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
import os
from utils import LOG
# gpt-3.5-turbo  gpt-4
# class AiTranslate:
#     def __init__(self, model_name: str = "gpt-3.5-turbo", verbose: bool = True):
        
#         # 任务指令始终由 System 角色承担
#         template = (
#             """In order to ask questions more effectively, it is necessary to process the questions raised by the customer, 
#             so that the AI system can better understand the problem, please follow the following example for transformation: 
#             Please give all the relevant reasons, scenarios, solutions and verification schemes for the problem of password failure.
#             """
#         )
#         system_message_prompt = SystemMessagePromptTemplate.from_template(template)

#         # 待解决的问题由 Human 角色输入
#         human_template = "the problem is {question}"
#         human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

#         # 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
#         chat_prompt_template = ChatPromptTemplate.from_messages(
#             [system_message_prompt, human_message_prompt]
#         )

#         print(chat_prompt_template)

#         # 为了翻译结果的稳定性，将 temperature 设置为 0
#         chat = ChatOpenAI(model_name=model_name, temperature=0, verbose=verbose,openai_api_key = os.environ["OPENAI_API_KEY"],openai_api_base = "https://pro.aiskt.com/v1")

#         self.chain = LLMChain(llm=chat, prompt=chat_prompt_template, verbose=verbose)

#     def run(self, text: str, source_language: str, target_language: str) -> (str, bool):
#         result = ""
#         try:
#             result = self.chain.run({
#                 "text": text,
#                 "source_language": source_language,
#                 "target_language": target_language,
#             })
#         except Exception as e:
#             LOG.error(f"An error occurred during translation: {e}")
#             return result, False

#         return result, True

# 任务指令始终由 System 角色承担
template = (
    """In order to ask questions more effectively, it is necessary to process the questions raised by the customer, 
    so that the AI system can better understand the problem, please follow the following example for transformation: 
    Please give all the relevant reasons, scenarios, solutions and verification schemes for the problem of password failure.
    """
)
system_message_prompt = SystemMessagePromptTemplate.from_template(template)

# 待解决的问题由 Human 角色输入
human_template = "the problem is {question}"
human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

# 使用 System 和 Human 角色的提示模板构造 ChatPromptTemplate
chat_prompt_template = ChatPromptTemplate.from_messages(
    [system_message_prompt, human_message_prompt]
)

print(chat_prompt_template)