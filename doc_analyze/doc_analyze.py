
from langchain.llms import OpenAI
from utils import LOG
from langchain.chains.summarize import load_summarize_chain
import os
class DocAnalyze:
    def __init__(self, model_name, verbose: bool = False):
        llm =  OpenAI(model_name=model_name, max_tokens=1500,openai_api_key = os.environ["OPENAI_API_KEY"],openai_api_base = "https://pro.aiskt.com/v1")
        self.chain = load_summarize_chain(llm, chain_type="refine", verbose=True)

    def run(self,doc) -> (list):
        result = ""
        try:
            result = self.chain.run(doc)
        except Exception as e:
            LOG.error(f"An error occurred during translation: {e}")
            return result
        
        return result