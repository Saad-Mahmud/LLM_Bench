from LLM_agents.base_llm_agent import BaseLLMAgent
from GSM_agents.base_gsm_agent import BaseGSMAgent
from LLM_agents.cot_llm_agent import COTLLMAgent


if __name__ == "__main__":
    #llm_agent = BaseLLMAgent(name = "Qwen2.5_32B_INS_Base", n_sample=5, selection="majority")
    llm_agent = COTLLMAgent(name = "LLama 3.3 70B", n_sample=3, selection="majority")
    benchmark_agent = BaseGSMAgent()
    benchmark_agent.test(llm_agent, 200)
