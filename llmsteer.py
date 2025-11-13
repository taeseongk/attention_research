import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

from dataclasses import dataclass


@dataclass
class LLMSteerConfig:
    alpha: float
    top_k: int
    prefix_prompt_1: str = """
Answer the question based on the given passages. Only give me the answer and do not output any other words. The following are given passages.
"""
    prefix_prompt_2: str = """
Respond to the query using only the provided texts. Only return the answer. DO NOT return any extra words. Below are the provided texts.
"""


class LLMSteer:
    """
    LLMSteer implementation from the paper.
    """

    def __init__(self, config: LLMSteerConfig, model_name: str):
        self.config = config
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = "cuda"

    def contextual_rereading(self, context: str):
        text_1 = f"{self.config.prefix_prompt_1}\n\n{context}"
        inputs_1 = self.tokenizer(text_1, return_tensors="pt").to(self.device)

        registered_hooks = []


def main():
    model_name = 
    config = LLMSteerConfig(alpha=2.0, top_k=50)
    llmsteer = LLMSteer(config,)
