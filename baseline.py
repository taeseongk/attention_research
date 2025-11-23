import torch

import argparse
from typing import List
from tqdm import tqdm
import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForCausalLM

from data import QADatasetLoader
from eval import QAEvaluator
from util import answer_prompt


class Baseline:
    """
    Baseline Method
    """
    def __init__(self, model_name: str):
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
    
    def generate_pred(self, question: str, context: List[str], dataset_name: str):
        prompt = answer_prompt(question, context, "baseline")
        #print(f"Prompt:\n{prompt}")
        return self._generate(prompt)

        
    def generate_all_preds(self, dataset, dataset_name: str):
        samples = []
        predictions = []
        for example in tqdm(dataset, desc=f"Predicting {dataset_name}"):
            prediction = self.generate_pred(example["question"], example["context"], dataset_name)
            print(f"\nQ: {example['question']}")
            print(f"A: {prediction}")
            samples.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"]
                }
            )
            predictions.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"],
                    "prediction": prediction,
                    "gold": example["answers"],
                }
            )
        return predictions, samples

    def _generate(
        self,
        prompt: str,
        max_new_tokens: int = 50
    ):
        messages = [{"role": "user", "content": prompt}]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated_text.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed for dataset sampling")
    parser.add_argument("--dataset", type=str, choices=["squad", "hotpotqa", "nq"], help="Dataset to use")
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    baseline = Baseline(model_name=model_name)

    if args.dataset == "squad":
        dataset = QADatasetLoader.load_squad(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "hotpotqa":
        dataset = QADatasetLoader.load_hotpotqa(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "nq":
        dataset = QADatasetLoader.load_natural_questions_mrqa(n_samples=args.n_samples, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    predictions, samples = baseline.generate_all_preds(dataset, args.dataset)
    config = {
        "dataset": args.dataset,
        "method": "baseline",
        "seed": args.seed,
        "n": args.n_samples,
    }
    QAEvaluator.save_samples(samples, config)
    _, scores = QAEvaluator.evaluate(predictions, True, config)
    print(scores)

if __name__ == "__main__":
    main()
