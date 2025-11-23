import torch

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import argparse
import json
from pathlib import Path
from typing import List
from tqdm import tqdm

from data import QADatasetLoader
from eval import QAEvaluator
from util import key_sentence_prompt, split_into_sentences, answer_prompt

class IterPrompt:
    """
    Iterative Prompting Method from AutoPASTA Paper
    """
    def __init__(self, model_name, encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
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
        self.encoder = SentenceTransformer(encoder_model)

    def generate_key_sentence(
        self, question: str, context: str, max_new_tokens: int = 100
    ):
        prompt = key_sentence_prompt(question, context)
        return self._generate(prompt, max_new_tokens)

    def match_to_context(self, key_sentence: str, context: str):
        context_sentences = split_into_sentences(context) 
        key_sentence_embd = self.encoder.encode(
            key_sentence, convert_to_tensor=True, device=self.device
        )
        context_embd = self.encoder.encode(
            context_sentences, convert_to_tensor=True, device=self.device
        )

        similarities = torch.nn.functional.cosine_similarity(
            key_sentence_embd.unsqueeze(0), context_embd, dim=1
        )

        best_idx = similarities.argmax().item()
        best_sentence = context_sentences[best_idx]

        return best_sentence, best_idx

    def generate_pred(
        self,
        question: str,
        context: List[str],
        dataset_name: str
    ):
        if dataset_name == "squad" or dataset_name == "nq":
            key_sentence = self.generate_key_sentence(question, context[0])
            matched_sentence, _ = self.match_to_context(key_sentence, context[0])
            prompt = answer_prompt(question, context, "iter_prompt", [matched_sentence])
            print(f"Prompt:\n{prompt}")
            answer = self._generate(prompt)
            return answer, [key_sentence], [matched_sentence]

        elif dataset_name == "hotpotqa":
            key_sentences= []
            for ctxt in context:
                key_sentences.append(self.generate_key_sentence(question, ctxt))
            matched_sentences = []
            for i in range(len(context)):
                matched_sentence, _ = self.match_to_context(key_sentences[i], context[i])
                matched_sentences.append(matched_sentence)
            prompt = answer_prompt(question, context, "iter_prompt", matched_sentences)
            print(f"Prompt:\n{prompt}")
            answer = self._generate(prompt)
            return answer, key_sentences, matched_sentences
        return "", [], []

    def generate_all_preds(self, dataset, dataset_name: str):
        samples = []
        predictions = []
        for example in tqdm(dataset, desc=f"Predicting {dataset_name}"):
            prediction, key_sentences, matched_sentences = self.generate_pred(example["question"], example["context"], dataset_name)
            print(f"{prediction}\n")
            samples.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"],
                    "key_sentence": key_sentences,
                    "matched_sentence": matched_sentences
                }
            )
            predictions.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"],
                    "prediction": prediction,
                    "gold": example["answers"]
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
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed for dataset sampling")
    parser.add_argument("--dataset", type=str, default="squad", choices=["squad", "hotpotqa", "nq"], help="Dataset to use")
    args = parser.parse_args()

    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    iter_prompt = IterPrompt(
        model_name=model_name
    )

    if args.dataset == "squad":
        dataset = QADatasetLoader.load_squad(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "hotpotqa":
        dataset = QADatasetLoader.load_hotpotqa(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "nq":
        dataset = QADatasetLoader.load_natural_questions_mrqa(n_samples=args.n_samples, seed=args.seed)
    else:
        dataset = {}

    predictions, samples = iter_prompt.generate_all_preds(dataset, args.dataset)

    config = {
        "dataset": args.dataset,
        "method": "iter_prompt",
        "seed": args.seed,
        "n_samples": args.n_samples,
    }
    QAEvaluator.save_samples(samples, config)
    _, scores = QAEvaluator.evaluate(predictions, True, config)
    print(scores)

if __name__ == "__main__":
    main()
