import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import argparse
import json
from pathlib import Path
from typing import List

from tqdm import tqdm
from data import QADatasetLoader
from eval import QAEvaluator

from pastalib.pasta import PASTA


class AutoPASTA(PASTA):
    """
    AutoPASTA: Automatic Post-hoc Attention Steering Approach
    """

    def __init__(
        self,
        model_name,
        head_config: dict | list | None = None,
        alpha: float = 0.01,
        scale_position: str = "exclude",
        encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.pad_token = tokenizer.eos_token
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            head_config=head_config,
            alpha=alpha,
            scale_position=scale_position,
        )
        self.encoder = SentenceTransformer(encoder_model)

    def generate_key_sentence(
        self,
        question: str,
        context: str,
        max_new_tokens: int = 100,
        temperature: float = 0.0,
    ):
        """
        Step 1 of AutoPASTA: Generate key sentence using LLM.

        Prompts the model to identify which sentence in the context
        is most important for answering the question.
        """
        prompt = self._key_sentence_prompt(question, context)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return generated_text.strip()

    def match_to_context(
        self,
        key_sentence: str,
        context: str,
    ):
        """
        Step 2 of AutoPASTA: Match generated sentence back to original context.

        Uses semantic similarity (cosine similarity of embeddings) to find
        the best matching sentence from the original context.
        """
        context_sentences = self._split_into_sentences(context)
        key_sentence_embd = self.encoder.encode(
            key_sentence, convert_to_tensor=True, device=self.encoder.device
        )
        context_embd = self.encoder.encode(
            context_sentences, convert_to_tensor=True, device=self.encoder.device
        )

        # Compute cosine similarity
        similarities = torch.nn.functional.cosine_similarity(
            key_sentence_embd.unsqueeze(0), context_embd, dim=1
        )

        best_idx = similarities.argmax().item()
        best_sentence = context_sentences[best_idx]

        return best_sentence, best_idx

    def answer_with_steering(
        self,
        question: str,
        context: List[str],
        key_sentence: List[str],
        max_new_tokens: int = 50,
        temperature: float = 0.0,
    ) -> str:
        """
        Step 3 of AutoPASTA: Answer question with attention steering.

        Uses PASTA's attention steering mechanism to highlight the key sentence
        while generating the answer.
        """
        prompt = self._answer_prompt(question, context)

        inputs, offset_mapping = self.inputs_from_batch(
            text=[prompt], tokenizer=self.tokenizer, device="cuda"
        )
        with self.apply_steering(
            model=self.model,
            strings=[prompt],
            substrings=key_sentence,
            model_input=inputs,
            offsets_mapping=offset_mapping,
        ) as steered_model:
            with torch.no_grad():
                outputs = steered_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else 1.0,
                    pad_token_id=self.tokenizer.pad_token_id
                    or self.tokenizer.eos_token_id,
                )

        answer = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )
        return answer.strip()

    def answer_question(self, question: str, context: List[str], dataset_name: str, max_new_tokens: int = 50):
        """
        Complete AutoPASTA pipeline: Identify key sentence and answer with steering.
        """

        if dataset_name == "squad":
            # Step 1: Generate key sentence
            key_sentence = self.generate_key_sentence(question, context[0])
            # print(f"Key Sentence: {key_sentence}")

            # Step 2: Match to original context
            matched_sentence, _ = self.match_to_context(key_sentence, context[0])

            # Step 3: Answer with steering
            answer = self.answer_with_steering(
                question, context, [matched_sentence], max_new_tokens
            )
            return answer, [matched_sentence]
        elif dataset_name == "hotpotqa":
            key_sentences = []
            for ctxt in context:
                key_sentences.append(self.generate_key_sentence(question, ctxt))
            matched_sentences = []
            for i in range(len(context)):
                matched_sentence, _ = self.match_to_context(key_sentences[i], context[i])
                matched_sentences.append(matched_sentence)
            #print(matched_sentences)
            answer = self.answer_with_steering(
                question, context, matched_sentences, max_new_tokens
            )
            return answer, [matched_sentences]


    def generate_all_preds(self, dataset, dataset_name: str):
        """Generate all predictions for the dataset"""
        
        samples = []
        predictions = []
        for example in tqdm(dataset, desc=f"Predicting {dataset_name}"):
            prediction, matched_sentences = self.answer_question(example["question"], example["context"], dataset_name)
            print(f"Question: {example['question']}\n")
            print(f"Context: {example['context']}\n")
            print(f"Prediction: {prediction}\n")
            samples.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"],
                    "key_sentence": matched_sentences
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

    def _split_into_sentences(self, context: str) -> List[str]:
        """
        Split text into sentences.
        """
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", context)
        # Filter empty strings and strip whitespace
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def _key_sentence_prompt(self, question: str, context: str) -> str:
        """Build prompt for key sentence identification"""
        return f"""A question and a passage are shown below. Please select the key sentence in the passage that supports to answer the question correctly. Only output the exactly same sentence from the passage without other additional words.

Question: {question}

Passage: {context}

Sentence:"""

    def _answer_prompt(self, question: str, context: List[str]) -> str:
        """Build prompt for direct answer generation."""
        prompt = f"""Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words.
        
Context: {" ".join(context)}
        
Question: {question}
        
Answer:"""
        return prompt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed for dataset sampling")
    parser.add_argument("--dataset", type=str, default="squad", choices=["squad", "hotpotqa"], help="Dataset to use")
    args = parser.parse_args()

    head_config = {
        26: [0, 8, 16, 24],
        27: [4, 12, 20, 28],
        28: [2, 10, 18, 26],
        29: [6, 14, 22, 30],
        30: [1, 9, 17, 25],
        31: [3, 11, 19, 27],
    } 
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    autopasta = AutoPASTA(
        model_name=model_name,
        head_config=head_config,
        alpha=0.01,
    )

    dataset = ""
    if args.dataset == "squad":
        dataset = QADatasetLoader.load_squad(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "hotpotqa":
        dataset = QADatasetLoader.load_hotpotqa(n_samples=args.n_samples)

    predictions, samples = autopasta.generate_all_preds(dataset, args.dataset)
    path = Path(f"samples/{args.dataset}/autopasta_{args.n_samples}_{args.seed}.json")
    with open(path, "w") as f:
        json.dump({"samples": samples}, f, indent=2)
    results, scores = QAEvaluator.evaluate(predictions, f"results/{args.dataset}/autopasta.json")
    print(results)
    print(scores)


if __name__ == "__main__":
    main()
