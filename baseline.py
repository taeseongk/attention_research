import torch
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
from eval.data import QADatasetLoader
from eval.eval import QAEvaluator
from tqdm import tqdm


class Baseline:
    def __init__(self, model_name: str, device: str = "cuda"):
        print(f"Loading model: {model_name}")
        self.model_name = model_name
        self.device = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            attn_implementation="eager",
            torch_dtype=torch.float16,
            device_map="cuda:0",
            low_cpu_mem_usage=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        print(f"Model loaded on {device}")

    def create_prompt(self, question: str, context: str, dataset_name: str):
        """
        Create prompt for the model
        """
        prompt = ""
        if dataset_name == "squad":
            prompt = f"""Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words.

Context: {context}

Question: {question}

Answer:"""
        elif dataset_name == "hotpot_qa":
            prompt = f"""Answer the question below, paired with a context that provides background knowledge. Only output the answer without other context words.
        
Context: {context}
        
Question: {question}
        
Answer:"""
        return prompt

    def generate_pred(self, prompt: str, max_new_tokens: int = 50):
        """Generate prediction from model"""
        inputs = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=2048
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                pad_token_id=self.tokenizer.pad_token_id,
            )

        # Decode only the new tokens
        generated_text = self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

        return generated_text.strip()

    def generate_all_preds(self, dataset, dataset_name: str):
        """Generate all predictions for the dataset"""

        predictions = []
        for example in tqdm(dataset, desc=f"Predicting {dataset_name}"):
            prompt = self.create_prompt(
                example["question"], example["context"], dataset_name
            )
            prediction = self.generate_pred(prompt)
            print(f"Question: {example['question']}\n")
            print(f"Context: {example['context']}\n")
            print(f"Prediction: {prediction}\n")
            predictions.append(
                {
                    "id": example["id"],
                    "question": example["question"],
                    "context": example["context"],
                    "prediction": prediction,
                    "gold": example["answers"],
                }
            )
        return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed for dataset sampling")
    parser.add_argument("--dataset", type=str, default="squad", choices=["squad", "hotpotqa"], help="Dataset to use")
    args = parser.parse_args()
    # model_name = "huggyllama/llama-7b"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    # model_name = "lmsys/vicuna-7b-v1.5"
    predictor = Baseline(model_name=model_name)

    dataset = ""
    if args.dataset == "squad":
        dataset = QADatasetLoader.load_squad(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "hotpotqa":
        dataset = QADatasetLoader.load_hotpotqa(n_samples=args.n_samples)

    predictions = predictor.generate_all_preds(dataset, args.dataset)
    results, scores = QAEvaluator.evaluate(predictions, "results/baseline.json")
    print(results)
    print(scores)


if __name__ == "__main__":
    main()
