import torch
import json
from tqdm import tqdm
from typing import List, Tuple, Dict
import argparse
from pathlib import Path

from data import QADatasetLoader
from eval import QAEvaluator

from autopasta import AutoPASTA
from llmsteer_3 import LLMSteer
from llmsteer_3 import LLMSteerConfig

class Profiler:
    """
    Coarse-to-Fine Profiling from the AutoPASTA paper
    """
    def __init__(
        self,
        method_name,
        method,
        dataset,
        dataset_name,
        evaluator,
        top_l_layers: int = 6,
        top_k_heads: int = 4,
    ):
        self.method_name = method_name
        self.method = method
        self.dataset = dataset
        self.dataset_name = dataset_name
        self.evaluator = evaluator
        self.top_l_layers = top_l_layers
        self.top_k_heads = top_k_heads
        self.num_layers = method.model.config.num_hidden_layers
        self.num_heads = method.model.config.num_attention_heads

    def coarse_profiling(self) -> List[Tuple[int, float]]:
        layer_scores = {}
        for layer_id in tqdm(range(self.num_layers), desc="Coarse profiling"):
            head_config = {layer_id: list(range(self.num_heads))}
            self.method.head_config = head_config
            if self.method_name == "autopasta":
                self.method.setup_head_config(head_config)
            predictions, _ = self.method.generate_all_preds(
                self.dataset, self.dataset_name 
            )
            _, scores = self.evaluator.evaluate(predictions, False, {})
            token_f1 = scores["aggr_f1_score"]
            layer_scores[layer_id] = token_f1
            print(f"\nLayer {layer_id}: F1 = {token_f1:.2f}\n")
        layer_scores = [(layer_id, score) for layer_id, score in layer_scores.items()]
        layer_scores.sort(key=lambda x: x[1], reverse=True)
        return layer_scores[:self.top_l_layers]

    def fine_profiling(self, top_layers: List[Tuple[int, float]]) -> Dict[int, List[int]]:
        head_scores = {}
        for layer_id, _ in top_layers:
            for head_id in tqdm(range(self.num_heads), desc=f"Layer {layer_id}"):
                head_config = {layer_id: [head_id]}
                self.method.head_config = head_config
                if self.method_name == "autopasta":
                    self.method.setup_head_config(head_config)
                predictions, _ = self.method.generate_all_preds(
                    self.dataset, self.dataset_name
                )
                _, scores = self.evaluator.evaluate(predictions, False, {})
                token_f1 = scores["aggr_f1_score"]
                head_scores[(layer_id, head_id)] = token_f1
                print(f"Layer {layer_id} Head {head_id}: F1 = {token_f1:.2f}")
        head_config = {}
        for layer_id, _ in top_layers:
            layer_heads = [(head_id, head_scores[(layer_id, head_id)]) for head_id in range(self.num_heads)]
            layer_heads.sort(key=lambda x: x[1], reverse=True)
            top_heads = [head_id for head_id, _ in layer_heads[:self.top_k_heads]]
            head_config[layer_id] = top_heads
        return head_config

    def coarse_to_file_profiling(self):
        top_layers = self.coarse_profiling()
        head_config = self.fine_profiling(top_layers)
        return head_config

    def save_config(self, head_config, config):
        output_path = f"config/{config['method']}.json"
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        save_data = {"head_config": head_config}
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=None, help="Number of samples")
    parser.add_argument("--seed", type=int, default=None, help="Random Seed for dataset sampling")
    parser.add_argument("--dataset", type=str, choices=["squad", "hotpotqa", "nq"], help="Dataset to use")
    args = parser.parse_args()
    

    if args.dataset == "squad":
        dataset = QADatasetLoader.load_squad(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "hotpotqa":
        dataset = QADatasetLoader.load_hotpotqa(n_samples=args.n_samples, seed=args.seed)
    elif args.dataset == "nq":
        dataset = QADatasetLoader.load_natural_questions_mrqa(n_samples=args.n_samples, seed=args.seed)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
    config = LLMSteerConfig(alpha=3.0, top_k=30, filter_special_tokens=True)
    llmsteer = LLMSteer(
        config=config,
        head_config={},
        model_name=model_name,
    )
    autopasta = AutoPASTA(
        model_name=model_name,
        head_config={}
    )      

    evaluator = QAEvaluator()
    profiler = Profiler(
        method_name="llmsteer",
        method=llmsteer,
        dataset=dataset,
        dataset_name=args.dataset,
        evaluator=evaluator
    )
    config = {
        "dataset": args.dataset,
        "method": "llmsteer",
        "n": args.n_samples,
        "alpha": 3.0,
        "top_k": 30
    }
    head_config = profiler.coarse_to_file_profiling()
    print(head_config)
    profiler.save_config(head_config, config)

if __name__ == "__main__":
    main()
