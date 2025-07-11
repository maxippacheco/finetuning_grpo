#!/usr/bin/env python3
"""
GRPO Training Script for Financial Q&A JSON Dataset - Optimized Version
"""

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import os
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import time

from grpo import GRPOTrainer
from data_loader import FinancialQADataset, get_financial_qa_reward
from training_visualization import plot_training_curves, plot_reward_distribution, create_training_report


class FinancialQATrainer:
    """
    Trainer for GRPO using JSON financial Q&A dataset - Optimized for speed.
    """
    
    def __init__(
        self,
        model_str: str = 'meta-llama/Meta-Llama-3-8B-Instruct',
        train_path: str = 'data/train.json',
        dev_path: str = 'data/dev.json',
        test_path: str = 'data/test.json',
        save_path: str = 'grpo_financial_model',
        max_context_length: int = 800
    ):
        print("[INFO] Initializing data loader...")
        self.model_str = model_str
        self.save_path = save_path
        self.data_loader = FinancialQADataset(
            train_path, dev_path, test_path, 
            max_context_length=max_context_length
        )
        print("[INFO] Initializing GRPO trainer and model...")
        self.trainer = GRPOTrainer(model_str)
        print("[INFO] Initializing reward function...")
        self.reward_fn = get_financial_qa_reward()
        self.training_history = {
            'losses': [],
            'rewards': [],
            'completions': [],
            'targets': []
        }
    def generate_completion_fast(self, prompt: str, max_tokens: int = 30) -> str:
        try:
            completion = self.trainer.generate_completion(prompt, max_new_tokens=max_tokens)
            if '0000' in completion or completion.count('0') > len(completion) * 0.3:
                completion = self.trainer.generate_completion(prompt, max_new_tokens=max_tokens)
            return completion
        except Exception as e:
            return "Unable to generate response."
    def train(
        self,
        num_epochs: int = 1,
        batch_size: int = 8,
        learning_rate: float = 1e-4,
        eval_interval: int = 50,
        save_interval: int = 100,
        generate_frequency: float = 0.5
    ):
        print("[INFO] Loading training data...")
        train_data = self.data_loader.load_training_data()
        dev_data = self.data_loader.load_dev_data()
        print(f"[INFO] Training examples: {len(train_data)}")
        print(f"[INFO] Development examples: {len(dev_data)}")
        optimizer = optim.AdamW(self.trainer.get_trainable_parameters(), lr=learning_rate)
        print("[INFO] Starting optimized training...")
        self.trainer.model.train()
        global_step = 0
        for epoch in range(num_epochs):
            print(f"\n[INFO] Epoch {epoch + 1}/{num_epochs}")
            np.random.shuffle(train_data)
            epoch_losses = []
            epoch_rewards = []
            num_batches = (len(train_data) + batch_size - 1) // batch_size
            batch_iter = tqdm(range(0, len(train_data), batch_size), 
                             desc=f"Epoch {epoch+1} Batches", ncols=100)
            start_time = time.time()
            for i in batch_iter:
                batch = train_data[i:i + batch_size]
                batch_loss = 0.0
                batch_rewards = []
                batch_completions = []
                batch_targets = []
                for j, example in enumerate(batch):
                    prompt = example['prompt']
                    target = example['target']
                    if np.random.random() < generate_frequency:
                        completion = self.generate_completion_fast(prompt, max_tokens=30)
                        reward = self.reward_fn(prompt, completion, target)
                    else:
                        completion = target
                        reward = 1.0
                    loss = self.trainer.compute_loss(prompt, completion, reward)
                    batch_loss += loss.item()
                    batch_rewards.append(reward)
                    batch_completions.append(completion)
                    batch_targets.append(target)
                avg_loss = batch_loss / len(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                self.training_history['losses'].append(avg_loss)
                self.training_history['rewards'].extend(batch_rewards)
                self.training_history['completions'].extend(batch_completions)
                self.training_history['targets'].extend(batch_targets)
                epoch_losses.append(avg_loss)
                epoch_rewards.extend(batch_rewards)
                global_step += 1
                batch_iter.set_postfix({
                    "Loss": f"{avg_loss:.4f}", 
                    "AvgReward": f"{np.mean(batch_rewards):.3f}",
                    "GenFreq": f"{generate_frequency:.1f}"
                })
                if global_step % eval_interval == 0:
                    print(f"\n[INFO] Evaluating on dev set (step {global_step})...")
                    self.evaluate_fast(dev_data[:5])
                if global_step % save_interval == 0:
                    self.save_model(f"{self.save_path}_step_{global_step}")
            elapsed = time.time() - start_time
            print(f"[INFO] Epoch {epoch + 1} completed. Avg Loss: {np.mean(epoch_losses):.4f}, "
                  f"Avg Reward: {np.mean(epoch_rewards):.3f}, Time: {elapsed:.1f}s")
        self.save_model(self.save_path)
        print("[INFO] Training completed!")
    def evaluate_fast(self, eval_data: List[Dict[str, Any]]):
        print("[INFO] Running fast evaluation...")
        self.trainer.model.eval()
        prompts = []
        completions = []
        targets = []
        rewards = []
        with torch.no_grad():
            for example in eval_data:
                prompt = example['prompt']
                target = example['target']
                completion = self.generate_completion_fast(prompt, max_tokens=30)
                reward = self.reward_fn(prompt, completion, target)
                prompts.append(prompt)
                completions.append(completion)
                targets.append(target)
                rewards.append(reward)
        if len(prompts) > 0:
            print(f"\n[INFO] Sample prediction:")
            print(f"Prompt: {prompts[0][:150]}...")
            print(f"Target: {targets[0]}")
            print(f"Generated: {completions[0]}")
            print(f"Reward: {rewards[0]:.3f}")
        avg_reward = np.mean(rewards)
        print(f"[INFO] Average reward: {avg_reward:.3f}")
        self.trainer.model.train()
    def save_model(self, path: str):
        print(f"[INFO] Saving model to {path}...")
        self.trainer.model.save_pretrained(path)
        self.trainer.tokenizer.save_pretrained(path)
        history_path = f"{path}_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    def load_model(self, path: str):
        print(f"[INFO] Loading model from {path}...")
        self.trainer.model = self.trainer.model.from_pretrained(path)
        history_path = f"{path}_history.json"
        if os.path.exists(history_path):
            with open(history_path, 'r') as f:
                self.training_history = json.load(f)
    def generate_sample(self, prompt: str, max_tokens: int = 30) -> str:
        return self.generate_completion_fast(prompt, max_tokens=max_tokens)
    def create_training_plots(self, save_dir: str = "training_plots"):
        os.makedirs(save_dir, exist_ok=True)
        plot_training_curves(
            self.training_history['losses'],
            self.training_history['rewards'],
            f"{save_dir}/training_curves.png"
        )
        plot_reward_distribution(
            self.training_history['rewards'],
            f"{save_dir}/reward_distribution.png"
        )
        create_training_report(
            self.training_history['losses'],
            self.training_history['rewards'],
            self.training_history['completions'],
            self.training_history['targets'],
            save_dir
        )
        print(f"[INFO] Training plots saved to {save_dir}/")

def evaluate_model_on_questions(model_path: str, test_questions: List[str]):
    print("[INFO] Loading trained model...")
    trainer = FinancialQATrainer()
    trainer.load_model(model_path)
    print("\n[INFO] Testing model on new questions:")
    for question in test_questions:
        prompt = f"Question: {question}\n\nAnswer:"
        completion = trainer.generate_sample(prompt, max_tokens=30)
        print(f"\nQ: {question}")
        print(f"A: {completion}")

def main():
    print("[INFO] GRPO Financial Q&A Training with JSON Data - OPTIMIZED")
    print("=" * 70)
    data_files = ["data/train.json", "data/dev.json", "data/test.json"]
    missing_files = [f for f in data_files if not os.path.exists(f)]
    if missing_files:
        print(f"[ERROR] Missing data files: {missing_files}")
        print("Please ensure the JSON data files are in the data/ directory")
        return
    trainer = FinancialQATrainer(
        model_str='meta-llama/Meta-Llama-3-8B-Instruct',
        train_path='data/train.json',
        dev_path='data/dev.json',
        test_path='data/test.json',
        save_path='grpo_financial_model',
        max_context_length=800
    )
    trainer.train(
        num_epochs=1,
        batch_size=8,
        learning_rate=1e-4,
        eval_interval=50,
        save_interval=100,
        generate_frequency=0.5
    )
    trainer.create_training_plots()
    test_questions = [
        "What is the interest expense in 2009?",
        "How much is the annual interest expense?",
        "What is the foreign currency exposure?",
        "What are the derivative instruments?",
        "What is the fair value of forward exchange contracts?"
    ]
    evaluate_model_on_questions('grpo_financial_model', test_questions)

if __name__ == "__main__":
    main() 