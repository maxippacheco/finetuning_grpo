import json
import pandas as pd
from typing import List, Tuple, Dict, Any
from datasets import Dataset
import numpy as np


class FinancialQADataset:
    """
    Data loader for the financial Q&A JSON dataset.
    """
    
    def __init__(self, train_path: str, dev_path: str = None, test_path: str = None, max_context_length: int = 800):
        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.max_context_length = max_context_length
    
    def load_json_data(self, file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def truncate_context(self, context: str, max_length: int = 800) -> str:
        if len(context) <= max_length:
            return context
        sentences = context.split('. ')
        truncated = ""
        for sentence in sentences:
            if len(truncated + sentence) < max_length:
                truncated += sentence + ". "
            else:
                break
        if len(truncated) > max_length:
            truncated = context[:max_length-3] + "..."
        return truncated.strip()
    
    def process_qa_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        question = item['qa']['question']
        answer = item['qa']['answer']
        explanation = item['qa'].get('explanation', '')
        context = ' '.join(item['pre_text']) if item['pre_text'] else ''
        context = self.truncate_context(context, self.max_context_length)
        if context:
            prompt = f"Financial Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        if explanation and explanation.strip():
            target = f"{answer}. {explanation}"
        else:
            target = answer
        return {
            'prompt': prompt,
            'target': target,
            'question': question,
            'answer': answer,
            'explanation': explanation,
            'context': context,
            'id': item.get('id', ''),
            'filename': item.get('filename', '')
        }
    
    def load_training_data(self) -> List[Dict[str, Any]]:
        print(f"Loading training data from {self.train_path}...")
        raw_data = self.load_json_data(self.train_path)
        processed_data = []
        skipped_count = 0
        for item in raw_data:
            try:
                if not item['qa'].get('answer', '').strip():
                    skipped_count += 1
                    continue
                processed_item = self.process_qa_item(item)
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {e}")
                skipped_count += 1
                continue
        print(f"Loaded {len(processed_data)} training examples (skipped {skipped_count})")
        return processed_data
    
    def load_dev_data(self) -> List[Dict[str, Any]]:
        if not self.dev_path:
            return []
        print(f"Loading development data from {self.dev_path}...")
        raw_data = self.load_json_data(self.dev_path)
        processed_data = []
        skipped_count = 0
        for item in raw_data:
            try:
                if not item['qa'].get('answer', '').strip():
                    skipped_count += 1
                    continue
                processed_item = self.process_qa_item(item)
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {e}")
                skipped_count += 1
                continue
        print(f"Loaded {len(processed_data)} development examples (skipped {skipped_count})")
        return processed_data
    
    def load_test_data(self) -> List[Dict[str, Any]]:
        if not self.test_path:
            return []
        print(f"Loading test data from {self.test_path}...")
        raw_data = self.load_json_data(self.test_path)
        processed_data = []
        skipped_count = 0
        for item in raw_data:
            try:
                if not item['qa'].get('answer', '').strip():
                    skipped_count += 1
                    continue
                processed_item = self.process_qa_item(item)
                processed_data.append(processed_item)
            except Exception as e:
                print(f"Error processing item: {e}")
                skipped_count += 1
                continue
        print(f"Loaded {len(processed_data)} test examples (skipped {skipped_count})")
        return processed_data
    
    def create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        train_data = self.load_training_data()
        dev_data = self.load_dev_data()
        test_data = self.load_test_data()
        train_dataset = Dataset.from_list(train_data)
        dev_dataset = Dataset.from_list(dev_data) if dev_data else None
        test_dataset = Dataset.from_list(test_data) if test_data else None
        return train_dataset, dev_dataset, test_dataset
    
    def get_sample_data(self, num_samples: int = 5) -> List[Dict[str, Any]]:
        train_data = self.load_training_data()
        return train_data[:num_samples]


def get_financial_qa_reward():
    from data_utils import RewardFunction
    class FinancialQAReward(RewardFunction):
        def __init__(self):
            self.financial_keywords = {
                'revenue', 'profit', 'loss', 'income', 'expense', 'asset', 'liability',
                'equity', 'cash', 'investment', 'return', 'risk', 'market', 'stock',
                'bond', 'interest', 'rate', 'percentage', 'growth', 'decline', 'increase',
                'decrease', 'balance', 'statement', 'report', 'quarter', 'annual',
                'financial', 'economic', 'trading', 'portfolio', 'dividend', 'earnings',
                'million', 'billion', 'dollar', 'currency', 'exchange', 'hedge',
                'derivative', 'contract', 'obligation', 'credit', 'debt', 'loan',
                'payment', 'transaction', 'volume', 'average', 'amount', 'total'
            }
            self.analysis_keywords = {
                'because', 'therefore', 'however', 'although', 'while', 'since',
                'due to', 'as a result', 'consequently', 'furthermore', 'additionally',
                'in contrast', 'on the other hand', 'specifically', 'in particular',
                'according to', 'based on', 'indicates', 'shows', 'demonstrates'
            }
            self.numerical_patterns = [
                r'\$\s*\d+\.?\d*',
                r'\d+\.?\d*\s*%',
                r'\d+\.?\d*\s*million',
                r'\d+\.?\d*\s*billion',
            ]
        def __call__(self, prompt: str, completion: str, target: str = None) -> float:
            import re
            completion_lower = completion.lower()
            words = set(completion_lower.split())
            reward = 0.0
            if '0000' in completion or completion.count('0') > len(completion) * 0.3:
                return -1.0
            financial_matches = len(words.intersection(self.financial_keywords))
            reward += min(1.0, financial_matches * 0.2)
            analysis_matches = len(words.intersection(self.analysis_keywords))
            reward += min(1.0, analysis_matches * 0.3)
            length = len(completion.split())
            if 5 <= length <= 50:
                reward += 0.5
            elif length > 50:
                reward += 0.3
            else:
                reward -= 0.3
            numbers = len(re.findall(r'\d+', completion))
            reward += min(0.5, numbers * 0.1)
            currency_matches = len(re.findall(r'\$', completion))
            percentage_matches = len(re.findall(r'%', completion))
            reward += min(0.3, (currency_matches + percentage_matches) * 0.1)
            if length < 3:
                reward -= 0.5
            words_list = completion_lower.split()
            if len(words_list) > 3:
                unique_words = len(set(words_list))
                repetition_ratio = unique_words / len(words_list)
                if repetition_ratio < 0.5:
                    reward -= 0.5
            if target:
                target_lower = target.lower()
                common_words = len(set(completion_lower.split()) & set(target_lower.split()))
                total_words = len(set(completion_lower.split()) | set(target_lower.split()))
                if total_words > 0:
                    similarity = common_words / total_words
                    reward += similarity * 0.5
                if target.lower().strip() in completion_lower:
                    reward += 0.5
            return max(-1.0, min(1.0, reward))
    return FinancialQAReward()


if __name__ == "__main__":
    # Test the data loader
    loader = FinancialQADataset(
        train_path="data/train.json",
        dev_path="data/dev.json",
        test_path="data/test.json",
        max_context_length=800  # Much longer context for better understanding
    )
    
    # Get sample data
    samples = loader.get_sample_data(3)
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Prompt length: {len(sample['prompt'])} characters")
        print(f"Prompt: {sample['prompt'][:300]}...")
        print(f"Target: {sample['target']}")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print("-" * 50) 