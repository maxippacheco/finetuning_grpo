import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Any
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import re


class RewardFunction:
    """
    Base class for reward functions.
    """
    
    def __call__(self, prompt: str, completion: str, target: str = None) -> float:
        """
        Compute reward for a prompt-completion pair.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            target: Target completion (if available)
            
        Returns:
            Reward value
        """
        raise NotImplementedError


class SentimentReward(RewardFunction):
    """
    Reward function based on sentiment analysis accuracy.
    """
    
    def __init__(self, positive_words: List[str] = None, negative_words: List[str] = None):
        """
        Initialize sentiment reward function.
        
        Args:
            positive_words: List of positive words
            negative_words: List of negative words
        """
        if positive_words is None:
            positive_words = [
                "good", "great", "excellent", "amazing", "wonderful", "fantastic",
                "love", "like", "enjoy", "happy", "positive", "best", "awesome"
            ]
        
        if negative_words is None:
            negative_words = [
                "bad", "terrible", "awful", "horrible", "worst", "hate", "dislike",
                "negative", "poor", "disappointing", "frustrated", "angry", "sad"
            ]
        
        self.positive_words = set(positive_words)
        self.negative_words = set(negative_words)
    
    def __call__(self, prompt: str, completion: str, target: str = None) -> float:
        """
        Compute sentiment-based reward.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            target: Target sentiment (positive/negative)
            
        Returns:
            Reward value (1.0 for correct sentiment, -1.0 for incorrect)
        """
        if target is None:
            return 0.0
        
        # Simple word-based sentiment analysis
        words = completion.lower().split()
        positive_count = sum(1 for word in words if word in self.positive_words)
        negative_count = sum(1 for word in words if word in self.negative_words)
        
        if positive_count > negative_count:
            predicted_sentiment = "positive"
        elif negative_count > positive_count:
            predicted_sentiment = "negative"
        else:
            predicted_sentiment = "neutral"
        
        if predicted_sentiment == target.lower():
            return 1.0
        else:
            return -1.0


class LengthReward(RewardFunction):
    """
    Reward function based on completion length.
    """
    
    def __init__(self, target_length: int = 10, tolerance: int = 2):
        """
        Initialize length reward function.
        
        Args:
            target_length: Target completion length
            tolerance: Acceptable deviation from target length
        """
        self.target_length = target_length
        self.tolerance = tolerance
    
    def __call__(self, prompt: str, completion: str, target: str = None) -> float:
        """
        Compute length-based reward.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            target: Not used
            
        Returns:
            Reward value based on length
        """
        completion_length = len(completion.split())
        
        if abs(completion_length - self.target_length) <= self.tolerance:
            return 1.0
        else:
            # Penalty based on distance from target
            distance = abs(completion_length - self.target_length)
            return max(-1.0, 1.0 - distance * 0.2)


class KeywordReward(RewardFunction):
    """
    Reward function based on keyword presence.
    """
    
    def __init__(self, required_keywords: List[str] = None, forbidden_keywords: List[str] = None):
        """
        Initialize keyword reward function.
        
        Args:
            required_keywords: Keywords that should be present
            forbidden_keywords: Keywords that should not be present
        """
        self.required_keywords = set(required_keywords or [])
        self.forbidden_keywords = set(forbidden_keywords or [])
    
    def __call__(self, prompt: str, completion: str, target: str = None) -> float:
        """
        Compute keyword-based reward.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            target: Not used
            
        Returns:
            Reward value based on keyword presence
        """
        completion_lower = completion.lower()
        words = set(completion_lower.split())
        
        reward = 0.0
        
        # Reward for required keywords
        for keyword in self.required_keywords:
            if keyword.lower() in completion_lower:
                reward += 0.5
        
        # Penalty for forbidden keywords
        for keyword in self.forbidden_keywords:
            if keyword.lower() in completion_lower:
                reward -= 1.0
        
        return max(-1.0, min(1.0, reward))


class CombinedReward(RewardFunction):
    """
    Combine multiple reward functions.
    """
    
    def __init__(self, reward_functions: List[RewardFunction], weights: List[float] = None):
        """
        Initialize combined reward function.
        
        Args:
            reward_functions: List of reward functions
            weights: Weights for each reward function (if None, equal weights)
        """
        self.reward_functions = reward_functions
        self.weights = weights or [1.0] * len(reward_functions)
        
        if len(self.weights) != len(self.reward_functions):
            raise ValueError("Number of weights must match number of reward functions")
    
    def __call__(self, prompt: str, completion: str, target: str = None) -> float:
        """
        Compute combined reward.
        
        Args:
            prompt: Input prompt
            completion: Generated completion
            target: Target completion
            
        Returns:
            Weighted average of all reward functions
        """
        rewards = []
        for reward_fn in self.reward_functions:
            reward = reward_fn(prompt, completion, target)
            rewards.append(reward)
        
        # Compute weighted average
        total_weight = sum(self.weights)
        weighted_sum = sum(r * w for r, w in zip(rewards, self.weights))
        
        return weighted_sum / total_weight


def load_sentiment_dataset(file_path: str) -> Dataset:
    """
    Load sentiment analysis dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        HuggingFace Dataset
    """
    df = pd.read_csv(file_path)
    
    # Create prompts and targets
    prompts = []
    targets = []
    
    for _, row in df.iterrows():
        text = row.get('text', row.get('review', ''))
        sentiment = row.get('sentiment', row.get('label', ''))
        
        if text and sentiment:
            prompt = f"Analyze the sentiment of this text: '{text}'. The sentiment is:"
            prompts.append(prompt)
            targets.append(sentiment)
    
    return Dataset.from_dict({
        'prompt': prompts,
        'target': targets
    })


def load_qa_dataset(file_path: str) -> Dataset:
    """
    Load question-answering dataset.
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        HuggingFace Dataset
    """
    df = pd.read_csv(file_path)
    
    prompts = []
    targets = []
    
    for _, row in df.iterrows():
        question = row.get('question', '')
        answer = row.get('answer', row.get('target', ''))
        
        if question and answer:
            prompt = f"Question: {question}\nAnswer:"
            prompts.append(prompt)
            targets.append(answer)
    
    return Dataset.from_dict({
        'prompt': prompts,
        'target': targets
    })


def create_training_batch(dataset: Dataset, batch_size: int = 4) -> List[Dict[str, Any]]:
    """
    Create training batch from dataset.
    
    Args:
        dataset: Input dataset
        batch_size: Batch size
        
    Returns:
        List of training examples
    """
    indices = np.random.choice(len(dataset), batch_size, replace=False)
    
    batch = []
    for idx in indices:
        example = dataset[idx]
        batch.append({
            'prompt': example['prompt'],
            'target': example['target']
        })
    
    return batch


def evaluate_completions(prompts: List[str], completions: List[str], targets: List[str], 
                        reward_function: RewardFunction) -> Dict[str, float]:
    """
    Evaluate generated completions.
    
    Args:
        prompts: List of prompts
        completions: List of generated completions
        targets: List of target completions
        reward_function: Reward function to use
        
    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []
    accuracies = []
    
    for prompt, completion, target in zip(prompts, completions, targets):
        # Compute reward
        reward = reward_function(prompt, completion, target)
        rewards.append(reward)
        
        # Simple accuracy (exact match)
        accuracy = 1.0 if completion.strip().lower() == target.strip().lower() else 0.0
        accuracies.append(accuracy)
    
    return {
        'mean_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'accuracy': np.mean(accuracies),
        'rewards': rewards
    } 