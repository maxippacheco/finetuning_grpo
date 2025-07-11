#!/usr/bin/env python3
"""
Test script for the JSON data loader.
"""

from data_loader import FinancialQADataset, get_financial_qa_reward
from grpo import GRPOTrainer


def test_data_loader():
    """Test the data loader functionality."""
    print("Testing JSON Data Loader")
    print("=" * 40)
    
    # Initialize data loader
    loader = FinancialQADataset(
        train_path="data/train.json",
        dev_path="data/dev.json",
        test_path="data/test.json"
    )
    
    # Get sample data
    print("\nLoading sample data...")
    samples = loader.get_sample_data(3)
    
    for i, sample in enumerate(samples):
        print(f"\nSample {i+1}:")
        print(f"Question: {sample['question']}")
        print(f"Answer: {sample['answer']}")
        print(f"Explanation: {sample['explanation']}")
        print(f"Context length: {len(sample['context'])} characters")
        print(f"Prompt: {sample['prompt'][:200]}...")
        print(f"Target: {sample['target']}")
        print("-" * 50)
    
    # Test reward function
    print("\nTesting reward function...")
    reward_fn = get_financial_qa_reward()
    
    test_completions = [
        "The interest expense in 2009 is $3.8 million based on the financial statements.",
        "It is $3.8 million.",
        "The annual interest expense would change by $3.8 million if LIBOR changes by 100 basis points.",
        "I don't know.",
        "The company has significant foreign currency exposure that is hedged through forward contracts."
    ]
    
    for i, completion in enumerate(test_completions):
        reward = reward_fn("What is the interest expense?", completion, "3.8 million")
        print(f"Completion {i+1}: {completion}")
        print(f"Reward: {reward:.3f}")
        print()


def test_grpo_integration():
    """Test GRPO integration with the data loader."""
    print("\nTesting GRPO Integration")
    print("=" * 40)
    
    # Initialize trainer
    trainer = GRPOTrainer('babylm/babyllama-100m-2024')
    
    # Get sample data
    loader = FinancialQADataset("data/train.json")
    samples = loader.get_sample_data(2)
    
    # Test generation and reward computation
    reward_fn = get_financial_qa_reward()
    
    for i, sample in enumerate(samples):
        print(f"\nTest {i+1}:")
        print(f"Prompt: {sample['prompt'][:100]}...")
        print(f"Target: {sample['target']}")
        
        # Generate completion
        completion = trainer.generate_completion(sample['prompt'], max_new_tokens=30)
        print(f"Generated: {completion}")
        
        # Compute reward
        reward = reward_fn(sample['prompt'], completion, sample['target'])
        print(f"Reward: {reward:.3f}")
        
        # Compute loss
        loss = trainer.compute_loss(sample['prompt'], completion, reward)
        print(f"Loss: {loss.item():.4f}")
        print("-" * 50)


def test_dataset_creation():
    """Test creating HuggingFace datasets."""
    print("\nTesting Dataset Creation")
    print("=" * 40)
    
    loader = FinancialQADataset("data/train.json", "data/dev.json", "data/test.json")
    
    try:
        train_dataset, dev_dataset, test_dataset = loader.create_datasets()
        
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Dev dataset size: {len(dev_dataset) if dev_dataset else 0}")
        print(f"Test dataset size: {len(test_dataset) if test_dataset else 0}")
        
        # Show first example
        if len(train_dataset) > 0:
            first_example = train_dataset[0]
            print(f"\nFirst example:")
            print(f"Keys: {list(first_example.keys())}")
            print(f"Prompt length: {len(first_example['prompt'])}")
            print(f"Target: {first_example['target']}")
        
    except Exception as e:
        print(f"Error creating datasets: {e}")


if __name__ == "__main__":
    # Run tests
    test_data_loader()
    test_grpo_integration()
    test_dataset_creation()
    
    print("\nAll tests completed!") 