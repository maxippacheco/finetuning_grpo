import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import torch


def plot_training_curves(losses: List[float], rewards: List[float], save_path: str = None):
    """
    Plot training curves for loss and rewards.
    
    Args:
        losses: List of loss values
        rewards: List of reward values
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot losses
    ax1.plot(losses, label='Training Loss', color='red')
    ax1.set_title('Training Loss Over Time')
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot rewards
    ax2.plot(rewards, label='Average Reward', color='blue')
    ax2.set_title('Average Reward Over Time')
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Reward')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_reward_distribution(rewards: List[float], save_path: str = None):
    """
    Plot distribution of rewards.
    
    Args:
        rewards: List of reward values
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    plt.hist(rewards, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(np.mean(rewards), color='red', linestyle='--', 
                label=f'Mean: {np.mean(rewards):.3f}')
    plt.axvline(np.median(rewards), color='green', linestyle='--', 
                label=f'Median: {np.median(rewards):.3f}')
    
    plt.title('Distribution of Rewards')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_completion_lengths(completions: List[str], save_path: str = None):
    """
    Plot distribution of completion lengths.
    
    Args:
        completions: List of generated completions
        save_path: Path to save the plot
    """
    lengths = [len(completion.split()) for completion in completions]
    
    plt.figure(figsize=(10, 6))
    
    plt.hist(lengths, bins=20, alpha=0.7, color='lightgreen', edgecolor='black')
    plt.axvline(np.mean(lengths), color='red', linestyle='--', 
                label=f'Mean: {np.mean(lengths):.1f} words')
    plt.axvline(np.median(lengths), color='green', linestyle='--', 
                label=f'Median: {np.median(lengths):.1f} words')
    
    plt.title('Distribution of Completion Lengths')
    plt.xlabel('Number of Words')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_ratio_clipping(ratios: torch.Tensor, epsilon: float = 0.2, save_path: str = None):
    """
    Visualize policy ratio clipping.
    
    Args:
        ratios: Policy ratios before clipping
        epsilon: Clipping parameter
        save_path: Path to save the plot
    """
    ratios_np = ratios.detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(10, 6))
    
    # Plot original ratios
    plt.hist(ratios_np, bins=50, alpha=0.5, label='Original Ratios', color='blue')
    
    # Plot clipped ratios
    clipped_ratios = np.clip(ratios_np, 1 - epsilon, 1 + epsilon)
    plt.hist(clipped_ratios, bins=50, alpha=0.5, label='Clipped Ratios', color='red')
    
    # Add clipping boundaries
    plt.axvline(1 - epsilon, color='green', linestyle='--', label=f'Lower bound: {1-epsilon}')
    plt.axvline(1 + epsilon, color='green', linestyle='--', label=f'Upper bound: {1+epsilon}')
    plt.axvline(1, color='black', linestyle='-', alpha=0.5, label='Reference (1.0)')
    
    plt.title('Policy Ratio Clipping Visualization')
    plt.xlabel('Policy Ratio (π/π_ref)')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def create_training_report(
    losses: List[float],
    rewards: List[float],
    completions: List[str],
    targets: List[str],
    save_dir: str = "training_report"
):
    """
    Create a comprehensive training report with multiple plots.
    
    Args:
        losses: List of loss values
        rewards: List of reward values
        completions: List of generated completions
        targets: List of target completions
        save_dir: Directory to save the report
    """
    import os
    os.makedirs(save_dir, exist_ok=True)
    
    # Create plots
    plot_training_curves(losses, rewards, f"{save_dir}/training_curves.png")
    plot_reward_distribution(rewards, f"{save_dir}/reward_distribution.png")
    plot_completion_lengths(completions, f"{save_dir}/completion_lengths.png")
    
    # Create summary statistics
    stats = {
        'final_loss': losses[-1] if losses else None,
        'final_reward': rewards[-1] if rewards else None,
        'mean_reward': np.mean(rewards) if rewards else None,
        'std_reward': np.std(rewards) if rewards else None,
        'mean_completion_length': np.mean([len(c.split()) for c in completions]) if completions else None,
        'num_examples': len(completions)
    }
    
    # Save statistics
    with open(f"{save_dir}/training_stats.json", 'w') as f:
        import json
        json.dump(stats, f, indent=2)
    
    print(f"Training report saved to {save_dir}/")
    print("Statistics:", stats)


def plot_model_comparison(
    base_completions: List[str],
    trained_completions: List[str],
    prompts: List[str],
    save_path: str = None
):
    """
    Compare completions from base and trained models.
    
    Args:
        base_completions: Completions from base model
        trained_completions: Completions from trained model
        prompts: Input prompts
        save_path: Path to save the plot
    """
    # Calculate metrics for comparison
    base_lengths = [len(c.split()) for c in base_completions]
    trained_lengths = [len(c.split()) for c in trained_completions]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Length comparison
    ax1.boxplot([base_lengths, trained_lengths], labels=['Base Model', 'Trained Model'])
    ax1.set_title('Completion Length Comparison')
    ax1.set_ylabel('Number of Words')
    ax1.grid(True)
    
    # Length distribution
    ax2.hist(base_lengths, alpha=0.5, label='Base Model', bins=15)
    ax2.hist(trained_lengths, alpha=0.5, label='Trained Model', bins=15)
    ax2.set_title('Length Distribution')
    ax2.set_xlabel('Number of Words')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def print_sample_comparisons(
    prompts: List[str],
    base_completions: List[str],
    trained_completions: List[str],
    targets: List[str],
    num_samples: int = 3
):
    """
    Print sample comparisons between base and trained models.
    
    Args:
        prompts: Input prompts
        base_completions: Completions from base model
        trained_completions: Completions from trained model
        targets: Target completions
        num_samples: Number of samples to show
    """
    print("\n" + "="*80)
    print("SAMPLE COMPARISONS")
    print("="*80)
    
    for i in range(min(num_samples, len(prompts))):
        print(f"\nSample {i+1}:")
        print(f"Prompt: {prompts[i]}")
        print(f"Target: {targets[i]}")
        print(f"Base Model: {base_completions[i]}")
        print(f"Trained Model: {trained_completions[i]}")
        print("-" * 40) 