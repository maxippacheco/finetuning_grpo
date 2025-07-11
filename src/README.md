# GRPO (Generative Reward-Powered Optimization) for Financial Q&A

This project implements GRPO, a reinforcement learning approach for fine-tuning language models using reward functions. The implementation is specifically tailored for financial question-answering tasks using JSON datasets.

## Overview

GRPO combines:
1. **Policy gradient loss** with clipping to prevent large policy updates
2. **KL divergence penalty** to keep the model close to the reference model
3. **Custom reward functions** for financial Q&A evaluation
4. **JSON data processing** for financial datasets

## Project Structure

```
finetuning_grpo/
├── grpo.py              # Core GRPO implementation
├── data_utils.py        # Data processing and reward functions
├── json_data_loader.py  # JSON data loader for financial datasets
├── train_grpo_json.py   # Training script for JSON data
├── test_data_loader.py  # Test script for data loader
├── example_usage.py     # Simple usage examples
├── visualization.py     # Training monitoring and plots
├── requirements.txt     # Dependencies
├── data/               # JSON dataset files
│   ├── train.json      # Training data
│   ├── dev.json        # Development data
│   └── test.json       # Test data
├── loss_function.ipynb # Original notebook
└── README.md           # This file
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure JSON data files are in the `data/` directory:
   - `data/train.json` - Training dataset
   - `data/dev.json` - Development dataset  
   - `data/test.json` - Test dataset

## Quick Start

### 1. Test Data Loader
```bash
python test_data_loader.py
```

### 2. Run Training
```bash
python train_grpo_json.py
```

### 3. Run Examples
```bash
python example_usage.py
```

## Data Format

The JSON dataset contains financial Q&A examples with the following structure:

```json
{
  "pre_text": ["context paragraph 1", "context paragraph 2", ...],
  "qa": {
    "question": "What is the interest expense in 2009?",
    "answer": "380",
    "explanation": "Based on the financial statements...",
    "steps": [...],
    "program": "divide(100, 100), divide(3.8, #0)"
  },
  "id": "unique_id",
  "filename": "source_file"
}
```

## Core Components

### JSON Data Loader
- `FinancialJSONDataLoader`: Handles JSON financial Q&A datasets
- Processes context, questions, and answers
- Creates prompts and targets for training

### GRPOLoss Class
The main loss function that implements:
- Policy ratio clipping with epsilon parameter
- KL divergence penalty with beta parameter
- Per-token loss computation

### GRPOTrainer Class
Handles:
- Model initialization with LoRA adapters
- Reference model management
- Training utilities

### Reward Functions
- `FinancialQAReward`: Custom reward for financial Q&A
- Evaluates financial keywords, analytical language, length, and numerical content
- Domain-specific for financial analysis

## Training Process

1. **Data Loading**: Load JSON financial Q&A dataset
2. **Prompt Creation**: Format context + question as prompts
3. **Generation**: Generate completions for prompts
4. **Reward Computation**: Calculate rewards using financial reward function
5. **Loss Computation**: Apply GRPO loss with clipping and KL penalty
6. **Optimization**: Update LoRA parameters

## Customization

### Adding New Reward Functions
```python
class CustomReward(RewardFunction):
    def __call__(self, prompt: str, completion: str, target: str = None) -> float:
        # Your reward logic here
        return reward_value
```

### Modifying Training Parameters
```python
trainer = GRPOJSONTrainer(
    model_str='your-model',
    train_path='data/train.json',
    dev_path='data/dev.json',
    test_path='data/test.json'
)

trainer.train(
    num_epochs=5,
    batch_size=4,
    learning_rate=1e-4,
    eval_interval=50,
    save_interval=100
)
```

## Model Architecture

- **Base Model**: BabyLlama-100M (or any causal LM)
- **LoRA Adapters**: Applied to q_proj and v_proj layers
- **Reference Model**: Frozen copy of base model
- **Policy Model**: Base model + trainable LoRA adapters

## Reward Function Details

The `FinancialQAReward` function evaluates:
- **Financial keywords**: Revenue, profit, investment, etc.
- **Analytical language**: Because, therefore, however, etc.
- **Answer length**: Optimal 5-50 words
- **Numerical content**: Presence of numbers and currency symbols
- **Similarity to target**: Word overlap with reference answer
- **Exact answer matching**: Direct answer verification

## Example Output

```
GRPO Financial Q&A Training with JSON Data
==================================================
Loading training data...
Training examples: 1500
Development examples: 200
Initializing GRPO trainer...
Starting training...

Epoch 1/3
Step 10, Loss: -0.2341, Avg Reward: 0.456
Step 20, Loss: -0.1987, Avg Reward: 0.523
...

Evaluating...
Sample predictions:
Prompt: Context: interest rate to a variable interest rate...
Target: 380
Generated: The interest expense in 2009 is $3.8 million
Reward: 0.750

Average reward: 0.623
```

## Training Features

1. **Custom Reward Function**: Evaluates financial keywords, analytical language, length, and numerical content
2. **Progress Monitoring**: Real-time loss and reward tracking
3. **Visualization**: Training curves, reward distributions, and model comparisons
4. **Evaluation**: Sample predictions and metrics during training
5. **Model Checkpointing**: Regular model saves during training

## Customization Options

- **Different Models**: Change `model_str` in trainer initialization
- **Reward Functions**: Create custom reward functions by inheriting from `RewardFunction`
- **Training Parameters**: Adjust epochs, batch size, learning rate, epsilon, beta
- **Datasets**: Extend to other JSON Q&A datasets with custom data loaders

## Monitoring & Visualization

The `visualization.py` module provides:
- Training curves (loss and rewards)
- Reward distributions
- Completion length analysis
- Policy ratio clipping visualization
- Model comparison plots

## Real-World Usage

The implementation is specifically designed for financial Q&A datasets with:
- Domain-specific reward functions
- Financial keyword detection
- Analytical language evaluation
- Length optimization for detailed answers
- Numerical content recognition

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use smaller model
2. **JSON data not found**: Ensure JSON files are in `data/` directory
3. **Import errors**: Install all requirements with `pip install -r requirements.txt`

### Performance Tips

- Use smaller models for faster training
- Adjust batch size based on GPU memory
- Monitor reward values during training
- Use gradient clipping if needed

## Extending the Project

### New Datasets
1. Create custom data loading function in `json_data_loader.py`
2. Implement domain-specific reward function
3. Modify training script for new data format

### New Reward Functions
1. Inherit from `RewardFunction` base class
2. Implement `__call__` method
3. Add to training pipeline

### Different Models
1. Change `model_str` in trainer initialization
2. Adjust LoRA configuration if needed
3. Update tokenizer settings

## License

This project is for educational and research purposes.

## Citation

If you use this implementation, please cite the original GRPO paper and this repository.
