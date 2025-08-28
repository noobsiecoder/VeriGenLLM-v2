# test_rlft.py
"""
Test script for running RLFT on a subset of HDLBits dataset

Author: Test Script
Date:   Aug 27th, 2025
Place:  Boston, MA
"""

import os
import sys
import argparse
import json
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.rlft import RLFineTuner
from src.constants import RLPolicy


def create_test_dataset_order(dataset_path: str, num_problems: int = 5) -> str:
    """
    Create a temporary orders.txt file with only the first N problems
    
    Parameters:
    -----------
    dataset_path: str
        Path to the dataset directory
    num_problems: int
        Number of problems to include in the test
        
    Returns:
    --------
    str: Path to the temporary orders file
    """
    # Read the original orders.txt
    original_orders_path = os.path.join(dataset_path, "orders.txt")
    if not os.path.exists(original_orders_path):
        raise FileNotFoundError(f"Original orders.txt not found at {original_orders_path}")
    
    with open(original_orders_path, "r") as f:
        all_problems = [line.strip() for line in f.readlines() if line.strip()]
    
    # Take only the first N problems
    test_problems = all_problems[:num_problems]
    
    # Create temporary orders file
    test_orders_path = os.path.join(dataset_path, "test_orders.txt")
    with open(test_orders_path, "w") as f:
        for problem in test_problems:
            f.write(f"{problem}\n")
    
    print(f"Created test dataset with {len(test_problems)} problems:")
    for i, problem in enumerate(test_problems, 1):
        print(f"  {i}. {problem}")
    
    return test_orders_path


def run_test_rlft(
    model_id: str = "codegen-2b",
    model_name: str = "Salesforce/codegen-2B-multi",
    policy: str = "PPO",
    num_problems: int = 5,
    batch_size: int = 2,
    epochs: int = 2,
    use_wandb: bool = False,
):
    """
    Run RLFT test on a small subset of the dataset
    
    Parameters:
    -----------
    model_id: str
        Short identifier for the model
    model_name: str
        HuggingFace model path
    policy: str
        RL policy to use (PPO or GRPO)
    num_problems: int
        Number of problems to test on
    batch_size: int
        Training batch size
    epochs: int
        Number of training epochs
    use_wandb: bool
        Whether to use W&B logging
    """
    print("="*80)
    print("RLFT TEST CONFIGURATION")
    print("="*80)
    print(f"Model ID: {model_id}")
    print(f"Model Name: {model_name}")
    print(f"Policy: {policy}")
    print(f"Number of Problems: {num_problems}")
    print(f"Batch Size: {batch_size}")
    print(f"Epochs: {epochs}")
    print(f"W&B Logging: {'Enabled' if use_wandb else 'Disabled'}")
    print("="*80)
    
    # Convert policy string to enum
    policy_enum = RLPolicy[policy.upper()]
    
    # Initialize RL Fine-tuner
    model_info = {
        "model_id": model_id,
        "model_name": model_name,
    }
    
    rlft = RLFineTuner(model_info=model_info, policy=policy_enum)
    
    # Prepare dataset
    dataset_path = "dataset/testbench/hdlbits/"
    
    # Create test dataset order file
    test_orders_path = create_test_dataset_order(dataset_path, num_problems)
    
    # Temporarily replace the orders.txt path
    original_orders_path = os.path.join(dataset_path, "orders.txt")
    
    try:
        # Backup original and use test version
        if os.path.exists(original_orders_path):
            os.rename(original_orders_path, original_orders_path + ".backup")
        os.rename(test_orders_path, original_orders_path)
        
        # Run fine-tuning
        print("\nStarting RLFT training...")
        rlft.fine_tune(
            model_id=model_id,
            model_name=model_name,
            batch_size=batch_size,
            epochs=epochs,
            policy=policy_enum,
            save_to_hub=False,  # Disable HF upload for testing
            wandb_project="verilog-rlft-test" if use_wandb else None,
        )
        
        print("\nRLFT test completed successfully!")
        
    except Exception as e:
        print(f"\nError during RLFT test: {e}")
        raise
        
    finally:
        # Restore original orders.txt
        if os.path.exists(original_orders_path + ".backup"):
            if os.path.exists(original_orders_path):
                os.remove(original_orders_path)
            os.rename(original_orders_path + ".backup", original_orders_path)
        
        # Clean up test orders file if it still exists
        if os.path.exists(test_orders_path):
            os.remove(test_orders_path)


def main():
    """Main function with argument parsing"""
    parser = argparse.ArgumentParser(
        description="Test RLFT on a subset of HDLBits dataset"
    )
    
    parser.add_argument(
        "--model-id",
        type=str,
        default="codegen-2b",
        help="Short identifier for the model (default: codegen-2b)"
    )
    
    parser.add_argument(
        "--model-name",
        type=str,
        default="Salesforce/codegen-2B-multi",
        help="HuggingFace model path (default: Salesforce/codegen-2B-multi)"
    )
    
    parser.add_argument(
        "--policy",
        type=str,
        choices=["PPO", "GRPO"],
        default="PPO",
        help="RL policy to use (default: PPO)"
    )
    
    parser.add_argument(
        "--num-problems",
        type=int,
        default=5,
        help="Number of problems to test on (default: 5)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Training batch size (default: 2)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=2,
        help="Number of training epochs (default: 2)"
    )
    
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Enable W&B logging"
    )
    
    args = parser.parse_args()
    
    # Check environment variables
    if not os.getenv("HUGGINGFACE_API_KEY"):
        print("Warning: HUGGINGFACE_API_KEY not set in environment")
        print("You may need to set it to access gated models")
    
    # Run the test
    run_test_rlft(
        model_id=args.model_id,
        model_name=args.model_name,
        policy=args.policy,
        num_problems=args.num_problems,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_wandb=args.wandb,
    )


if __name__ == "__main__":
    main()