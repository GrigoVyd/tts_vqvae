#!/usr/bin/env python3
"""
NMNIST Training with Original Brain LIF

This script trains the NMNIST model using the original brain LIF (LIFNode from braincog)
instead of the stochastic implementation.

Usage:
    python train_nmnist_original.py --stage 1    # Train stage 1 with original LIF
    python train_nmnist_original.py --stage 2    # Train stage 2 with original LIF
"""

import sys
import os
import argparse

# Add current directory to path
sys.path.append(os.getcwd())

# Set neuron type to original BEFORE importing any modules that use neurons
from src.modules.sblock import set_global_neuron_type, get_global_neuron_type

def setup_original_lif():
    """Set up the original brain LIF implementation."""
    try:
        set_global_neuron_type('original')
        current_type = get_global_neuron_type()
        print(f"✓ Successfully configured to use: {current_type} brain LIF")
        print("  Using LIFNode from braincog library")
        return True
    except ImportError as e:
        print(f"✗ Error: {e}")
        print("\nTo use the original brain LIF, you need to install braincog:")
        print("pip install braincog")
        print("\nAlternatively, you can train with the stochastic LIF:")
        print("python train/stage_1/run_snn_te_nmnist.sh")
        return False
    except Exception as e:
        print(f"✗ Unexpected error: {e}")
        return False


def train_stage_1():
    """Train NMNIST stage 1 with original brain LIF."""
    print("=" * 60)
    print("NMNIST STAGE 1 TRAINING - Original Brain LIF")
    print("=" * 60)
    
    # Import after setting neuron type
    from omegaconf import OmegaConf
    from src.utils.parser import get_parser_2
    from pytorch_lightning import seed_everything
    from src.utils.auto_instance import instantiate_from_config, instantiate_from_config_args
    from pytorch_lightning.trainer import Trainer
    
    # Import training functions from the original script
    sys.path.append('train/stage_1')
    from snn_te_nmnist import (get_dir_name, setup_model, setup_lightning, 
                               setup_data, setup_callback, setup_logger, setup_trainer)
    
    # Load config
    config = OmegaConf.load('train/stage_1/snn_te_nmnist.yaml')
    
    # Add original LIF indicator to experiment name
    config.exp.index = config.exp.index + '_original_lif'
    
    print(f"Experiment: {config.exp.name}/{config.exp.index}")
    print(f"Neuron type: {get_global_neuron_type()}")
    
    # Get directories
    res_dir, ckpt_dir, img_dir, tb_dir = get_dir_name(config.exp)
    print(f"Results will be saved to: {res_dir}")
    
    # Seed for reproducibility
    seed_everything(config.exp.seed)
    
    # Setup model (will use original LIF neurons)
    print("Setting up model with original brain LIF neurons...")
    model = setup_model(config.model)
    
    # Setup lightning trainer
    lightning_args = {
        'res_dir': res_dir,
        'ckpt_dir': ckpt_dir,
        'img_dir': img_dir,
        'tb_dir': tb_dir
    }
    trainer = setup_lightning(config.lightning, lightning_args)
    
    # Setup data
    print("Setting up NMNIST data...")
    data = setup_data(config.data)
    
    # Calculate learning rate
    accumulate_grad_batches = config.lightning.trainer.accumulate_grad_batches
    batch_size = data.batch_size
    base_lr = config.model.base_learning_rate
    ngpus = len(config.lightning.trainer.devices)
    model.learning_rate = accumulate_grad_batches * ngpus * batch_size * base_lr
    print(f"Learning rate: {model.learning_rate}")
    
    # Start training
    print("Starting training...")
    if config.exp.is_resume:
        trainer.fit(model=model, datamodule=data, ckpt_path=config.exp.resume_path)
    else:
        trainer.fit(model=model, datamodule=data)
    
    print("✓ Stage 1 training completed!")


def train_stage_2():
    """Train NMNIST stage 2 with original brain LIF."""
    print("=" * 60)
    print("NMNIST STAGE 2 TRAINING - Original Brain LIF")
    print("=" * 60)
    
    # Import after setting neuron type
    from omegaconf import OmegaConf
    from pytorch_lightning import seed_everything
    
    # Import training functions from the original script
    sys.path.append('train/stage_2')
    from snn_transformer_nmnist import (get_dir_name, setup_model, setup_lightning, 
                                        setup_data)
    
    # Load config
    config = OmegaConf.load('train/stage_2/snn_transformer_nmnist.yaml')
    
    # Add original LIF indicator to experiment name
    config.exp.index = config.exp.index + '_original_lif'
    
    # Update checkpoint path to use stage 1 original LIF results
    original_ckpt_path = config.model.params.ckpt_path
    new_ckpt_path = original_ckpt_path.replace('/nmnist/', '/nmnist_original_lif/')
    config.model.params.ckpt_path = new_ckpt_path
    
    print(f"Experiment: {config.exp.name}/{config.exp.index}")
    print(f"Neuron type: {get_global_neuron_type()}")
    print(f"Using checkpoint: {new_ckpt_path}")
    
    # Get directories
    res_dir, ckpt_dir, img_dir, tb_dir = get_dir_name(config.exp)
    print(f"Results will be saved to: {res_dir}")
    
    # Seed for reproducibility
    seed_everything(config.exp.seed)
    
    # Setup model (will use original LIF neurons)
    print("Setting up transformer model with original brain LIF neurons...")
    model = setup_model(config.model)
    
    # Setup lightning trainer
    lightning_args = {
        'res_dir': res_dir,
        'ckpt_dir': ckpt_dir,
        'img_dir': img_dir,
        'tb_dir': tb_dir
    }
    trainer = setup_lightning(config.lightning, lightning_args)
    
    # Setup data
    print("Setting up NMNIST data...")
    data = setup_data(config.data)
    
    # Calculate learning rate
    accumulate_grad_batches = config.lightning.trainer.accumulate_grad_batches
    batch_size = data.batch_size
    base_lr = config.model.base_learning_rate
    ngpus = len(config.lightning.trainer.devices)
    model.learning_rate = accumulate_grad_batches * ngpus * batch_size * base_lr
    print(f"Learning rate: {model.learning_rate}")
    
    # Start training
    print("Starting training...")
    if config.exp.is_resume:
        trainer.fit(model=model, datamodule=data, ckpt_path=config.exp.resume_path)
    else:
        trainer.fit(model=model, datamodule=data)
    
    print("✓ Stage 2 training completed!")


def main():
    parser = argparse.ArgumentParser(
        description="Train NMNIST with original brain LIF implementation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --stage 1    # Train stage 1 (autoencoder) with original LIF
  %(prog)s --stage 2    # Train stage 2 (transformer) with original LIF
  
Prerequisites:
  - Install braincog: pip install braincog
  - Make sure NMNIST dataset is available in ../dataset/DVS/NMNIST/
        """
    )
    
    parser.add_argument(
        '--stage',
        type=int,
        choices=[1, 2],
        required=True,
        help='Training stage (1 for autoencoder, 2 for transformer)'
    )
    
    args = parser.parse_args()
    
    print("NMNIST Training with Original Brain LIF")
    print("=" * 60)
    
    # Setup original LIF neurons
    if not setup_original_lif():
        sys.exit(1)
    
    print()
    
    # Run the appropriate training stage
    try:
        if args.stage == 1:
            train_stage_1()
        elif args.stage == 2:
            train_stage_2()
    except Exception as e:
        print(f"\n✗ Training failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure braincog is installed: pip install braincog")
        print("2. Check that NMNIST dataset is in ../dataset/DVS/NMNIST/")
        print("3. For stage 2, make sure stage 1 training completed successfully")
        sys.exit(1)


if __name__ == "__main__":
    main()