#!/usr/bin/env python3
"""
Configuration script for switching between brain LIF implementations.

This script allows you to easily switch between:
- Original brain LIF (LIFNode from braincog)
- Developed stochastic LIF (ProbabilisticLIFActivation)

Usage:
    python configure_neuron_type.py --type original
    python configure_neuron_type.py --type stochastic
    python configure_neuron_type.py --info
"""

import argparse
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.modules.sblock import set_global_neuron_type, get_global_neuron_type


def main():
    parser = argparse.ArgumentParser(
        description="Configure neuron type for brain LIF comparison",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --type stochastic    # Use your developed stochastic LIF
  %(prog)s --type original      # Use original brain LIF from braincog
  %(prog)s --info               # Show current configuration
        """
    )
    
    parser.add_argument(
        '--type', 
        choices=['original', 'stochastic'],
        help='Type of LIF neuron to use'
    )
    
    parser.add_argument(
        '--info',
        action='store_true',
        help='Show current neuron type configuration'
    )
    
    args = parser.parse_args()
    
    if args.info:
        current_type = get_global_neuron_type()
        print(f"Current neuron type: {current_type}")
        print()
        print("Available types:")
        print("  - 'original': LIFNode from braincog (original brain LIF)")
        print("  - 'stochastic': ProbabilisticLIFActivation (your developed stochastic LIF)")
        return
    
    if args.type:
        try:
            set_global_neuron_type(args.type)
            print(f"✓ Successfully set neuron type to: {args.type}")
            
            if args.type == "original":
                print("  Using original brain LIF (LIFNode from braincog)")
            elif args.type == "stochastic":
                print("  Using your developed stochastic LIF (ProbabilisticLIFActivation)")
                
        except ImportError as e:
            print(f"✗ Error: {e}")
            print("  Make sure braincog is installed if using 'original' type")
            sys.exit(1)
        except Exception as e:
            print(f"✗ Error: {e}")
            sys.exit(1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()