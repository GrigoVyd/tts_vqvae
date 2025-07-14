#!/bin/bash

# NMNIST Training with Original Brain LIF
# 
# This script trains NMNIST using the original brain LIF (LIFNode from braincog)
# instead of the stochastic implementation for comparison purposes.

set -e  # Exit on any error

echo "============================================================"
echo "NMNIST Training with Original Brain LIF"
echo "============================================================"
echo ""

# Check if braincog is installed
echo "Checking prerequisites..."
python3 -c "import braincog" 2>/dev/null || {
    echo "✗ braincog is not installed"
    echo ""
    echo "Please install braincog first:"
    echo "  pip install braincog"
    echo ""
    echo "Or train with stochastic LIF instead:"
    echo "  ./train/stage_1/run_snn_te_nmnist.sh"
    echo "  ./train/stage_2/run_snn_transformer_nmnist.sh"
    exit 1
}

echo "✓ braincog is available"
echo ""

# Check if dataset exists
if [ ! -d "../dataset/DVS/NMNIST" ]; then
    echo "⚠ Warning: NMNIST dataset not found at ../dataset/DVS/NMNIST/"
    echo "   Make sure the dataset is downloaded and placed correctly"
    echo ""
fi

# Parse command line arguments
STAGE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 --stage <1|2>"
            echo ""
            echo "Stages:"
            echo "  1    Train stage 1 (SNN autoencoder) with original LIF"
            echo "  2    Train stage 2 (SNN transformer) with original LIF"
            echo ""
            echo "Examples:"
            echo "  $0 --stage 1    # Train stage 1"
            echo "  $0 --stage 2    # Train stage 2"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate stage argument
if [ -z "$STAGE" ]; then
    echo "Error: --stage argument is required"
    echo "Use --help for usage information"
    exit 1
fi

if [ "$STAGE" != "1" ] && [ "$STAGE" != "2" ]; then
    echo "Error: Stage must be 1 or 2"
    echo "Use --help for usage information"
    exit 1
fi

# Check stage 1 completion for stage 2
if [ "$STAGE" == "2" ]; then
    STAGE1_CKPT="res/snn_te_dvs/nmnist_original_lif/ckpt/"
    if [ ! -d "$STAGE1_CKPT" ]; then
        echo "✗ Error: Stage 1 training results not found"
        echo "   Expected checkpoint directory: $STAGE1_CKPT"
        echo ""
        echo "   Please complete stage 1 training first:"
        echo "   $0 --stage 1"
        exit 1
    fi
    echo "✓ Stage 1 results found for stage 2 training"
fi

echo "Starting NMNIST training..."
echo "Stage: $STAGE"
echo "Neuron type: Original brain LIF (LIFNode from braincog)"
echo ""

# Run the training
python3 train_nmnist_original.py --stage "$STAGE"

echo ""
echo "============================================================"
echo "Training completed!"
echo ""

if [ "$STAGE" == "1" ]; then
    echo "Next steps:"
    echo "1. Check results in: res/snn_te_dvs/nmnist_original_lif/"
    echo "2. Train stage 2: $0 --stage 2"
    echo ""
    echo "To compare with stochastic LIF:"
    echo "1. Train with stochastic: ./train/stage_1/run_snn_te_nmnist.sh"
    echo "2. Compare results in res/snn_te_dvs/nmnist/ vs res/snn_te_dvs/nmnist_original_lif/"
elif [ "$STAGE" == "2" ]; then
    echo "Complete training finished!"
    echo "Results saved in: res/snn_transformer/nmnist_original_lif/"
    echo ""
    echo "To compare with stochastic LIF:"
    echo "1. Train stage 2 with stochastic: ./train/stage_2/run_snn_transformer_nmnist.sh"
    echo "2. Compare results in res/snn_transformer/nmnist/ vs res/snn_transformer/nmnist_original_lif/"
fi

echo "============================================================"