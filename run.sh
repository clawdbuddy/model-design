#!/bin/bash
set -e

# 激活 Conda 环境
source $(conda info --base)/etc/profile.d/conda.sh
conda activate model-design

MODE="mock"
API_KEY=""
MODEL="gpt-4"
CONFIG="config/default.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --mock) MODE="mock"; shift ;;
        --api-key) API_KEY="$2"; MODE="api"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --config) CONFIG="$2"; shift 2 ;;
        *) echo "Unknown: $1"; exit 1 ;;
    esac
done

if [ "$MODE" = "api" ] && [ -z "$API_KEY" ]; then
    if [ -f ".env" ]; then
        export $(grep -v '^#' .env | xargs)
        API_KEY=$OPENAI_API_KEY
    fi
    if [ -z "$API_KEY" ]; then
        echo "❌ API key required. Set OPENAI_API_KEY or use --api-key"
        exit 1
    fi
fi

echo "🚀 Model Design Agent (Conda)"
echo "   Environment: model-design"
echo "   Mode: $MODE"
echo "   CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"

python -m src.main --mode $MODE --config $CONFIG ${API_KEY:+--api-key $API_KEY} ${MODEL:+--model $MODEL}
