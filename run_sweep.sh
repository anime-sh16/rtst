#!/bin/bash
# Sequential training sweep — run multiple configs back-to-back
# Usage: bash run_sweep.sh configs/sweep_*.yaml

set -e  # Stop on first failure. Remove if you want all runs to attempt even if one fails.

if [ $# -eq 0 ]; then
    echo "Usage: bash run_sweep.sh config1.yaml config2.yaml ..."
    echo "   or: bash run_sweep.sh configs/sweep_*.yaml"
    exit 1
fi

for config in "$@"; do
    echo "=========================================="
    echo "Starting run with: $config"
    echo "Time: $(date)"
    echo "=========================================="
    python src/train.py --config "$config"
    echo "Finished: $config at $(date)"
    echo ""
done

echo "All sweep runs complete!"
