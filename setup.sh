#!/usr/bin/env bash
# Madison-RL setup script
set -e

echo "Creating virtual environment..."
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Setup complete. Activate with: source .venv/bin/activate"
echo "Run smoke test:   python -m tests.test_env"
echo "Train orchestrator: python -m madison_rl.training.train_orchestrator"
