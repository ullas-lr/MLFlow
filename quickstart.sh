#!/bin/bash

# Quick Start Script for ML Model Validation and Monitoring
# This script helps you get started quickly

set -e

echo "=========================================="
echo "ML Model Validation & Monitoring Setup"
echo "=========================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python
echo "Checking Python..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version)
    echo -e "${GREEN}✓${NC} Python found: $PYTHON_VERSION"
else
    echo -e "${RED}✗${NC} Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Check Ollama
echo ""
echo "Checking Ollama..."
if command -v ollama &> /dev/null; then
    echo -e "${GREEN}✓${NC} Ollama is installed"
    
    # Check if Ollama is running
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Ollama server is running"
        
        # List models
        echo ""
        echo "Available Ollama models:"
        ollama list
    else
        echo -e "${YELLOW}⚠${NC} Ollama is installed but not running"
        echo "  Start it with: ollama serve"
        echo ""
        read -p "Would you like to start Ollama now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Starting Ollama in background..."
            nohup ollama serve > /dev/null 2>&1 &
            sleep 2
            echo -e "${GREEN}✓${NC} Ollama started"
        fi
    fi
else
    echo -e "${RED}✗${NC} Ollama not found"
    echo ""
    echo "Please install Ollama first:"
    echo "  macOS:  brew install ollama"
    echo "  Linux:  curl -fsSL https://ollama.com/install.sh | sh"
    echo "  Or visit: https://ollama.ai"
    exit 1
fi

# Setup virtual environment
echo ""
echo "Setting up Python environment..."
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
else
    echo -e "${GREEN}✓${NC} Virtual environment already exists"
fi

# Activate and install dependencies
echo ""
echo "Installing dependencies..."
source venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -r requirements.txt > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Dependencies installed"

# Check for .env file
echo ""
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}⚠${NC} .env file not found"
    echo ""
    read -p "Would you like to set up DagsHub integration? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        cp .env.example .env
        echo ""
        echo "Please edit .env file with your DagsHub credentials:"
        echo "  1. Create account at https://dagshub.com"
        echo "  2. Create a new repository"
        echo "  3. Get your token from Settings → Tokens"
        echo "  4. Edit .env file with your credentials"
        echo ""
        echo "After editing .env, run experiments with: python run_experiments.py --dagshub"
    else
        echo "Skipping DagsHub setup. Will use local MLflow tracking."
    fi
else
    echo -e "${GREEN}✓${NC} .env file exists"
fi

# Pull a model if none exist
echo ""
echo "Checking for Ollama models..."
MODEL_COUNT=$(ollama list | tail -n +2 | wc -l | tr -d ' ')

if [ "$MODEL_COUNT" -eq "0" ]; then
    echo -e "${YELLOW}⚠${NC} No models found"
    echo ""
    echo "Recommended models:"
    echo "  1. phi3:mini  - Fast, small (2GB) - Good for testing"
    echo "  2. llama2     - Balanced (4GB) - Good all-around"
    echo "  3. mistral    - High quality (4GB)"
    echo ""
    read -p "Which model would you like to pull? (1/2/3) " -n 1 -r
    echo
    case $REPLY in
        1)
            MODEL="phi3:mini"
            ;;
        2)
            MODEL="llama2"
            ;;
        3)
            MODEL="mistral"
            ;;
        *)
            MODEL="phi3:mini"
            ;;
    esac
    
    echo ""
    echo "Pulling $MODEL (this may take a few minutes)..."
    ollama pull $MODEL
    echo -e "${GREEN}✓${NC} Model $MODEL pulled successfully"
fi

# Run quick test
echo ""
echo "Running quick test..."
python run_experiments.py --quick-test

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo -e "${GREEN}✓ Setup Complete!${NC}"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Run full validation:"
    echo "   python run_experiments.py"
    echo ""
    echo "2. View results in MLflow:"
    echo "   mlflow ui"
    echo "   Then open: http://localhost:5000"
    echo ""
    echo "3. Run monitoring dashboard:"
    echo "   python monitoring_dashboard.py --demo"
    echo "   Then open: http://localhost:8050"
    echo ""
    echo "4. Analyze in Jupyter:"
    echo "   jupyter notebook notebooks/analysis.ipynb"
    echo ""
    echo "For detailed guide, see: SETUP_GUIDE.md"
    echo ""
else
    echo ""
    echo -e "${RED}✗ Quick test failed${NC}"
    echo "Please check the error messages above"
    exit 1
fi
