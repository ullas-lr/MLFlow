#!/bin/bash
# Script to restart the drift dashboard

echo "🔄 Restarting Drift Dashboard..."
echo ""

# Kill any existing dashboard processes
echo "1. Stopping existing dashboard processes..."
pkill -f "drift_dashboard.py" 2>/dev/null
lsof -ti:8051 | xargs kill -9 2>/dev/null
sleep 2
echo "   ✓ Stopped"

# Clear Python cache
echo ""
echo "2. Clearing Python cache..."
cd /Users/lakkurra/mlops/MLFlow
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "   ✓ Cache cleared"

# Activate venv and start dashboard
echo ""
echo "3. Starting dashboard..."
cd /Users/lakkurra/mlops/MLFlow
source venv/bin/activate
python drift_dashboard.py
