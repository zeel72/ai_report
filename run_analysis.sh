#!/bin/bash

# Lab Assignment 5: Financial HMM Analysis Runner
# CS307 - Artificial Intelligence Week 5

echo "=============================================="
echo "Financial Time Series Analysis with Gaussian HMM"
echo "Lab Assignment 5 - CS307 AI Week 5"
echo "=============================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

echo "Python version:"
python3 --version

# Navigate to Python_HMM directory
cd Python_HMM

# Check if requirements are installed
echo ""
echo "Checking dependencies..."
python3 -c "import yfinance, numpy, pandas, matplotlib, seaborn, hmmlearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing required packages..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Error: Failed to install dependencies"
        echo "Please run: pip3 install -r requirements.txt"
        exit 1
    fi
fi

echo "✓ Dependencies installed"

# Check for command line arguments
if [ $# -eq 0 ]; then
    echo ""
    echo "Running complete analysis with AAPL (10 years)..."
    echo "This may take 3-5 minutes..."
    python3 main_analysis.py
elif [ $# -eq 1 ]; then
    echo ""
    echo "Running analysis for $1 (10 years)..."
    python3 main_analysis.py $1
elif [ $# -eq 2 ]; then
    echo ""
    echo "Running analysis for $1 ($2)..."
    python3 main_analysis.py $1 $2
else
    echo "Usage: $0 [TICKER] [PERIOD]"
    echo "Examples:"
    echo "  $0                    # Analyze AAPL for 10 years"
    echo "  $0 TSLA              # Analyze TSLA for 10 years"  
    echo "  $0 ^GSPC 5y          # Analyze S&P 500 for 5 years"
    exit 1
fi

echo ""
echo "=============================================="
echo "Analysis complete! Check the results/ folder for:"
echo "- Data files and saved model"
echo "- Static plots (PNG files)"
echo "- Interactive analysis (HTML file)"
echo "- Comprehensive report (JSON file)"
echo "=============================================="