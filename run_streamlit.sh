#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"


if [ -d "venv" ]; then
    source venv/bin/activate
else
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi


if [ -f "Combined_dataset.csv" ]; then
    echo "[OK] Combined_dataset.csv exist"
else
    echo "[WARNING] Combined_dataset.csv not exist"
fi

echo ""
echo "========================================"
echo "  Start Streamlit app..."
echo "========================================"
echo ""
echo "  Network URL: http://0.0.0.0:8501"
echo ""

streamlit run streamlit_app.py --server.port 8501
