#!/bin/bash
# Adaptive RAG System - One-click startup

set -e

echo ""
echo "⚡ Adaptive RAG System"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found. Please install Python 3.9+"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found. Please install Node.js 18+"
    exit 1
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "📦 Installing Python dependencies..."
cd "$PROJECT_DIR/backend"
pip install -r requirements.txt -q

echo "📦 Installing Node dependencies..."
cd "$PROJECT_DIR/frontend"
npm install --silent

echo ""
echo "🚀 Starting services..."
echo ""

# Start backend
cd "$PROJECT_DIR/backend"
python3 main.py &
BACKEND_PID=$!
echo "✅ Backend running → http://localhost:8000  (PID: $BACKEND_PID)"

sleep 1

# Start frontend
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!
echo "✅ Frontend running → http://localhost:3000  (PID: $FRONTEND_PID)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌐 Open in browser: http://localhost:3000"
echo "📡 API docs:        http://localhost:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Press Ctrl+C to stop all services"

# Handle cleanup
cleanup() {
    echo ""
    echo "🛑 Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM

wait
