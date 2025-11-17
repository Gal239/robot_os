#!/bin/bash
#
# Start Echo Scene Maker UI
# =========================
#
# Run this from simulation_center directory

echo ""
echo "============================================================"
echo "🤖 Starting Echo Scene Maker UI..."
echo "============================================================"
echo ""

# Check if we're in the right directory
if [ ! -d "ai_orchestration" ]; then
    echo "❌ Error: Must run from simulation_center directory"
    echo "   Current directory: $(pwd)"
    echo ""
    echo "   Run this instead:"
    echo "   cd /path/to/simulation_center"
    echo "   ./start_echo_ui.sh"
    exit 1
fi

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo "⚠️  Flask not found. Installing dependencies..."
    pip install -r core/sim_agents/ui/requirements.txt
    echo ""
fi

# Start server
echo "🚀 Starting server on http://localhost:5050"
echo ""
python core/sim_agents/ui/backend/server.py
