#!/bin/bash
# ═══════════════════════════════════════════════════════
#   Parallel Traffic Simulation - Startup Script
# ═══════════════════════════════════════════════════════

echo ""
echo "  Parallel Traffic Simulation"
echo "  Time-Dependent Signal Optimization"
echo ""
echo "  Starting Python server..."
echo "  Open browser at: http://localhost:8765"
echo ""
echo "  TABS:"
echo "  1. Live Simulation  - Watch real-time adaptive vs static signals"
echo "  2. Benchmark        - Sequential vs Parallel comparison"
echo "  3. About            - Architecture and parallelism explanation"
echo ""
echo "  Press Ctrl+C to stop the server"
echo ""

cd "$(dirname "$0")"
python3 server.py
