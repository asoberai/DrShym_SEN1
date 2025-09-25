#!/bin/bash
"""
DrShym Climate Output Cleanup Utility
Clean up outputs, logs, and temporary files for fresh training/testing
"""

echo "🧹 DrShym Climate Cleanup Utility"
echo "=================================="

# Default cleanup mode
CLEANUP_MODE=${1:-outputs}

case $CLEANUP_MODE in
    "outputs"|"output")
        echo "🗂️ Cleaning outputs directory..."
        rm -rf outputs/*
        mkdir -p outputs/tiles outputs/scenes
        echo "✅ Outputs cleaned"
        ;;

    "artifacts"|"models")
        echo "🎯 Cleaning artifacts and model checkpoints..."
        rm -rf artifacts/checkpoints/*
        rm -f artifacts/thresholds.json
        rm -f artifacts/validation_summary.txt
        mkdir -p artifacts/checkpoints
        echo "✅ Artifacts cleaned"
        ;;

    "logs")
        echo "📋 Cleaning logs..."
        find . -name "*.log" -delete
        find . -name "*.out" -delete
        echo "✅ Logs cleaned"
        ;;

    "temp"|"tmp")
        echo "🗃️ Cleaning temporary files..."
        rm -rf /tmp/drshym_*
        rm -rf .pytest_cache
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete
        echo "✅ Temporary files cleaned"
        ;;

    "docker")
        echo "🐳 Cleaning Docker resources..."
        docker-compose down 2>/dev/null || true
        docker system prune -f
        echo "✅ Docker resources cleaned"
        ;;

    "all")
        echo "🧽 Full cleanup - removing all generated files..."
        rm -rf outputs/*
        mkdir -p outputs/tiles outputs/scenes
        rm -rf artifacts/checkpoints/*
        rm -f artifacts/thresholds.json
        rm -f artifacts/validation_summary.txt
        mkdir -p artifacts/checkpoints
        find . -name "*.log" -delete
        find . -name "*.out" -delete
        rm -rf /tmp/drshym_*
        rm -rf .pytest_cache
        find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
        find . -name "*.pyc" -delete
        docker-compose down 2>/dev/null || true
        echo "✅ Complete cleanup done"
        ;;

    *)
        echo "Usage: ./cleanup.sh [mode]"
        echo "Modes:"
        echo "  outputs   - Clean output files (default)"
        echo "  artifacts - Clean model checkpoints and artifacts"
        echo "  logs      - Clean log files"
        echo "  temp      - Clean temporary files and cache"
        echo "  docker    - Clean Docker resources"
        echo "  all       - Complete cleanup"
        exit 1
        ;;
esac

echo "🎉 Cleanup complete!"