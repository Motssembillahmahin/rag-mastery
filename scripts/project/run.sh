#!/bin/bash
set -e

PROJECT=$1

if [ ! -d "$PROJECT" ]; then
    echo "Error: Project $PROJECT not found"
    exit 1
fi

echo "Running project: ${PROJECT}"
cd "$PROJECT"
uv run python main.py
