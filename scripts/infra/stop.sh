#!/bin/bash
set -e

echo "Stopping all RAG infrastructure..."

cd terraform/environments/dev
terraform apply -auto-approve \
    -var enable_chromadb=false \
    -var enable_ollama=false \
    -var enable_neo4j=false \
    -var enable_monitoring=false

echo "All services stopped"
