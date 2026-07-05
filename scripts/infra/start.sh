#!/bin/bash
set -e

PROJECT=$1
SERVICES_FILE="${PROJECT}/services.yaml"

if [ ! -f "$SERVICES_FILE" ]; then
    echo "Error: $SERVICES_FILE not found"
    exit 1
fi

echo "Starting infrastructure for: ${PROJECT}"

# Parse required services from services.yaml
REQUIRED_SERVICES=$(grep -A 10 "required:" $SERVICES_FILE | grep "^\s*-" | awk '{print $2}')

# Build terraform variables dynamically
TF_VARS=""
for service in $REQUIRED_SERVICES; do
    case $service in
        chromadb) TF_VARS="${TF_VARS} -var enable_chromadb=true" ;;
        ollama) TF_VARS="${TF_VARS} -var enable_ollama=true" ;;
        neo4j) TF_VARS="${TF_VARS} -var enable_neo4j=true" ;;
        monitoring) TF_VARS="${TF_VARS} -var enable_monitoring=true" ;;
        *) echo "Warning: Unknown service: $service" ;;
    esac
done

# Add optional services if --monitor flag is passed
if [[ "$*" == *"--monitor"* ]]; then
    TF_VARS="${TF_VARS} -var enable_monitoring=true"
fi

# Apply terraform with dynamic variables
cd terraform/environments/dev
terraform apply -auto-approve $TF_VARS

echo "Services started: ${REQUIRED_SERVICES}"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
