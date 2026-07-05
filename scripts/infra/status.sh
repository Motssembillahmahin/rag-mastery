#!/bin/bash

echo "Running Containers:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep -E "rag-mastery|NAMES"

echo ""
echo "Volumes:"
docker volume ls --format "table {{.Name}}" | grep -E "rag-mastery|NAME"

echo ""
echo "Networks:"
docker network ls --format "table {{.Name}}\t{{.Driver}}" | grep -E "rag-mastery|NAME"
