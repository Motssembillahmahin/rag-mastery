#!/bin/bash
set -e

echo "Destroying all infrastructure..."

cd terraform/environments/dev
terraform destroy -auto-approve

echo "Infrastructure destroyed"
