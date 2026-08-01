#!/bin/bash

set -e

echo "=== Starting Topographic UQ Project Setup ==="

echo -e "\n Building Environments..."
micromamba env create -f env-model.yml
micromamba env create -f env-montecarlo.yml
micromamba env create -f env-energy-analysis.yml

echo -e "\n=== Setup Complete! ==="