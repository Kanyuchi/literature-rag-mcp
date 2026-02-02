#!/bin/bash
# ============================================================================
# Lightsail Instance Setup Script
# Run this on a fresh Ubuntu 22.04 Lightsail instance
# ============================================================================

set -e

echo "=========================================="
echo "Literature RAG - Lightsail Setup"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update && sudo apt-get upgrade -y

# Install Docker
echo "Installing Docker..."
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
echo "Installing Docker Compose..."
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# Install useful tools
echo "Installing additional tools..."
sudo apt-get install -y git htop curl wget unzip

# Create app directory
echo "Creating application directory..."
sudo mkdir -p /opt/lit-rag
sudo chown $USER:$USER /opt/lit-rag

# Create data directories
mkdir -p /opt/lit-rag/data/db
mkdir -p /opt/lit-rag/data/indices
mkdir -p /opt/lit-rag/data/uploads
mkdir -p /opt/lit-rag/certbot/conf
mkdir -p /opt/lit-rag/certbot/www
mkdir -p /opt/lit-rag/nginx/ssl

echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Log out and back in (for docker group)"
echo "2. Clone your repository to /opt/lit-rag"
echo "3. Copy .env.example to .env and configure"
echo "4. Run: docker-compose up -d"
echo "=========================================="
