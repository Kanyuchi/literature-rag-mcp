# Literature RAG - AWS Lightsail Deployment Guide

## Overview

This guide walks you through deploying the Literature RAG system to AWS Lightsail (~$5-10/month).

## Prerequisites

- AWS Account with Lightsail access
- S3 bucket created (e.g., `lit-rag-flow`)
- AWS credentials (Access Key ID and Secret Access Key)
- Groq API key for LLM synthesis (free at https://console.groq.com)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    AWS Lightsail Instance                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │   Nginx     │───▶│  FastAPI    │───▶│   ChromaDB      │  │
│  │  (Port 80)  │    │  (Port 8001)│    │   (Embedded)    │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                  │                                 │
│         │                  ▼                                 │
│         │           ┌─────────────┐                         │
│         │           │   SQLite    │                         │
│         │           │  (Users/Jobs)│                         │
│         │           └─────────────┘                         │
└─────────│───────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────┐
│    AWS S3       │
│  (PDF Storage)  │
│  lit-rag-flow   │
└─────────────────┘
```

## Step 1: Create Lightsail Instance

1. Go to [AWS Lightsail Console](https://lightsail.aws.amazon.com/)
2. Click "Create instance"
3. Select:
   - **Region**: EU (Stockholm) `eu-north-1` (same as S3 bucket)
   - **Platform**: Linux/Unix
   - **Blueprint**: Ubuntu 22.04 LTS
   - **Instance plan**: $10/month (2 GB RAM, 1 vCPU) - recommended
     - $5/month (1 GB RAM) works but may be slow for embeddings
4. Name your instance: `lit-rag-server`
5. Click "Create instance"

## Step 2: Configure Networking

1. In Lightsail, go to your instance → Networking
2. Add firewall rules:
   - HTTP (80) - TCP - Any IP
   - HTTPS (443) - TCP - Any IP
   - Custom (8001) - TCP - Any IP (for direct API access during setup)

3. (Optional) Create a static IP:
   - Go to Networking → Create static IP
   - Attach to your instance

## Step 3: Connect and Setup

SSH into your instance (click "Connect using SSH" in Lightsail console or use your terminal):

```bash
ssh -i ~/.ssh/your-key.pem ubuntu@<your-instance-ip>
```

Run the setup script:

```bash
# Download and run setup script
curl -fsSL https://raw.githubusercontent.com/Kanyuchi/literature-rag-mcp/main/literature_review_rag_api/scripts/deploy/setup-lightsail.sh | bash

# Log out and back in for docker group
exit
# SSH back in
```

## Step 4: Deploy Application

```bash
# Clone the repository
cd ~
git clone https://github.com/Kanyuchi/literature-rag-mcp.git literature_review_rag_api

# Navigate to API directory
cd ~/literature_review_rag_api/literature_review_rag_api

# Create environment file
cp .env.example .env
nano .env  # Edit with your credentials
```

### Configure .env

Fill in these required values:

```env
# AWS S3
AWS_S3_BUCKET=lit-rag-flow
AWS_REGION=eu-north-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key

# JWT Secret (generate one)
JWT_SECRET_KEY=$(python3 -c "import secrets; print(secrets.token_hex(32))")

# Groq API Key
GROQ_API_KEY=your_groq_api_key

# CORS (update with your frontend URL)
CORS_ORIGINS=http://localhost:5173,https://your-frontend-domain.com

# Security hardening
AUTH_COOKIE_SECURE=true
AUTH_COOKIE_SAMESITE=lax
ENABLE_HSTS=true
REQUIRE_HTTPS=true
AUTH_REQUIRE_VERIFIED=false
```

Generate a secure JWT secret:
```bash
python3 -c "import secrets; print(secrets.token_hex(32))"
```

## Step 5: Build and Start

```bash
# Build and start containers
docker-compose up -d --build

# Check logs
docker-compose logs -f api

# Verify API is running
curl http://localhost:8001/healthz
```

## Step 6: Copy Pre-built Index (Optional)

If you have the pre-built 85-paper index, copy it to the server:

```bash
# From your local machine
scp -r indices/ ubuntu@<your-ip>:/home/ubuntu/literature_review_rag_api/literature_review_rag_api/

# Or use rsync
rsync -avz indices/ ubuntu@<your-ip>:/home/ubuntu/literature_review_rag_api/literature_review_rag_api/indices/
```

## Step 7: Setup Domain (Optional)

### Option A: Use Lightsail DNS
1. In Lightsail → Networking → DNS
2. Create a DNS zone for your domain
3. Add an A record pointing to your static IP

### Option B: Use External DNS
Point your domain's A record to your Lightsail static IP.

### Get SSL Certificate
```bash
# Install certbot
sudo apt-get install certbot

# Get certificate (replace with your domain)
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ~/literature_review_rag_api/literature_review_rag_api/nginx/ssl/
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ~/literature_review_rag_api/literature_review_rag_api/nginx/ssl/
```

Start Nginx production profile (HTTPS-only reverse proxy):

```bash
docker-compose --profile production up -d --build nginx certbot
```

Validate redirect and TLS:

```bash
curl -I http://your-domain.com/api/healthz
curl -I https://your-domain.com/api/healthz
```

## Domain Name Suggestions

Since you need a domain, here are some creative options:

**Professional:**
- `litrag.io` - short and memorable
- `scholarflow.app` - academic focus
- `raglit.dev` - developer-friendly

**Descriptive:**
- `literature-rag.com`
- `academic-search.ai`
- `paperquery.com`

**Budget-friendly TLDs:**
- `.dev` - $12/year (Google)
- `.app` - $14/year (Google)
- `.io` - ~$30/year
- `.com` - ~$12/year

You can register domains at:
- [Namecheap](https://namecheap.com) - often cheapest
- [Cloudflare Registrar](https://dash.cloudflare.com) - at-cost pricing
- [Google Domains](https://domains.google) (now Squarespace)

## Useful Commands

```bash
# View logs
docker-compose logs -f

# Restart services
docker-compose restart

# Stop services
docker-compose down

# Update and redeploy
git pull
docker-compose up -d --build

# Check disk space
df -h

# Check memory
free -m

# Check running containers
docker ps
```

## Troubleshooting

### API not starting
```bash
# Check logs
docker-compose logs api

# Common issues:
# - Missing environment variables
# - Database connection errors
# - Port already in use
```

### Out of memory
```bash
# Increase swap
sudo fallocate -l 2G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### S3 connection errors
- Verify AWS credentials in .env
- Check bucket region matches AWS_REGION
- Ensure IAM user has S3 permissions

## Costs Breakdown

| Service | Monthly Cost |
|---------|-------------|
| Lightsail $10 plan | $10 |
| S3 storage (~10GB) | ~$0.25 |
| S3 requests | ~$0.10 |
| Data transfer | ~$0.50 |
| **Total** | **~$11/month** |

## Security Checklist

- [ ] Change default SSH port (optional)
- [ ] Use SSH keys only (disable password auth)
- [ ] Keep system updated: `sudo apt update && sudo apt upgrade`
- [ ] Rotate AWS credentials periodically
- [ ] Use strong JWT secret
- [ ] Enable Lightsail automatic snapshots
- [ ] Serve production traffic over HTTPS only (no plaintext HTTP API exposure)
- [ ] Verify security headers are present (`X-Frame-Options`, `CSP`, `HSTS`, `Referrer-Policy`)
- [ ] Confirm auth tokens are not persisted in browser localStorage
