# Production Deployment Guide

Deploy Legal RAG AI to a VPS with nginx, Typesense (Docker), and the FastAPI backend as a systemd service.

## Prerequisites

- **VPS**: Ubuntu 22.04+, 12GB RAM, 6 vCPUs, 80GB+ disk
- **Domain name** with DNS A record pointing to the VPS IP
- **SSH access** with sudo privileges
- **Gemini API key** from [Google AI Studio](https://aistudio.google.com/apikey)

## 1. Initial Server Setup

```bash
# SSH into your VPS
ssh root@your-server-ip

# Create swap (recommended for 12GB RAM)
fallocate -l 4G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab

# Firewall
ufw allow OpenSSH
ufw allow 'Nginx Full'
ufw enable
```

## 2. Clone and Configure

```bash
# Clone the repo
cd /opt
git clone <your-repo-url> legal-rag-ai
cd legal-rag-ai
git checkout deploy/vps-production

# Create .env from example
cp .env.example .env
nano .env
```

### Required `.env` changes for production

```env
# Set your Gemini API key
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-api-key-here

# Strong Typesense API key (generate one)
TYPESENSE_API_KEY=<random-32-char-string>

# Production CORS
CORS_ALLOW_ORIGINS=https://your-domain.com

# Storage paths
PDF_STORAGE_DIR=/opt/legal-rag-ai/storage/pdfs
TYPESENSE_DATA_DIR=/opt/legal-rag-ai/typesense-data

# Optional: API auth token for write endpoints
API_AUTH_TOKEN=<your-secret-token>

# Force CPU for embeddings (no GPU on VPS)
EMBEDDING_DEVICE=cpu
```

## 3. Run Deploy Script

```bash
sudo bash deploy/deploy.sh
```

This installs all dependencies (Docker, Node.js, uv, Tesseract), starts Typesense, builds the frontend, and configures nginx + systemd.

## 4. Configure Domain in Nginx

```bash
# Replace YOUR_DOMAIN in the nginx config
sudo sed -i 's/YOUR_DOMAIN/your-domain.com/g' /etc/nginx/sites-available/legal-rag-ai
sudo nginx -t && sudo systemctl reload nginx
```

## 5. SSL with Let's Encrypt

```bash
sudo certbot --nginx -d your-domain.com
```

After certbot succeeds, uncomment the HTTPS server block in `/etc/nginx/sites-available/legal-rag-ai` and reload:

```bash
sudo nginx -t && sudo systemctl reload nginx
```

## 6. Rebuild Frontend with Production URL

```bash
cd /opt/legal-rag-ai/frontend
VITE_API_BASE_URL=https://your-domain.com/api npm run build
```

## 7. Verify

```bash
# Backend health
curl http://127.0.0.1:8000/health

# Readiness (checks Typesense + LLM)
curl http://127.0.0.1:8000/health/ready

# Storage stats
curl http://127.0.0.1:8000/health/storage

# Via domain
curl https://your-domain.com/api/health
```

## Updating

```bash
cd /opt/legal-rag-ai
git pull origin deploy/vps-production
sudo bash deploy/deploy.sh
```

## Monitoring and Logs

```bash
# Backend logs
journalctl -u legal-rag-backend -f

# Typesense logs
docker compose -f deploy/docker-compose.prod.yml logs -f

# Nginx logs
tail -f /var/log/nginx/access.log
tail -f /var/log/nginx/error.log

# Service status
systemctl status legal-rag-backend
docker compose -f deploy/docker-compose.prod.yml ps
```

## Architecture

```
                    +-----------+
   HTTPS (443) --> |   nginx   | --> /          Frontend (static)
                    |           | --> /api/*     Backend (port 8000)
                    |           | --> /pdfs/*    Backend (PDF files)
                    +-----------+
                         |
                    +-----------+
                    |  Backend  |  systemd service (uvicorn)
                    |  FastAPI  |  LLM: Gemini API / Ollama
                    +-----------+
                         |
                    +-----------+
                    | Typesense |  Docker container
                    |  :8108    |  Vector + keyword search
                    +-----------+
```

## Backup

```bash
# Typesense data
tar czf typesense-backup-$(date +%Y%m%d).tar.gz /opt/legal-rag-ai/typesense-data

# PDF storage
tar czf pdfs-backup-$(date +%Y%m%d).tar.gz /opt/legal-rag-ai/storage/pdfs

# Environment
cp /opt/legal-rag-ai/.env /opt/legal-rag-ai/.env.backup-$(date +%Y%m%d)
```
