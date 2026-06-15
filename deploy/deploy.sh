#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Legal RAG AI — VPS Deploy Script
#
# Run from the project root on the VPS:
#   sudo bash deploy/deploy.sh
#
# Prerequisites: Ubuntu/Debian VPS with SSH, nginx, git, and a domain name.
# ============================================================================

APP_DIR="/opt/legal-rag-ai"
APP_USER="legalrag"
BRANCH="${DEPLOY_BRANCH:-deploy/vps-production}"

echo "==> Legal RAG AI deploy starting..."

# --- 1. System packages ---
echo "==> Installing system packages..."
apt-get update -qq
apt-get install -y -qq \
    git curl wget build-essential \
    tesseract-ocr \
    nginx certbot python3-certbot-nginx \
    ca-certificates gnupg

# --- 2. Docker (if not installed) ---
if ! command -v docker &>/dev/null; then
    echo "==> Installing Docker..."
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    chmod a+r /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-compose-plugin
fi

# --- 3. Node.js (if not installed) ---
if ! command -v node &>/dev/null; then
    echo "==> Installing Node.js 20..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y -qq nodejs
fi

# --- 4. uv (if not installed) ---
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# --- 5. App user ---
if ! id "$APP_USER" &>/dev/null; then
    echo "==> Creating app user: $APP_USER"
    useradd -r -m -s /bin/bash "$APP_USER"
    usermod -aG docker "$APP_USER"
fi

# --- 6. Clone / update repo ---
if [ -d "$APP_DIR/.git" ]; then
    echo "==> Pulling latest code..."
    cd "$APP_DIR"
    git fetch origin
    git checkout "$BRANCH"
    git pull origin "$BRANCH"
else
    echo "==> Cloning repository..."
    git clone --branch "$BRANCH" "$(pwd)" "$APP_DIR" 2>/dev/null || {
        echo "ERROR: Clone failed. Place the repo at $APP_DIR or run from within the repo."
        exit 1
    }
fi

cd "$APP_DIR"

# --- 7. Env file check ---
if [ ! -f "$APP_DIR/.env" ]; then
    echo "ERROR: $APP_DIR/.env not found. Copy .env.example and configure it first."
    echo "  cp $APP_DIR/.env.example $APP_DIR/.env"
    echo "  nano $APP_DIR/.env"
    exit 1
fi

# --- 8. Typesense data directory ---
mkdir -p "$APP_DIR/typesense-data"
chown -R "$APP_USER:$APP_USER" "$APP_DIR/typesense-data"

# --- 9. Start Typesense ---
echo "==> Starting Typesense..."
docker compose -f deploy/docker-compose.prod.yml up -d

# --- 10. Backend dependencies ---
echo "==> Installing backend dependencies..."
su - "$APP_USER" -c "cd $APP_DIR && uv sync"

# --- 11. Frontend build ---
echo "==> Building frontend..."
DOMAIN=$(grep -oP '(?<=server_name )\S+' /etc/nginx/sites-enabled/legal-rag-ai 2>/dev/null || echo "localhost")
su - "$APP_USER" -c "cd $APP_DIR/frontend && npm ci && VITE_API_BASE_URL=https://$DOMAIN/api npm run build"

# --- 12. Systemd service ---
echo "==> Installing systemd service..."
cp deploy/legal-rag-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable legal-rag-backend
systemctl restart legal-rag-backend

# --- 13. Nginx config ---
echo "==> Configuring nginx..."
cp deploy/nginx.conf /etc/nginx/sites-available/legal-rag-ai
ln -sf /etc/nginx/sites-available/legal-rag-ai /etc/nginx/sites-enabled/
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

# --- 14. Ownership ---
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# --- 15. Health check ---
echo "==> Waiting for backend to start..."
sleep 5
if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    echo "==> Backend is healthy!"
else
    echo "WARNING: Backend health check failed. Check logs: journalctl -u legal-rag-backend"
fi

echo ""
echo "============================================================"
echo "  Deploy complete!"
echo ""
echo "  Next steps:"
echo "  1. Edit /etc/nginx/sites-available/legal-rag-ai"
echo "     Replace YOUR_DOMAIN with your actual domain"
echo "  2. sudo certbot --nginx -d YOUR_DOMAIN"
echo "  3. Uncomment the HTTPS block in nginx config"
echo "  4. sudo nginx -t && sudo systemctl reload nginx"
echo "============================================================"
