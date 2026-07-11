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

# --- 4. uv (install system-wide to /usr/local/bin) ---
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | env INSTALLER_NO_MODIFY_PATH=1 sh
    cp "$HOME/.local/bin/uv" /usr/local/bin/uv
    cp "$HOME/.local/bin/uvx" /usr/local/bin/uvx
else
    echo "==> uv already installed at $(command -v uv)"
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

# Source .env for all subsequent steps
set -a; source "$APP_DIR/.env"; set +a

# Resolve domain: DEPLOY_DOMAIN from .env, or fallback
DOMAIN="${DEPLOY_DOMAIN:-localhost}"
if [ "$DOMAIN" = "localhost" ] || [ "$DOMAIN" = "your-domain.com" ]; then
    echo "WARNING: DEPLOY_DOMAIN is not set in .env (got '$DOMAIN')."
    echo "         Frontend will be built with localhost API URL."
    echo "         Set DEPLOY_DOMAIN in .env and re-run to fix."
fi
echo "==> Using domain: $DOMAIN"

# --- 8. Ownership + Typesense data directory ---
mkdir -p "$APP_DIR/typesense-data"
mkdir -p "$APP_DIR/storage/pdfs"
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

# --- 9. Start Typesense ---
echo "==> Starting Typesense..."
docker compose -f deploy/docker-compose.prod.yml --env-file "$APP_DIR/.env" up -d

# --- 10. Backend dependencies ---
echo "==> Installing backend dependencies..."
su - "$APP_USER" -c "cd $APP_DIR && /usr/local/bin/uv sync"

# --- 11. Frontend build ---
# API_AUTH_TOKEN (from .env) is baked into the build so uploads can send X-API-Key.
echo "==> Building frontend..."
su - "$APP_USER" -c "cd $APP_DIR/frontend && npm ci && VITE_API_BASE_URL='https://$DOMAIN/api' VITE_API_AUTH_TOKEN='${API_AUTH_TOKEN:-}' npm run build"

# --- 12. Systemd service ---
echo "==> Installing systemd service..."
cp deploy/legal-rag-backend.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable legal-rag-backend
systemctl restart legal-rag-backend

# --- 13. Nginx config ---
echo "==> Configuring nginx..."
NGINX_CONF="/etc/nginx/sites-available/legal-rag-ai"
if [ -f "$NGINX_CONF" ] && grep -q "managed by Certbot" "$NGINX_CONF"; then
    # Certbot has added SSL to this config — don't clobber it, or HTTPS breaks.
    # To force a fresh nginx config, delete $NGINX_CONF and re-run certbot.
    echo "==> nginx config already has SSL (certbot) — leaving it untouched."
else
    cp deploy/nginx.conf "$NGINX_CONF"
    sed -i "s/YOUR_DOMAIN/$DOMAIN/g" "$NGINX_CONF"
    ln -sf "$NGINX_CONF" /etc/nginx/sites-enabled/
    rm -f /etc/nginx/sites-enabled/default
fi
nginx -t && systemctl reload nginx

# --- 14. Health check ---
echo "==> Waiting for backend to start..."
sleep 5
if curl -sf http://127.0.0.1:8000/health > /dev/null; then
    echo "==> Backend is healthy!"
else
    echo "WARNING: Backend health check failed. Check logs: journalctl -u legal-rag-backend"
fi

echo ""
echo "============================================================"
echo "  Deploy complete!  Domain: $DOMAIN"
echo ""
echo "  Next steps (if not done already):"
echo "  1. sudo certbot --nginx -d $DOMAIN"
echo "  2. sudo nginx -t && sudo systemctl reload nginx"
echo "============================================================"
