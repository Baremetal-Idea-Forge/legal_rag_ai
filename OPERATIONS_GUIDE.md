# Operations Guide

Monitoring, logging, troubleshooting, and maintenance for production deployments.

---

## Health Checks

### Liveness & Readiness Endpoints

```bash
# Liveness (always 200 if process is alive)
curl http://localhost:8000/health
# Response: {"status":"ok","version":"1.0.0"}

# Readiness (503 if dependencies unavailable)
curl http://localhost:8000/health/ready
# Response: {"status":"ready","version":"1.0.0","checks":{"typesense":true,"llm":true}}
```

### Configure Load Balancer

```nginx
# Nginx upstream config
upstream legal_rag_backend {
    server backend:8000;
}

# Health probe for load balancer
server {
    listen 8080;
    location /health {
        proxy_pass http://legal_rag_backend;
    }
}
```

---

## Storage Monitoring

### Check Disk Usage

```bash
# Via API
curl http://localhost:8000/storage
# Response:
# {
#   "total_bytes": 1099511627776,
#   "used_bytes": 10737418240,
#   "quota_bytes": 85899345920,
#   "percent_used": 12.5,
#   "status": "ok"
# }

# Via CLI
du -sh storage/pdfs
du -sh typesense-data
```

### Storage Status Interpretation

| Status | Condition | Action |
|--------|-----------|--------|
| `ok` | < 80% of quota | Normal operation |
| `warning` | 80–95% of quota | Plan cleanup or expansion |
| `critical` | ≥ 95% of quota | Immediately free space |

### Free Up Space

```bash
# Remove old PDFs (keep last 7 days)
find storage/pdfs -type f -mtime +7 -delete

# Clear Typesense (WARNING: this re-indexes everything)
docker-compose down typesense
rm -rf typesense-data/*
docker-compose up -d typesense
```

---

## Logging

### Enable Debug Logging

Set in `.env`:
```env
DEBUG=true
```

Restart backend:
```bash
systemctl restart legal-rag-backend
```

### Log Files

```bash
# Frontend (React)
# Browser DevTools Console (F12 → Console tab)

# Backend (FastAPI/Python)
# Stdout/stderr (captured by systemd journal)
journalctl -u legal-rag-backend -f  # Follow logs

# Docker logs
docker-compose logs -f backend
docker-compose logs -f typesense
```

### Structured Logging

Logs include correlation IDs for tracing requests:

```
2026-07-25T10:15:30.123Z [req-id:abc-123] INFO: Search 'Article 21' (hybrid) → 5 hits
2026-07-25T10:15:30.245Z [req-id:abc-123] DEBUG: Query embedding: 1024 dims, first 3 values: 0.123456, -0.234567, 0.345678
2026-07-25T10:15:30.456Z [req-id:abc-123] INFO: LLM response: 'Article 21 states that...'
```

---

## Common Issues & Troubleshooting

### Typesense Won't Connect

**Symptoms**: All searches fail with `SearchError`

**Diagnosis**:
```bash
# Check if Typesense is running
curl http://localhost:8108/health

# View logs
docker-compose logs typesense

# Check network
docker network inspect legal_rag_ai_default
```

**Solutions**:
```bash
# Restart Typesense
docker-compose restart typesense

# Or re-initialize
docker-compose down
docker-compose up -d typesense

# Verify connectivity from backend
docker-compose exec backend curl http://typesense:8108/health
```

### LLM Timeouts

**Symptoms**: Chat requests fail with `LLMError` after ~120 seconds

**Diagnosis**:
```bash
# Check if Gemini API is reachable (if using Gemini)
curl "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key=$GEMINI_API_KEY"

# Or check Ollama (if local)
curl http://localhost:11434/api/generate -X POST \
  -H "Content-Type: application/json" \
  -d '{"model":"gemma3:4b","prompt":"test"}'
```

**Solutions**:
```bash
# Increase timeout in .env
GEMINI_TIMEOUT_SECONDS=180  # Up from 120

# Reduce LLM_MAX_RETRIES if needed
GEMINI_MAX_RETRIES=1

# Check network/firewall
ping generativelanguage.googleapis.com  # For Gemini
```

### High Memory Usage

**Symptoms**: Backend crashes with OOM or becomes unresponsive

**Causes**:
1. Large embedding model not fitting in VRAM
2. Large context sent to LLM
3. Memory leak in long-running process

**Diagnosis**:
```bash
# Check memory usage
free -h
ps aux | grep python  # Look for RSS column

# Monitor in real-time
watch -n 1 'ps aux | grep python | grep -v grep'
```

**Solutions**:
```env
# Use smaller embedding model
EMBEDDING_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_DEVICE=cpu

# Reduce context size
RAG_MAX_CONTEXT_CHARS=8000

# Restart backend periodically (e.g., daily cron job)
0 2 * * * systemctl restart legal-rag-backend
```

### Search Returns No Results

**Symptoms**: Searches for indexed content return 0 hits

**Diagnosis**:
```bash
# Check Typesense collection exists
curl http://localhost:8108/collections \
  -H "X-TYPESENSE-API-KEY: xyz"

# Check if documents are indexed
curl "http://localhost:8108/collections/pdf_chunks/documents?filter_by=&limit=1" \
  -H "X-TYPESENSE-API-KEY: xyz"
```

**Solutions**:
1. Re-ingest PDFs:
   ```bash
   curl -X POST http://localhost:8000/ingest/bulk \
     -H "X-API-Key: $API_KEY"
   ```

2. Verify schema matches:
   ```python
   # backend/helpers/typesense_helper.py
   # Schema should match EMBEDDING_DIMENSION setting
   ```

### Embedding Model Won't Load

**Symptoms**: Backend crashes on startup with model loading error

**Diagnosis**:
```bash
# Test model loading directly
python -c "
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('yuriyvnv/legal-bge-m3')
print(f'Loaded: {model.model_cards[0].get_model_name()}')
"
```

**Solutions**:
- **Insufficient disk**: Model cache (~1–2 GB) needs space
- **Network error**: Hugging Face unreachable, pre-download model:
  ```bash
  python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('yuriyvnv/legal-bge-m3')"
  ```
- **Wrong dimension**: Ensure `EMBEDDING_DIMENSION` matches model output

---

## Backup & Recovery

### Backup Strategy

**What to backup**:
1. **PDFs** (`storage/pdfs/`)
2. **Typesense data** (`typesense-data/`)
3. **Configuration** (`.env`)

**Backup script**:
```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backups/legal-rag-$(date +%Y%m%d-%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup PDFs
rsync -av storage/pdfs/ "$BACKUP_DIR/pdfs/"

# Backup Typesense data
rsync -av typesense-data/ "$BACKUP_DIR/typesense-data/"

# Backup config
cp .env "$BACKUP_DIR/"

# Compress
tar -czf "$BACKUP_DIR.tar.gz" "$BACKUP_DIR"
rm -rf "$BACKUP_DIR"

# Upload to S3 (optional)
aws s3 cp "$BACKUP_DIR.tar.gz" s3://backups/legal-rag/

# Cleanup old backups (keep last 7)
find /backups -name "legal-rag-*.tar.gz" -mtime +7 -delete

echo "Backup complete: $BACKUP_DIR.tar.gz"
```

**Schedule with cron** (daily at 2 AM):
```bash
0 2 * * * /opt/legal-rag-ai/backup.sh
```

### Recovery Procedure

```bash
# Stop backend
systemctl stop legal-rag-backend

# Restore from backup
BACKUP_FILE="/backups/legal-rag-20260725-020000.tar.gz"
tar -xzf "$BACKUP_FILE" -C /opt/legal-rag-ai

# Restart
systemctl start legal-rag-backend

# Verify
curl http://localhost:8000/health/ready
```

---

## Performance Optimization

### Measure Latency

```bash
# Request timing (includes network)
time curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{"query":"Article 21","top_k":5}'

# Average over 10 requests
for i in {1..10}; do
  curl -s -X POST http://localhost:8000/search \
    -H "Content-Type: application/json" \
    -d '{"query":"Article 21","top_k":5}' > /dev/null
done
```

### Profile Backend

```bash
# CPU profiling (requires py-spy)
pip install py-spy
py-spy record -o profile.svg -- python backend/main.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler backend/main.py
```

### Optimization Tips

| Issue | Solution |
|-------|----------|
| Search latency > 1s | Increase `top_k`, use `keyword` mode only |
| Chat latency > 5s | Reduce `RAG_MAX_CONTEXT_CHARS`, use faster LLM |
| High CPU | Use CPU-efficient embedding model |
| High memory | Reduce batch sizes, use `cpu` device only |
| Typesense slow | Add index, increase server resources |

---

## Monitoring & Alerting

### Prometheus Metrics (Optional)

Add to `backend/core/dependencies.py`:

```python
from prometheus_client import Counter, Histogram, start_http_server

# Metrics
search_latency = Histogram(
    'search_latency_seconds',
    'Search request latency',
    buckets=[0.1, 0.5, 1.0, 2.0, 5.0]
)
search_errors = Counter('search_errors_total', 'Search errors')

# Expose on /metrics
start_http_server(8001)  # Separate port for metrics
```

### Grafana Dashboard

Create a dashboard to visualize:
- Request latency (p50, p95, p99)
- Error rate
- Disk usage
- Memory usage
- Typesense query time

---

## Security Hardening

### Before Production

- [ ] Set strong `API_AUTH_TOKEN` (32+ chars)
- [ ] Set strong `TYPESENSE_API_KEY`
- [ ] Enable HTTPS/TLS (`TYPESENSE_PROTOCOL=https`)
- [ ] Restrict CORS origins (no `*` wildcard)
- [ ] Rotate API keys monthly
- [ ] Enable rate limiting (`RATE_LIMIT_ENABLED=true`)
- [ ] Set `DEBUG=false`
- [ ] Use firewall rules to restrict access
- [ ] Enable audit logging (optional)

### Firewall Rules (UFW on Ubuntu)

```bash
# Allow SSH
ufw allow openssh

# Allow web traffic (only on ports 80, 443)
ufw allow 80/tcp
ufw allow 443/tcp

# Deny all else
ufw default deny incoming
ufw enable

# Allow specific IPs (optional)
ufw allow from 192.168.1.0/24 to any port 8000
```

---

## Maintenance Tasks

### Weekly

```bash
# Check disk usage
df -h

# Check for errors in logs
journalctl -u legal-rag-backend -p err --since "1 week ago"

# Test backup
ls -lh /backups/legal-rag-*.tar.gz
```

### Monthly

```bash
# Rotate API keys (generate new, update clients, revoke old)
# ...

# Review access logs for anomalies
# ...

# Update dependencies
pip list --outdated
```

### Quarterly

```bash
# Full disaster recovery test (restore from backup)
# ...

# Security audit (check for exposed secrets)
# ...

# Performance review (compare metrics)
# ...
```

---

## Scaling

### Horizontal Scaling (Multiple Backends)

```yaml
# docker-compose.prod.yml
version: '3.9'

services:
  backend1:
    image: legal-rag-backend:latest
    environment:
      WORKER_ID: 1
    ports:
      - "8001:8000"

  backend2:
    image: legal-rag-backend:latest
    environment:
      WORKER_ID: 2
    ports:
      - "8002:8000"

  nginx:
    image: nginx:latest
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - backend1
      - backend2
```

### Nginx Load Balancing

```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### Rate Limiting at Scale

Switch from in-process to Redis:

```python
# backend/core/security.py
import redis

redis_client = redis.Redis(host='redis', port=6379)

def rate_limit_redis(request: Request, settings: Settings):
    if not settings.RATE_LIMIT_ENABLED:
        return
    
    key = f"rate_limit:{request.client.host}"
    current = redis_client.incr(key)
    redis_client.expire(key, 60)
    
    if current > settings.RATE_LIMIT_PER_MINUTE:
        raise RateLimitError("Rate limit exceeded")
```

---

## Disaster Recovery Checklist

When things go wrong:

- [ ] Check health endpoints
- [ ] Review logs for errors
- [ ] Verify database connectivity
- [ ] Check disk space
- [ ] Restart affected services
- [ ] If still failing, restore from backup
- [ ] Verify data integrity
- [ ] Run health checks again
- [ ] Document incident

---

## Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Typesense Administration](https://typesense.org/docs/latest/guide/administration/)
- [Systemd Service Management](https://www.freedesktop.org/software/systemd/man/systemctl.html)
- [Docker Compose Best Practices](https://docs.docker.com/compose/production/)
- [Prometheus Monitoring](https://prometheus.io/docs/)

