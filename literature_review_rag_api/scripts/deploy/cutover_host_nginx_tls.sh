#!/usr/bin/env bash
set -euo pipefail

# Cut over the host-level Nginx to HTTPS and hide version disclosure.
# Usage:
#   sudo ./scripts/deploy/cutover_host_nginx_tls.sh <domain> [email]

if [[ "${EUID}" -ne 0 ]]; then
  echo "ERROR: Run as root (use sudo)." >&2
  exit 1
fi

DOMAIN="${1:-}"
EMAIL="${2:-}"
WEB_ROOT="${WEB_ROOT:-/home/ubuntu/lit_rag_webapp}"
API_UPSTREAM="${API_UPSTREAM:-http://127.0.0.1:8001}"
SITE_PATH="/etc/nginx/sites-available/lit_rag_webapp"
SITE_LINK="/etc/nginx/sites-enabled/lit_rag_webapp"

if [[ -z "${DOMAIN}" ]]; then
  echo "Usage: sudo $0 <domain> [email]" >&2
  exit 1
fi

if [[ -z "${EMAIL}" ]]; then
  EMAIL="admin@${DOMAIN}"
fi

echo "INFO: Installing runtime dependencies (nginx/certbot)..."
apt-get update -y
apt-get install -y nginx certbot

if [[ ! -d "${WEB_ROOT}" ]]; then
  echo "ERROR: Web root not found at ${WEB_ROOT}" >&2
  exit 1
fi

echo "INFO: Requesting/refreshing Let's Encrypt certificate for ${DOMAIN}..."
systemctl stop nginx || true
certbot certonly \
  --standalone \
  --non-interactive \
  --agree-tos \
  --email "${EMAIL}" \
  --keep-until-expiring \
  -d "${DOMAIN}"
systemctl start nginx

if [[ ! -s "/etc/letsencrypt/live/${DOMAIN}/fullchain.pem" || ! -s "/etc/letsencrypt/live/${DOMAIN}/privkey.pem" ]]; then
  echo "ERROR: Certificate files are missing for ${DOMAIN}" >&2
  exit 1
fi

echo "INFO: Enforcing server token masking..."
if grep -qE '^[[:space:]]*#[[:space:]]*server_tokens off;' /etc/nginx/nginx.conf; then
  sed -i 's/^[[:space:]]*#[[:space:]]*server_tokens off;/    server_tokens off;/' /etc/nginx/nginx.conf
elif ! grep -qE '^[[:space:]]*server_tokens off;' /etc/nginx/nginx.conf; then
  sed -i '/^[[:space:]]*http[[:space:]]*{/a\    server_tokens off;' /etc/nginx/nginx.conf
fi

echo "INFO: Writing HTTPS-only site configuration..."
cat > "${SITE_PATH}" <<EOF
server {
    listen 80;
    listen [::]:80;
    server_name ${DOMAIN};

    location /.well-known/acme-challenge/ {
        root /var/www/html;
    }

    location / {
        return 301 https://\$host\$request_uri;
    }
}

server {
    listen 80 default_server;
    listen [::]:80 default_server;
    server_name _;
    return 301 https://\$host\$request_uri;
}

server {
    listen 443 ssl http2;
    listen [::]:443 ssl http2;
    server_name ${DOMAIN};

    root ${WEB_ROOT};
    index index.html;
    client_max_body_size 100m;

    ssl_certificate /etc/letsencrypt/live/${DOMAIN}/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/${DOMAIN}/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_prefer_server_ciphers off;
    ssl_session_timeout 1d;
    ssl_session_cache shared:SSL:10m;
    ssl_session_tickets off;

    add_header Strict-Transport-Security "max-age=63072000; includeSubDomains" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;
    add_header Permissions-Policy "camera=(), microphone=(), geolocation=(), payment=(), usb=()" always;
    add_header Content-Security-Policy "default-src 'self'; frame-ancestors 'none'; base-uri 'self'; form-action 'self'" always;
    add_header X-XSS-Protection "0" always;

    location = /api/healthz {
        proxy_pass ${API_UPSTREAM}/healthz;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /api/ {
        proxy_pass ${API_UPSTREAM};
        proxy_http_version 1.1;
        proxy_set_header Upgrade \$http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host \$host;
        proxy_set_header X-Real-IP \$remote_addr;
        proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto \$scheme;
        proxy_read_timeout 600;
        proxy_send_timeout 600;
    }

    location = /health {
        allow 127.0.0.1;
        allow ::1;
        deny all;
        proxy_pass ${API_UPSTREAM}/health;
        proxy_http_version 1.1;
        proxy_set_header Host \$host;
        proxy_set_header X-Forwarded-Proto \$scheme;
    }

    location /assets/ {
        add_header Cache-Control "no-store";
        try_files \$uri =404;
    }

    location = /index.html {
        add_header Cache-Control "no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0";
        try_files \$uri =404;
    }

    location / {
        try_files \$uri /index.html;
    }
}
EOF

ln -sf "${SITE_PATH}" "${SITE_LINK}"

echo "INFO: Installing cert renew hooks for standalone mode..."
install -d /etc/letsencrypt/renewal-hooks/pre /etc/letsencrypt/renewal-hooks/post
cat > /etc/letsencrypt/renewal-hooks/pre/10-stop-nginx.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
systemctl stop nginx || true
EOF
cat > /etc/letsencrypt/renewal-hooks/post/10-start-nginx.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
systemctl start nginx
EOF
chmod +x /etc/letsencrypt/renewal-hooks/pre/10-stop-nginx.sh /etc/letsencrypt/renewal-hooks/post/10-start-nginx.sh
systemctl enable --now certbot.timer || true

echo "INFO: Validating and reloading nginx..."
nginx -t
systemctl reload nginx

echo "DONE: HTTPS cutover complete for ${DOMAIN}."
echo "Next checks:"
echo "  curl -I http://${DOMAIN}/api/healthz"
echo "  curl -I https://${DOMAIN}/api/healthz"
