# Healthcheck uses 127.0.0.1: in Alpine "localhost" resolves to ::1, where nginx does not listen.
# Static pages served by nginx, which also proxies /api to the backend.
# The browser only ever talks to this container, so there is no CORS setup to get wrong
# and the backend address stays a deployment detail.
FROM nginx:alpine

COPY services/frontend/default.conf.template /etc/nginx/templates/default.conf.template
COPY services/frontend/public /usr/share/nginx/html
# Generates config.js from the environment when the container starts.
COPY --chmod=755 services/frontend/40-leafguard-config.sh /docker-entrypoint.d/40-leafguard-config.sh

EXPOSE 80
HEALTHCHECK --interval=15s --timeout=3s --retries=5 \
  CMD wget -q --spider http://127.0.0.1/ || exit 1
