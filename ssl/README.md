# SSL Certificates

HTTPS is **opt-in**. By default, nginx only listens on port 80.

To enable HTTPS:

1. Generate a self-signed certificate (development only):

   ```bash
   mkdir -p ssl
   openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
     -keyout ssl/key.pem \
     -out ssl/cert.pem \
     -subj "/CN=localhost"
   ```

2. Copy `docker-compose.override.yml.example` to `docker-compose.override.yml`:

   ```bash
   cp docker-compose.override.yml.example docker-compose.override.yml
   ```

   Docker Compose automatically merges this file, enabling the 443 port
   mapping and the ssl volume mount.

3. For production, replace the self-signed certificates with certificates
   issued by a trusted CA (e.g. Let's Encrypt via certbot).

> **Note**: `ssl/cert.pem` and `ssl/key.pem` are git-ignored.
> Never commit private keys.
