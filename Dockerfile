FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       gcc \
       postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better Docker layer caching)
COPY requirements.txt /app/requirements.txt
RUN python -m pip install --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Copy SSL certs to /certs (referenced explicitly by DB connection code).
# Must be done BEFORE switching to non-root user, and made world-readable
# so appuser can access them at runtime.
COPY certs/ /certs/
RUN chmod -R a+r /certs

# Create a non-root user and fix permissions on /app
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8012

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8012/docs').read()" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8012"]