FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

ENV ENABLE_WEB_INTERFACE=true    

WORKDIR /app

# 1. Copy config files first (for caching)
COPY pyproject.toml requirements.txt ./

# 2. Copy src (CRITICAL for pip install)
COPY src ./src

# 3. Install dependencies + package
RUN pip install --upgrade pip && pip install .

# 4. Copy rest of the project (server/, etc.)
COPY . .

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]