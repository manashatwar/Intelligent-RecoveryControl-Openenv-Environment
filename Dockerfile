FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

<<<<<<< HEAD
# Copy dependency manifests + source (needed for pip install -e .)
COPY requirements.txt pyproject.toml ./
COPY src/ ./src/

# Install all dependencies + register the irce package
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install -e .

# Copy remaining project files (server/, inference.py, etc.)
=======
# Install package and dependencies in one layer.
# Copying pyproject.toml first leverages Docker layer caching.
COPY pyproject.toml requirements.txt ./
RUN pip install --upgrade pip && pip install .

# Copy source last so code changes don't bust the dependency cache.
>>>>>>> 9b34442430b98e9efb51d76a40f653bc7bde7b4d
COPY . .

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
