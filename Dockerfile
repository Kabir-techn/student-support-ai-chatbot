# ---------------------------------------------------------------------------
# AI Student Support Services Chatbot — Backend (FastAPI) Dockerfile
# ---------------------------------------------------------------------------
FROM python:3.12-slim

WORKDIR /app

# System deps needed by faiss / sentence-transformers wheels + build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p documents vectorstore database logs models

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
