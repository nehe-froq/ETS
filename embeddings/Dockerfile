FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    API_BASE_URL=http://api:8000

WORKDIR /app

COPY embeddings/requirements.txt /tmp/embeddings-requirements.txt
RUN pip install --no-cache-dir -r /tmp/embeddings-requirements.txt streamlit requests

COPY embeddings /app/embeddings

EXPOSE 8501

CMD ["streamlit", "run", "embeddings/st-app.py", "--server.port=8501", "--server.address=0.0.0.0"]


