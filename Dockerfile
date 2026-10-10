# ai-service (VPS): chat RAG, thực đơn, kiểm duyệt, Vision. stt-service dùng Dockerfile.stt (Cloud Run).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app
COPY knowledge/nutrition ./knowledge/nutrition
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import os,urllib.request as u;r=u.Request('http://localhost:'+os.environ.get('PORT','8080')+'/health',headers={'X-Internal-Api-Key':os.environ.get('INTERNAL_API_KEY','')});u.urlopen(r,timeout=4)"
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8080}"]
