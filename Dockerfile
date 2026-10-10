FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY requirements.txt constraints-tested.txt ./
RUN pip install -r requirements.txt -c constraints-tested.txt
COPY . .
# Train models at build time so containers start instantly and never fit per session.
RUN python scripts/train.py > /dev/null \
 && useradd --create-home app && chown -R app:app /app
USER app
EXPOSE 8501
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
