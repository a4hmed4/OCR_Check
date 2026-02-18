# تشغيل الموديل + API للتحقق من PDF
FROM python:3.11-slim

WORKDIR /app

# تبعيات النظام لـ PaddleOCR و PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model.py main.py ./

ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
