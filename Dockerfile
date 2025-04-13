FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

WORKDIR /code

# Sistem bağımlılıklarını yükleyin
RUN apt-get update && apt-get install -y \
    libgl1 \
    tesseract-ocr \
    tesseract-ocr-tur \
    python3-pip \
    python3-dev \
    python3-venv

# Sanal ortam oluşturun ve etkinleştirin
RUN python3 -m venv /code/venv
ENV PATH="/code/venv/bin:$PATH"

# Gereksinimlerin yükleneceği requirements.txt dosyasını kopyalayın
COPY ./requirements.txt /code/requirements.txt

# Sanal ortamda pip kullanarak Python bağımlılıklarını yükleyin
RUN pip install --no-cache-dir -r /code/requirements.txt

# Uygulama dosyalarını kopyalayın
COPY ./app /code/app
COPY models/bestLR.pt /code/bestLR.pt
COPY models/finger_detector.pt /code/finger_detector.pt
COPY models/WA_model.pt /code/WA_model.pt

EXPOSE 8000

ENV FLASK_APP=app.server:app

CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]
