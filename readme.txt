If you don't have Nvidia-container-toolkit , CUDA , cudnn on computer you can change the dockerfile with this:

# Python 3.11 tabanlı bir imaj kullan
FROM python:3.11

# Install system dependencies including Tesseract and Turkish language pack
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    tesseract-ocr \
    tesseract-ocr-tur

# Çalışma dizinini /code olarak ayarla
WORKDIR /code

# Gereksinimlerin yükleneceği requirements.txt dosyasını kopyalayın
COPY ./requirements.txt /code/requirements.txt

# Gereksinimleri yükleyin
RUN pip install --no-cache-dir -r /code/requirements.txt

# Tüm app klasörünü ve model dosyalarını kopyalayın
COPY ./app /code/app
COPY models/bestLR.pt /code/bestLR.pt
COPY models/finger_detector.pt /code/finger_detector.pt
COPY models/WA_model.pt /code/WA_model.pt

# Flask uygulamasını dinlemek için gerekli portu açın
EXPOSE 8000

# Flask uygulama değişkeni
ENV FLASK_APP=app.server:app

# Uygulamayı çalıştırın
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]

