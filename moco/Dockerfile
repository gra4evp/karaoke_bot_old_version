FROM python:3.6.10

# Создание рабочей директории
WORKDIR /app

# Копирование кода модели в контейнер
COPY . .

# Установка зависимостей
RUN pip install numpy
# Установка зависимостей
RUN pip install Flask
# Установка зависимостей
RUN pip install librosa
## Установка зависимостей
#RUN pip install torch
#RUN pip install torchvision

# Установка faiss-cpu (если вам не нужна GPU)
#RUN apt-get update && \
#    apt-get install -y libopenblas-base libomp-dev && \
#    pip install faiss-cpu
