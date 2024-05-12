FROM python:3.6.10

# Создание рабочей директории
WORKDIR /app

# Копирование кода модели в контейнер
COPY ./moco .

# Установка зависимостей
RUN pip install torch torchvision librosa numpy

# Установка faiss-cpu (если вам не нужна GPU)
RUN apt-get update && \
    apt-get install -y libopenblas-base libomp-dev && \
    pip install faiss-cpu

# Команда для запуска модели
CMD ["python", "model.py"]
