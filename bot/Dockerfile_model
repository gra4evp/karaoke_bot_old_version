FROM python:3.6.10

# Установка зависимостей
RUN pip install torch torchvision librosa numpy

# Установка faiss-cpu (если вам не нужна GPU)
RUN apt-get update && \
    apt-get install -y libopenblas-base libomp-dev && \
    pip install faiss-cpu

# Добавление кода модели в контейнер
COPY model.py /app/
WORKDIR /app

# Команда для запуска модели
CMD ["python", "model.py"]
