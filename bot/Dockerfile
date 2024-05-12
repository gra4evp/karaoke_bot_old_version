FROM python:3.10

# Создание рабочей директории
WORKDIR /app

# Копирование кода бота в контейнер
COPY . /app
COPY ../pyproject.toml /app

# Установка зависимостей
RUN pip install poetry

# Установка зависимостей проекта
RUN poetry install

# Установка ffmpeg и ffprobe
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# Команда для запуска бота
CMD ["python", "main.py"]
