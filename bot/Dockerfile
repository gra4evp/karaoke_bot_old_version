FROM python:3.10

# Создание рабочей директории
WORKDIR /bot

# Установка зависимостей
RUN pip install aiogram==2.25.1 yt-dlp==2023.3.4 pymysql==1.0.3 sqlalchemy==2.0.10

# Установка ffmpeg и ffprobe
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean

# Копирование кода бота в контейнер
COPY . .
