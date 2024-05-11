FROM python:3.10

# Установка зависимостей
RUN pip install poetry

# Создание рабочей директории
WORKDIR /app

# Копирование зависимостей и файлов проекта
COPY poetry.lock pyproject.toml /app/

# Установка зависимостей проекта
RUN poetry install

# Копирование кода бота в контейнер
COPY . /app

# Команда для запуска бота
CMD ["poetry", "run", "python", "bot.py"]
