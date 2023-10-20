#make with poetry, python 3.9
FROM python:3.9-slim-buster

# Install Poetry
RUN pip install poetry

# Copy only requirements to cache them in docker layer
WORKDIR /app
COPY poetry.lock pyproject.toml /app/

# Project initialization:
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

COPY ./requirements.txt /code/requirements.txt

# RUN pip install -r requirements.txt

# Creating folders, and files for a project:
COPY . /app

# Expose port
EXPOSE 8000

# Run the application:
CMD uvicorn app.main:app --port=8000 --host=0.0.0.0
