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

COPY ./requirements.txt /code/requirements.txt

# RUN pip install -r requirements.txt

# Creating folders, and files for a project:
COPY . /app

# Expose port
EXPOSE 8080

# Run the application:
CCMD ["uvicorn", "app.main:app", "--port", "8080", "--reload"]