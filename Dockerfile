FROM python:3.10 as build
WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

FROM python:3.10-slim as final
WORKDIR /app
COPY . /app
COPY --from=build /usr/local /usr/local
EXPOSE 5000
CMD ["python3", "Models/script.py"]