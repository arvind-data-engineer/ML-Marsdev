FROM python:3.10 AS build
WORKDIR /app
COPY ./requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
# CMD ["gunicorn", "--bind", "0.0.0.0:80", "app:create_app()"]
CMD ["python3", "Models/scripts.py"]

FROM python:3.10-slim AS final

# Creating a non-root user with a home directory and creating directories to store various logs
# Create /extra-01/* dir only if json logs path is defined in application to store
# create /app/logs/* directory if these type of logs defined in your application
RUN useradd -ms /bin/bash myuser && \
    mkdir -p /extra-01/logs/delivery_report_python_worker && \
    mkdir -p /app/logs/DeliveryReportPythonWorker/

# Installing packages and changing ownership of directories to 'myuser'
RUN apt-get update && apt-get install libmariadb3 -y && \
    rm -rf /var/lib/apt/lists/* && \
    chown -R myuser:myuser /extra-01 && \
    chown -R myuser:myuser /app

# Switching to the user 'myuser'
USER myuser

WORKDIR /app

# Copying files from the build stage and changing their ownership to 'myuser'
COPY --chown=myuser:myuser --from=build  /usr/local /usr/local
COPY --chown=myuser:myuser . /app
COPY --chown=myuser:myuser --from=build /app/requirements.txt /app/requirements.txt

# Setting the PYTHONPATH environment variable, if required
ENV PYTHONPATH="$PYTHONPATH:/app:/app/src"

CMD ["python3", "Models/script.py"]