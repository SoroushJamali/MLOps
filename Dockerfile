FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install cron
RUN apt-get update && apt-get install -y cron

# Copy cron job file
COPY cronjob /etc/cron.d/cronjob

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/cronjob

# Apply cron job
RUN crontab /etc/cron.d/cronjob

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Start cron and log output
CMD cron && tail -f /var/log/cron.log
