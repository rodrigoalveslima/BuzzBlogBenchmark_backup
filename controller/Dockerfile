# Copyright (C) 2020 Georgia Tech Center for Experimental Research in Computer
# Systems

# Define base configuration.
FROM ubuntu:20.04
MAINTAINER ral@gatech.edu
WORKDIR /usr/local/

# Environment variables.
ENV description null
ENV docker_hub_username null
ENV docker_hub_password null
ENV system_conf /usr/local/etc/BuzzBlogBenchmark/system.yml
ENV workload_conf /usr/local/etc/BuzzBlogBenchmark/workload.yml

# Install software dependencies.
RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    wget \
    lsb-release \
    unzip \
    gnupg2 \
    python3-pip

# Add PostgreSQL repository.
RUN wget --quiet -O - https://www.postgresql.org/media/keys/ACCC4CF8.asc | apt-key add - \
  && echo "deb http://apt.postgresql.org/pub/repos/apt/ `lsb_release -cs`-pgdg main" | tee  /etc/apt/sources.list.d/pgdg.list \
  && apt-get update

# Install PostgreSQL client 13.
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y postgresql-client-13

# Download BuzzBlog.
RUN wget --quiet -O /var/tmp/BuzzBlogApp.zip https://github.com/rodrigoalveslima/BuzzBlogApp/archive/main.zip \
  && unzip -d /opt /var/tmp/BuzzBlogApp.zip \
  && mv /opt/BuzzBlogApp-main /opt/BuzzBlogApp

# Copy source code.
COPY src /usr/local/src/BuzzBlogBenchmark

# Install Python dependencies.
RUN pip3 install -r /usr/local/src/BuzzBlogBenchmark/requirements.txt

# Install Python dependencies for analysis.
RUN pip3 install -r /usr/local/src/BuzzBlogBenchmark/analysis/requirements.txt

# Start the experiment.
CMD ["/bin/bash", "-c", "python3 -u /usr/local/src/BuzzBlogBenchmark/run_experiment.py --description \"$description\" --system_conf \"$system_conf\" --workload_conf \"$workload_conf\" --docker_hub_username \"$docker_hub_username\" --docker_hub_password \"$docker_hub_password\""]
