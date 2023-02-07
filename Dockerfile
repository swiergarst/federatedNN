# basic python3 image as base
FROM harbor2.vantage6.ai/infrastructure/algorithm-base:legacy

# This is a placeholder that should be overloaded by invoking
# docker build with '--build-arg PKG_NAME=...'
ARG PKG_NAME="v6_simpleNN_py"

#ENV PYTHONPATH="/home/swier/miniconda3/envs/vantage6/bin/python"
# install federated algorithm
#RUN pip install -e ./setup.py

COPY . /app

#FROM python 

RUN pip install /app

ENV PKG_NAME=${PKG_NAME}

# Tell docker to execute `docker_wrapper()` when the image is run.
CMD python -c "from vantage6.tools.docker_wrapper import docker_wrapper; docker_wrapper('${PKG_NAME}')"
