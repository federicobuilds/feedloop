ARG BASE_IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404
FROM ${BASE_IMAGE}
COPY pyproject.toml README.md LICENSE /opt/feedloop/
COPY src /opt/feedloop/src
COPY web /opt/feedloop/web
RUN python -m pip install '/opt/feedloop[extract]' && python -c 'import feedloop.remote_worker'
