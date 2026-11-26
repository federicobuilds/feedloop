ARG BASE_IMAGE=runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404@sha256:3e874356857adfa3e8faa3fd913b65bd127f77a0fe2e489513e7775e1c1e16b1
FROM ${BASE_IMAGE}
COPY pyproject.toml README.md LICENSE /opt/feedloop/
COPY src /opt/feedloop/src
COPY web /opt/feedloop/web

# Keep the GPU packages matched to the CUDA build already installed in the base.
RUN python -m pip list --format=freeze | grep -E '^(torch|torchvision|torchaudio)==' > /tmp/feedloop-torch-constraints.txt \
    && python -m pip install --no-cache-dir -c /tmp/feedloop-torch-constraints.txt '/opt/feedloop[extract]' \
    && rm /tmp/feedloop-torch-constraints.txt \
    && python -c 'import feedloop.remote_worker'
