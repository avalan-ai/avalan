FROM python:3.11-slim-bookworm@sha256:2e32f7d302adc1c37428355c1e646897c0c53f4fd60b6a551245fb90ee129f91

ARG TARGETARCH

COPY container_wheels/requirements.txt /wheelhouse/requirements.txt
COPY container_wheels/cffi-2.0.0-cp311-cp311-manylinux2014_aarch64.manylinux_2_17_aarch64.whl /wheelhouse/
COPY container_wheels/cffi-2.0.0-cp311-cp311-manylinux2014_x86_64.manylinux_2_17_x86_64.whl /wheelhouse/
COPY container_wheels/cryptography-48.0.1-cp311-abi3-manylinux2014_aarch64.manylinux_2_17_aarch64.whl /wheelhouse/
COPY container_wheels/cryptography-48.0.1-cp311-abi3-manylinux2014_x86_64.manylinux_2_17_x86_64.whl /wheelhouse/
COPY container_wheels/pycparser-2.23-py3-none-any.whl /wheelhouse/

RUN case "$TARGETARCH" in \
        amd64) test "$(uname -m)" = "x86_64" ;; \
        arm64) test "$(uname -m)" = "aarch64" ;; \
        *) exit 1 ;; \
    esac \
    && pip install --disable-pip-version-check --no-cache-dir --no-compile \
        --no-deps --no-index --find-links /wheelhouse --only-binary=:all: \
        --require-hashes \
        --requirement /wheelhouse/requirements.txt \
    && rm -rf /wheelhouse
