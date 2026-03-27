FROM python:3.12-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    SC2PATH=/root/StarCraftII

USER root
WORKDIR /root

RUN dpkg --add-architecture i386 \
    && apt-get update \
    && apt-get install --assume-yes --no-install-recommends --no-show-upgraded \
        wget \
        unzip \
        git \
        make \
        gcc \
        g++ \
        libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

RUN wget -q "http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip" \
    && unzip -q -P iagreetotheeula SC2.4.10.zip \
    && rm SC2.4.10.zip \
    && rm -rf "$SC2PATH/Maps" \
    && ln -s /workspace/examples/maps "$SC2PATH/Maps" \
    && ln -s "$SC2PATH/Maps" "$SC2PATH/maps"

RUN python -m pip install --no-cache-dir \
    numpy \
    scipy \
    burnysc2 \
    matplotlib \
    loguru

RUN printf '%s\n' \
    '#!/bin/sh' \
    'set -eu' \
    'mkdir -p /workspace/examples/data /workspace/examples/maps' \
    'python -m pip install --no-cache-dir --no-deps -e /workspace' \
    'exec python /workspace/examples/train_sc2_bot.py' \
    > /usr/local/bin/run-train-sc2-bot \
    && chmod +x /usr/local/bin/run-train-sc2-bot

WORKDIR /workspace
ENTRYPOINT ["run-train-sc2-bot"]
