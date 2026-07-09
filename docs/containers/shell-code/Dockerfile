FROM alpine:3.22

LABEL org.opencontainers.image.title="avalan-shell-code"
LABEL org.opencontainers.image.description="Read-only shell tool image for Avalan code exploration."

RUN apk add --no-cache \
    bash \
    coreutils \
    file \
    findutils \
    gawk \
    git \
    jq \
    ripgrep \
    sed

WORKDIR /workspace
USER 1000:1000

CMD ["sh"]
