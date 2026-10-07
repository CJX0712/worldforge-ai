# 多阶段构建：builder 装依赖，runtime 只带运行期文件。
# 核心运行时依赖为空，因此最终镜像只有源码与标准库。
FROM python:3.13-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip build \
 && python -m build --wheel --outdir /dist

FROM python:3.13-slim AS runtime
LABEL org.opencontainers.image.title="worldforge-ai" \
      org.opencontainers.image.authors="晨星" \
      org.opencontainers.image.licenses="MIT"
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app
COPY --from=builder /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -rf /tmp/*.whl
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).close()"
CMD ["python", "-m", "worldforge.serving.app"]
