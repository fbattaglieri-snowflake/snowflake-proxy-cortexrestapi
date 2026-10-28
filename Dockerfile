FROM python:3.14-alpine@sha256:f6a589d43c42b9e7f7dc67a12d37132491f362859a5d750607710cc56da3bc72
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /opt/proxy
COPY requirements.runtime.lock /opt/proxy/requirements.runtime.lock
RUN python -m pip install --no-cache-dir -r requirements.runtime.lock && python -m pip check && python -m pip uninstall -y pip
COPY src/llm_proxy_standard /opt/proxy/llm_proxy_standard
USER 1000:1000
EXPOSE 8080
ENTRYPOINT ["python", "-m", "uvicorn", "llm_proxy_standard.app:environment", "--factory", "--host", "0.0.0.0", "--port", "8080", "--no-access-log", "--timeout-graceful-shutdown", "605", "--h11-max-incomplete-event-size", "16384"]