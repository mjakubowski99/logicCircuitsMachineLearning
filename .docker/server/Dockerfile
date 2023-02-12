FROM python:3.8

WORKDIR /app

COPY .docker/server/requirements.txt /opt/requirements.txt

COPY .docker/server/entrypoint.sh /opt/entrypoint.sh

RUN chown $USER:$USER /opt/entrypoint.sh

ENTRYPOINT ["bash", "/opt/entrypoint.sh"]