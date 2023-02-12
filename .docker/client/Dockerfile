FROM node:18

WORKDIR /app

COPY .docker/client/entrypoint.sh /opt/entrypoint.sh

EXPOSE 3000

ENTRYPOINT ["bash", "/opt/entrypoint.sh"]