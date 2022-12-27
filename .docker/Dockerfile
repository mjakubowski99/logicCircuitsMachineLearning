FROM python:3.8

# Ustawienie katalogu roboczego
WORKDIR /app

# Kopiowanie pliku requirements.txt do katalogu roboczego
COPY .docker/requirements.txt .

COPY .docker/entrypoint.sh /opt/entrypoint.sh

RUN chown $USER:$USER /opt/entrypoint.sh

# Uruchomienie aplikacji
ENTRYPOINT ["bash", "/opt/entrypoint.sh"]