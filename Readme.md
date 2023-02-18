Wymagania
 - Docker
 - docker-compose

Aby postawić środowisko należy wykonać polecenie
```
docker-compose up -d
```  

Aby uruchomić program wykonujący proces uczenia się należy:
```
docker exec -it server /bin/bash 
```  

Następnie:
```
cd src
python main.py
```
Powyższe instrukcje uruchomią program, który prezentuje działanie biblioteki