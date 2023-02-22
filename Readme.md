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

# Opis biblioteki
Wszystkie pliki docelowej biblioteki znajdują się w katalogu server/src

## Wstępne przetwarzanie danych
Wszystkie pliki związane z wstępnym przetwarzaniem danych znajdują się w folderze preprocessing.
Znajdują się tu następujące konwerterty: konwerter obiektów na typ integer,
konwerter wartości zmiennoprzecinkowych na liczby całkowite oraz konwerter liczb całkowitych na wartości
binarne(zastosowano tzw. one hot encoding)

## Generatory formuł logicznych
W katalogu classifier/formula_creators znajdują się pliki związane z generowaniem formuł.
Formuły pozytywne jak i negatywne są generowane w liczbie i wymiarach określonych w parametrach
funkcji. Generator unika powtórzeń podczas losowania.

## Mierniki wydajności formuł
W katalogu classifier/formula_checkers znajdują się pliki, który są związane z mierzeniem poprawności oraz
wydajności formuł. W pliku get_score.py zdefiniowane są funkcje, które sprawdzają skuteczność wygenerowanych
formuł na zbiorze testowym. Natomiast w pliku best_formulas.py są zdefiniowane funkcje, których zadaniem jest
na podstawie wyników obliczonych przez poprzednio zdefiniowane funkcje wybrać najlepsze formuły, czyli takie
które mają 2 razy więcej poprawnych odpowiedzi

## Model agregujący wszystkie funkcje
W klasie LogicClassifier znajdującej się w pliku classififer/LogicClassifier jest zdefiniowany cały proces uczenia.
Wywołanie metody fit z parametrami treningowymi X i y powoduje przeprowadzenie całego procesu uczenia maszynowego 
z uwzględnieniem stałych dla algorytmu zdefiniowanych w konstrukturze, które domyślnie dostają optymalne wartości, które
można dowolnie zmieniać. Wykonuje się zadana liczba cykli treningowych, które w rezultacie dają nam zbiory formuł, które
są w stanie głosować na temat przynależności kolumny celu do danej klasy. Natomiast wywołanie metody score sprawdza jak 
skuteczne na zbiorze treningowym są formuły logiczne.

Przykład z kodu pokazujący jak wygląda przeprowadzenia procesu uczenia maszynowego oraz pobranie wyników:
```
clf = LogicClassifier()
clf.fit(X_train, y_train)

print("Wynik dla danych testowych: ", clf.score(X_test,y_test))
```
