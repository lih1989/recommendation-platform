#### Собрать образ на основе Dockerfile
```bash
# cd _docker
```
```bash
# sudo docker build . -t recommendation-server
```
#### Запустить сервер на собранном образе
```bash
# cd ..
```
```bash
# sudo docker run --rm -p 6789:6789 -v "$PWD"/app:/application recommendation-server
```
#### Запустить созданный образ и зайти в консоль
```bash
# sudo docker run -it --mount type=bind,source="$(pwd)"/app,target=/application --rm recommendation-server bash
```