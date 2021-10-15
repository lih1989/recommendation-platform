FROM python:3.9
## Локальная папка с кодом из проекта
# VOLUME /application
# Создаю папку для приложения
RUN mkdir /application
# Указываю рабочую папку
WORKDIR /application
## Копируем все файлы проекта в контейнер - не нужно ибо используются volume при запуске
#COPY app/ /application
## Копирую файл с зависимостями
COPY requirements.txt /application/requirements.txt
# Устанавливаю зависимости
RUN pip install -r /application/requirements.txt
#RUN pip install Flask
#RUN pip install matplotlib
#RUN pip install tensorflow-datasets
#RUN pip install tensorflow-recommenders
EXPOSE 6789
# COPY app/main.py /main.py
# RUN chmod +x /application/main.py
# CMD python /application/main.py
# CMD /bin/bash
CMD ["python", "./main.py"]