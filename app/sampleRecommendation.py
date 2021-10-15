from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# схема инструкция типов колонок
columnsTypeSchema = {
    "numeric": ["lat", "lon"],
    "category": ["type", "region", "locality"],
    "text": ["name"]
}

# Приведение в порядок полученных данных
def cleanData(dataFrame, typeSchema):
    print("Очитска колонок и вставка дефолтных значений")
    print(f"cleanData - Строк: {dataFrame.shape[0]}; Колонок: {dataFrame.shape[1]};")
    print(f"cleanData - Колонки: {list(dataFrame.columns)}")
    print(f"cleanData - Типы колонок: {typeSchema}")
    # ЧИСЛА - Очищаю, заменяю значения и превращаю в числовые значения
    for column in typeSchema["numeric"]:
        dataFrame[column] = dataFrame[column].fillna(0).replace("Not found", 0)
        dataFrame[column] = dataFrame[column].astype('float64')

    # КАТЕГОРИИ - Пустые значения получают категорию [undefined]
    for column in typeSchema["category"]:
        dataFrame[column] = dataFrame[column].fillna('undefined')

    # Тескты - заменяю на пустую строку
    for column in typeSchema["text"]:
        dataFrame[column] = dataFrame[column].fillna('')

    return dataFrame


# Изменить данные под обучение
def transformScaleData(dataFrame, typeSchema):
    print("Нормализация - векторизация колонок")
    # Удаляю колонку с ИД
    dataFrame.drop("id", axis=1, inplace=True)

    # метки для категориальных колонок
    le = LabelEncoder()
    for column in typeSchema["category"]:
        dataFrame[column] = le.fit_transform(dataFrame[column])

    # нормализация для числовых колонок
    sc = StandardScaler()
    dataFrame[typeSchema["numeric"]] = sc.fit_transform(dataFrame[typeSchema["numeric"]])

    # Векторизация текстовых данных
    framesList = []
    for column in typeSchema["text"]:
        column_vec = TfidfVectorizer(min_df=2, ngram_range=(1, 3), stop_words=[])
        vecFrameData = column_vec.fit_transform(dataFrame[column])
        framesList.append(vecFrameData)
        # удаляю колонку из общего фрейма данных
        dataFrame.drop(column, axis=1, inplace=True)

    # Добавляю осташиеся колонки к списку текстовых фреймов
    framesList.append(dataFrame)

    # Объединяю массив фреймов в стек для поиска ближайших похожих
    trainStack = hstack(framesList)

    # вычисляю ближайшие похожие элементы
    nearestNeighbors = cosine_similarity(trainStack, trainStack)
    return nearestNeighbors

# Вычислить ближайших соседей
def nearestNeighborsCalc ():
    nearestNeighborsData = transformScaleData(rawDataFrame.copy(), columnsTypeSchema)
    global nearestNeighborsResult
    nearestNeighborsResult = {}
    for idx, row in rawDataFrame.iterrows():
        # Вытянуть ИД соседних элементов и отсортировать по возрастанию
        nearestNeighborsIndices = nearestNeighborsData[idx].argsort()[:-100:-1]
        # Сопоставить ИД элементов с результатом значения близости соседства
        nearestNeighborsItems = [(nearestNeighborsData[idx][i], rawDataFrame['id'][i]) for i in nearestNeighborsIndices]
        # Записать результат для будущих сопоставлений
        nearestNeighborsResult[row['id']] = nearestNeighborsItems[1:]

    print("nearestNeighborsCalc - Готово!")

# получить данные
def getData ():
    global rawDataFrame
    global date
    df = pd.read_csv("data/tourist_attractions.csv")
    rawDataFrame = cleanData(df, columnsTypeSchema)
    date = datetime.now()
    nearestNeighborsCalc()
    return rawDataFrame, date


def predictRecommends(item_id, count):
    print("Запрос " + str(count) + " рекомендаций для элемента:")
    print(rawDataFrame.loc[rawDataFrame['id'] == item_id])
    print("-------")
    recs = nearestNeighborsResult[item_id][:count]
    result = []
    for rec in recs:
        itemName = rawDataFrame.loc[rawDataFrame['id'] == rec[1]]["name"].tolist()[0].split(' - ')[0]
        print("Рекоммендация: ID=" + str(rec[1]) + "; " + itemName + " (Близость:" + str(rec[0]) + ")")
        result.append(getById(rec[1]))
    return result

# получение списка объектов по индексам
def getIds (start=0, end=20):
    return rawDataFrame[start:end].to_dict(orient='records')

# Получение оригинального объекта по ИД
def getById(item_id = 1):
    return rawDataFrame.loc[rawDataFrame['id'] == item_id].to_dict(orient='records')[0]