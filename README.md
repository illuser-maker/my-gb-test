# MyGradientBoosting
Пример тупого бустинга для задачи регрессии.

Весь код был сделан за несколько часов. Алгоритм крайне тупой. 
Используется обычная композиция алгоритмов, без коэфициентов перед каждым базовым алогритмом.

## 1 Класс MyGradientBoostingRegressor
Данный класс собственно и является реализацией бустинга. В качестве функционала качества используется MSE, в том числе и в качестве метрики.
Описание всех параметров есть в докстринге класса. Интерфейс простой, склерновский, т.е. есть функции fit и predict (ну и fit_predict).
Есть поддержка базовых гиперпараметров деревьев и learning rate. 

Функционал качества и метрика могут быть пользовательскими. Метрика - просто функция формата metric_func(y_true, y_pred). Функционал качества - это объект, 
который должен поддерживать метод derivative(self, approxes, targets), который возвращает список первых анти-производных для предсказаний и таргетов.
В принципе, можно было и функцией сделать, но сделал объектом, потому как такой интерфейс в CatBoost.

## 2 Класс MyMSELoss
Собственно, класс MSE функционала, который возвращает первые производные MSE.

## 3 Использование
Пример использования есть в ноутбуке GB_Usage_Example. 

## 4 Комментарий от меня
Т.к. делалось это в качестве тествого, просто показать, как я кодю, многое я просто не сделал. Добавить можно оооооочень много, но я сделал просто базовую версию.
Странно было бы реализовывать хороший бустинг. Что конкретно может быть добавлено, есть в следующем пункте.

## 5 Сравнение с существующими бустингами
В моем бустинге куча всего, что можно добавить, а именно:
* Большее количество метрик
* Веса при базовых алгоритмах
* Регулировка этих весов с учетом того, что базовый алгоритм - дерево
* Большее количество гиперпараметров
* Критерий остановки по значчению метрики на тесте.
* Оптимизация по второй производной
* Параллельность и GPU
* Поддержка задачи классификации
* Нормальные исключения для ошибок

В общем-то, всем этим обладают другие бустинги (LightGBM и CatBoost). 