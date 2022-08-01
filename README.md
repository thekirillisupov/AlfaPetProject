# AlfaPetProject
## Introduction
Hi! This is my pet project using data from AlfaBattle competition. Part of the code was taken from open sources of the competition's organizers. 

## Data Description
В нашем расположении дата сет содержащий 1.5 миллиона записей о выдаче кредитных продуктов. Для каждого объекта выборки есть признаковое описание в виде истории клиентских транзакций глубиной в год. Тестовая выборка состоит из выдач в период N дней, тестовая выборка содержит выдачи за последующий K дней. Таким образом, каждый объект выборки представлен в виде многомерного временного ряда, состоящего из вещественных и категориальных признаков. В связи с большим размером все данные были разбиты на 120 файлов формата `parquet`. Целевой переменной в соревновании была бинарная величина, соответствующая флагу дефолта по кредитному продукту. Метрикой для оценки качества решений была выбрана AUC ROC. 

## Boosting + RF + LinReg model
Задачу классификации временных рядов можно решать классическим подходом, состоящим из генерации огромного количества признаков с последующим отбором наиболее значимых и стабильных. Признаки можно отобрать на основе важности, полученной методом permutaion importance. Add + Del Feature Selection.
На основе сгенерированных признаков была обучена модель, представляющая из себя стекинг LightBoost + RandomForest + RidgeRegression.

|             | baseline | my model                       |
|-------------|----------|--------------------------------|
| ROC AUC     | 75.1     | 75.2                           |
| description | boosting | feature gen. boosting stacking |

Идеи для дальнейшего развития:
* использовать LSA, LDA, BigARTM для конвертирования категориальных признаков в вектора меньших размерностях

## RNN model 
Мы имеем дело с последовательностями, один из интуитивных способов работы с ними - использование рекуррентных сетей. RNN позволяют не задумываться над созданием признакового пространство, которое необходимо для хорошего решения в первой модели. 
* С рекурентными моделями нам придется преобразовать изначальное признаковое пространство, представленное в табличном виде, к последовательностям.  `sequence` представляет из себя массив массивов длины `len(features)`, где каждый вложенный массив - значения одного конкретного признака во всех транзакциях клиента. С целью оптимизации решения будем использовать технику `sequence_bucketing`. 
* Векторное представление для каждого категориального признака можно получить, используя стандартный Embedding Layer
* Для реализации `nn.Embedding` нам необходимо, чтобы все признаки были категориальными. С этой целью все вещественные признаки такие , как [`amnt`, `days_before`, `hour_diff`] преобразуем в категорильные с помощью `np.digitize`
* In order to choose the size of embedding for categorical variables use the next formula: `def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))`

## RNN + CNN

## 

## На что стоит обратить внимание
    - сдвиг по времени
    - несбалансированные классы
