# AlfaPetProject
## Introduction
Hi! This is my pet project using data from AlfaBattle competition. Part of the code was taken from open sources of the competition's organizers. 

## Data Description
**Нейросетовой подход к моделированию карточных транзакций (по дебетовым или кредитным картам)**

В нашем расположении дата сет содержащий 1.5 миллиона записей о выдаче кредитных продуктов. Для каждого объекта выборки есть признаковое описание в виде истории клиентских транзакций глубиной в год. Тестовая выборка состоит из выдач в период N дней, тестовая выборка содержит выдачи за последующий K дней. Таким образом, каждый объект выборки представлен в виде многомерного временного ряда, состоящего из вещественных и категориальных признаков. В связи с большим размером все данные были разбиты на 120 файлов формата `parquet`. Целевой переменной в соревновании была бинарная величина, соответствующая флагу дефолта по кредитному продукту. Метрикой для оценки качества решений была выбрана AUC ROC. Данная задача - классический кредитнй скоринг.

**Нейросетевой подход к моделированию транзакций расчетного счета**

Оплата ЖКХ, оплата образования, крупные покупки и другие денежные переводы – это примеры транзакций, которые никак не привязаны к карте клиента, но при этом они ассоциируются с другой банковской сущностью – расчетным счетом. Логичным развитием предыдущей идеи использования карточной транзакционной истории клиента является использование данных, которые содержатся в клиентской истории транзакций расчетного счета. В область применения этих данных входит любая задача, где необходимо по клиенту банка предсказать некоторую целевую переменную.

Возникает вопрос: можно ли построить одну общую модель на транзакциях расчетного счета, включая карточные транзакции, и тем самым избавиться от дополнительной модели на карточных транзакциях? Простой ответ: нет. Карточные транзакции содержат в себе значительно больше признаков по сравнению с произвольной транзакцией расчетного счета. Несмотря на более бедное признаковое описание, нельзя определенно утверждать, что транзакции расчетного счета содержат в себе меньше информации, чем карточные транзакции, просто эта информация менее структурирована и записана в едином текстовом поле. Возникает вопрос: как обрабатывать эти поля? Можно обрабатывать описание как произвольный сырой текст.

**Предобработка описания**
* Замена всех токенов, содержащих цифру или фамилию, на специальный «unknown» токен <unk_tok> или <fio_tok> соответственно.

    Это шумная информация, которую разумно заменить на специальные токены.
* Лемматизация оставшихся токенов.
* Замена непопулярных токенов на специальный токен <unpop_token>.

В дальнейшем обработан достаточно большой корпус описаний и получено 100 миллионов уникальных предобработанных описаний, содержащие 40 тысяч уникальных токенов. Для этого корпуса обучена word2vec-модель, где для каждого токена выучили эмбеддинг размера 50. Набор таких эмбеддингов для каждого описания и есть финальный набор признаков.

**Кластеризация описания**
Реализована модель тематического моделирования с 35 основными и 10 фоновыми темами. В итоге темы получились достаточно интерпритируемые. Таким образом, в описаниях действительно содержится полезная информация, содержащая в себе данные о целях и различных деталях покупки.

**Модель на транзакции**
Эмбеддинг транзакции можно представить как совокупность эмбеддингов всех составляющих ее признаков:
* Для набора категориальных признаков мы используем подход `Entity Embeddings`. 
* Вещественные признаки можно представить как категориальные с помощью процедуры бакетизации.
* Для текстового описания мы уже имеем предобученную word2vec-модель, которая умеет отдельному токену сопоставлять эмбеддинг.

Эмбеддинг описания можно получить из последовательности эмбеддингов его токенов. В базовом случае его можно получить простым покоординатным усреднением эмбеддингов токенов. В более сложном варианте это может быть модель, параметры которой учатся вместе со всей нейронной сетью.

**Основная модель**
Классической архитектурой для обработки последовательностей являются рекуррентные нейронные сети. Будем используем двунаправленные GRU-сети.
На выходе рекуррентной сети мы используем все скрытые состояния. Для их агрегации будем использовать `max-average pooling` по размерности длины последовательности. Впоследствии идет несколько скрытых слоев (`Dropout + Dense + ReLu`) и на выходе мы имеем предсказание модели.

**Применение**
Разумно построить несколько независимых моделей (добавить модели, построенные для карточных транзакций). Использовать всю информацию от моделей в финальном предсказании можно с помощью взвешивания отдельных предсказаний всех 3 моделей. При моделировании мы смотрим на ROC-AUC, но более широкое распространение в банке имеет метрика Джини: 2*ROCAUC - 1. 

## Boosting + RF + LinReg model
Задачу классификации временных рядов можно решать классическим подходом, состоящим из генерации огромного количества признаков с последующим отбором наиболее значимых и стабильных. Признаки можно отобрать на основе важности, полученной методом permutaion importance. Add + Del Feature Selection.
На основе сгенерированных признаков была обучена модель, представляющая из себя стекинг LightBoost + RandomForest + RidgeRegression.

|             | baseline | my model                       |
|-------------|----------|--------------------------------|
| ROC AUC     | 75.1     | 75.2                           |
| description | boosting | feature gen. boosting stacking |

Идеи для дальнейшего развития:
* использовать LSA, LDA, BigARTM для конвертирования категориальных признаков в вектора меньших размерностях

## Data preprocessing for RNN model
Мы имеем дело с последовательностями, один из интуитивных способов работы с ними - использование рекуррентных сетей. RNN позволяют не задумываться над созданием признакового пространство, которое необходимо для хорошего решения в первой модели. Ниже отметим ключевые идеи: 
* С рекурентными моделями нам придется преобразовать изначальное признаковое пространство, представленное в табличном виде, к последовательностям.  `sequence` представляет из себя массив массивов длины `len(features)`, где каждый вложенный массив - значения одного конкретного признака во всех транзакциях клиента. С целью оптимизации решения будем использовать технику `sequence_bucketing`. 
* Векторное представление для каждого категориального признака можно получить, используя стандартный Embedding Layer
* Для реализации `nn.Embedding` нам необходимо, чтобы все признаки были категориальными. С этой целью все вещественные признаки такие , как [`amnt`, `days_before`, `hour_diff`] преобразуем в категорильные с помощью `np.digitize`
* In order to choose the size of embedding for categorical variables use the next formula: `def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))`
* вставить схему
### RNN + Pooling
Усложним рекуррентную модель, заменив GRU на BiGRU. Также добавим сверху этих слоев Pooling. Таким образом, модель стала содержать большее количество параметров и стала сильнее переобучаться.
Для борьбы с переобучением будем использовать L1, L2 регуляризацию. будем использовать Dropout перед полносвязным слоем и SpatialDropout1D после эмбеддинг-слоя
* вставить схему
### CNN+BiGRU + CRF

## На что стоит обратить внимание
    - сдвиг по времени
    - несбалансированные классы
    - размер бакетов и неравномерное разбиение по батчам
    - модель выбирать лучшую по early_stopping. В таком случае есть риск, что мы подгонимся под валидационную выборку, особенно если она не является очень репрезентативной, однако это самый базовый вариант (используем его). Можно делать разные версии ансамблирования, используя веса с разных эпох.
