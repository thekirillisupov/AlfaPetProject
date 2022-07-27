# AlfaPetProject
## Description
Hi! This is my pet project using data from AlfaBattle competition. Part of the code was taken from open sources of the competition's organizers. 

## Boosting + RF + LinReg model

|             | baseline | my model                       |
|-------------|----------|--------------------------------|
| ROC AUC     | 75.1     | 75.2                           |
| description | boosting | feature gen. boosting stacking |

## RNN model 
Мы имеем дело с последовательностями, один из интуитивных способов работы с ними - использование рекуррентных сетей. RNN позволяют не задумываться над созданием признакового пространство, которое необходимо для хорошего решения в первой модели. 
* С рекурентными моделями нам придется преобразовать изначальное признаковое пространство, представленное в табличном виде, к последовательностям.  `sequence` представляет из себя массив массивов длины `len(features)`, где каждый вложенный массив - значения одного конкретного признака во всех транзакциях клиента. С целью оптимизации решения будем использовать технику `sequence_bucketing`. 
* Для реализации `nn.Embedding` нам необходимо, чтобы все признаки были категориальными. С этой целью все вещественные признаки такие , как [`amnt`, `days_before`, `hour_diff`] преобразуем в категорильные с помощью `np.digitize`
* In order to choose the size of embedding for categorical variables use the next formula: `def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))`

## На что стоит обратить внимание
    - сдвиг по времени
    - несбалансированные классы
