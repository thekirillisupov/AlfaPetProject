# AlfaPetProject
## Description
Hi! This is my pet project using data from AlfaBattle competition. Part of the code was taken from open sources of the competition's organizers. 

## Boosting + RF + LinReg model

|             | baseline | my model                       |
|-------------|----------|--------------------------------|
| ROC AUC     | 75.1     | 75.2                           |
| description | boosting | feature gen. boosting stacking |

## RNN model 
In order to choose the size of embedding for categorical variables use the next formula: `def emb_sz_rule(n_cat:int)->int: return min(600, round(1.6 * n_cat**0.56))`

## На что стоит обратить внимание
    - сдвиг по времени
    - несбалансированные классы
