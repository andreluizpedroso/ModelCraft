# Relatorio executivo do modelo de churn

## Problema de negocio
O objetivo e prever quais clientes tem maior probabilidade de cancelar o servico de telecom. A empresa pode usar essa previsao para priorizar acoes de retencao, como contato proativo, melhoria de suporte ou oferta comercial.

## Metrica que mais importa
A metrica primaria e **ROC-AUC**, porque mede a capacidade de ranquear clientes de maior risco independentemente de um limiar fixo. Para operacao, **recall de churn** tambem e critico: perder um cliente que realmente iria cancelar tende a custar mais do que oferecer um incentivo a um cliente que ficaria.

## Custos dos erros
- Falso negativo: cliente com risco real nao recebe acao de retencao. Custo potencial: perda de receita recorrente e custo de reacquisicao.
- Falso positivo: cliente sem risco recebe incentivo ou contato. Custo potencial: desconto desnecessario, tempo da equipe e fadiga de comunicacao.

## Comparacao de modelos
| model                  |   cv_roc_auc_mean |   cv_roc_auc_std |   train_roc_auc_mean |   valid_roc_auc |   valid_average_precision |   valid_recall |   valid_f1 |   overfit_gap_auc |
|:-----------------------|------------------:|-----------------:|---------------------:|----------------:|--------------------------:|---------------:|-----------:|------------------:|
| logistic_regression    |            0.7456 |           0.0147 |               0.7616 |          0.7508 |                    0.7103 |         0.6421 |     0.6386 |            0.016  |
| random_forest          |            0.7201 |           0.013  |               1      |          0.7353 |                    0.6946 |         0.582  |     0.6147 |            0.2799 |
| catboost               |            0.726  |           0.0137 |               0.9245 |          0.7336 |                    0.6953 |         0.6284 |     0.6233 |            0.1985 |
| xgboost                |            0.7155 |           0.0167 |               0.9339 |          0.7268 |                    0.6742 |         0.582  |     0.6147 |            0.2184 |
| hist_gradient_boosting |            0.6886 |           0.0165 |               0.9992 |          0.7055 |                    0.6579 |         0.5984 |     0.6066 |            0.3106 |
| lightgbm               |            0.6926 |           0.016  |               0.999  |          0.6998 |                    0.648  |         0.582  |     0.5975 |            0.3064 |
| dummy_baseline         |            0.5    |           0      |               0.5    |          0.5    |                    0.4575 |         0      |     0      |            0      |

## Melhor modelo e hiperparametros
Modelo selecionado: **logistic_regression**

- `model__C`: `0.1`

## Metricas finais no teste
- **accuracy**: 0.6850
- **precision**: 0.6566
- **recall**: 0.6530
- **f1**: 0.6548
- **roc_auc**: 0.7364
- **average_precision**: 0.6924

## Decisao de uso
O modelo e util como sistema de priorizacao, nao como decisor automatico de cancelamento. Em producao, o limiar deve ser calibrado conforme capacidade do time de retencao e valor esperado de cada cliente.
