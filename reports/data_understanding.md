# Relatorio de entendimento dos dados

Shape: 4000 linhas x 21 colunas

## Tipos das colunas
|                  | 0       |
|:-----------------|:--------|
| customerID       | object  |
| gender           | object  |
| SeniorCitizen    | object  |
| Partner          | object  |
| Dependents       | object  |
| tenure           | int64   |
| PhoneService     | object  |
| MultipleLines    | object  |
| InternetService  | object  |
| OnlineSecurity   | object  |
| OnlineBackup     | object  |
| DeviceProtection | object  |
| TechSupport      | object  |
| StreamingTV      | object  |
| StreamingMovies  | object  |
| Contract         | object  |
| PaperlessBilling | object  |
| PaymentMethod    | object  |
| MonthlyCharges   | float64 |
| TotalCharges     | float64 |
| Churn            | object  |

## Valores ausentes
|                  |   0 |
|:-----------------|----:|
| TotalCharges     |  40 |
| customerID       |   0 |
| DeviceProtection |   0 |
| MonthlyCharges   |   0 |
| PaymentMethod    |   0 |
| PaperlessBilling |   0 |
| Contract         |   0 |
| StreamingMovies  |   0 |
| StreamingTV      |   0 |
| TechSupport      |   0 |
| OnlineBackup     |   0 |
| gender           |   0 |
| OnlineSecurity   |   0 |
| InternetService  |   0 |
| MultipleLines    |   0 |
| PhoneService     |   0 |
| tenure           |   0 |
| Dependents       |   0 |
| Partner          |   0 |
| SeniorCitizen    |   0 |
| Churn            |   0 |

Duplicados completos: 0

## Distribuicao da classe alvo
| Churn   |   count |   share |
|:--------|--------:|--------:|
| No      |    2170 |  0.5425 |
| Yes     |    1830 |  0.4575 |

## Estatisticas descritivas
|                  |   count |   unique | top                     |   freq |      mean |       std |   min |     25% |      50% |       75% |    max |
|:-----------------|--------:|---------:|:------------------------|-------:|----------:|----------:|------:|--------:|---------:|----------:|-------:|
| customerID       |    4000 |     4000 | SYN-00000               |      1 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| gender           |    4000 |        2 | Male                    |   2015 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| SeniorCitizen    |    4000 |        2 | No                      |   3362 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| Partner          |    4000 |        2 | Yes                     |   2021 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| Dependents       |    4000 |        2 | No                      |   2808 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| tenure           |    4000 |      nan | nan                     |    nan |   36.6482 |   20.5044 |     1 |   19    |   37     |   54      |   72   |
| PhoneService     |    4000 |        2 | Yes                     |   3589 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| MultipleLines    |    4000 |        3 | Yes                     |   1340 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| InternetService  |    4000 |        3 | Fiber optic             |   1764 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| OnlineSecurity   |    4000 |        3 | No internet service     |   1355 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| OnlineBackup     |    4000 |        3 | Yes                     |   1383 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| DeviceProtection |    4000 |        3 | No                      |   1348 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| TechSupport      |    4000 |        2 | No                      |   2588 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| StreamingTV      |    4000 |        3 | No                      |   1357 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| StreamingMovies  |    4000 |        3 | No                      |   1344 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| Contract         |    4000 |        3 | Month-to-month          |   2215 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| PaperlessBilling |    4000 |        2 | Yes                     |   2411 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| PaymentMethod    |    4000 |        4 | Credit card (automatic) |   1047 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |
| MonthlyCharges   |    4000 |      nan | nan                     |    nan |   70.009  |   26.982  |    20 |   51.45 |   68.855 |   88.6625 |  130   |
| TotalCharges     |    3960 |      nan | nan                     |    nan | 2578.11   | 1834.49   |     0 | 1079.16 | 2245.2   | 3754.86   | 8950.3 |
| Churn            |    4000 |        2 | No                      |   2170 |  nan      |  nan      |   nan |  nan    |  nan     |  nan      |  nan   |

## Significado das variaveis
|                  | descricao                                                                                     |
|:-----------------|:----------------------------------------------------------------------------------------------|
| customerID       | Identificador unico do cliente; removido do treino por nao ter valor preditivo generalizavel. |
| gender           | Genero informado pelo cliente.                                                                |
| SeniorCitizen    | Indicador de cliente idoso: 1 para sim, 0 para nao.                                           |
| Partner          | Se o cliente possui parceiro(a).                                                              |
| Dependents       | Se o cliente possui dependentes.                                                              |
| tenure           | Tempo de relacionamento com a empresa, em meses.                                              |
| PhoneService     | Se possui servico de telefone.                                                                |
| MultipleLines    | Se possui multiplas linhas telefonicas.                                                       |
| InternetService  | Tipo de servico de internet.                                                                  |
| OnlineSecurity   | Contratacao de seguranca online.                                                              |
| OnlineBackup     | Contratacao de backup online.                                                                 |
| DeviceProtection | Contratacao de protecao de dispositivo.                                                       |
| TechSupport      | Contratacao de suporte tecnico.                                                               |
| StreamingTV      | Contratacao de streaming de TV.                                                               |
| StreamingMovies  | Contratacao de streaming de filmes.                                                           |
| Contract         | Tipo de contrato: mensal, um ano ou dois anos.                                                |
| PaperlessBilling | Se usa cobranca sem papel.                                                                    |
| PaymentMethod    | Forma de pagamento.                                                                           |
| MonthlyCharges   | Valor mensal cobrado.                                                                         |
| TotalCharges     | Valor total cobrado historicamente.                                                           |
| Churn            | Target: se o cliente cancelou o servico.                                                      |

## Analise inicial da qualidade
- O identificador `customerID` nao deve entrar no modelo, pois memoriza clientes em vez de aprender padroes.
- `TotalCharges` pode conter strings vazias no dataset original; convertemos para numerico e imputamos dentro do pipeline.
- O target e desbalanceado moderadamente, por isso recall, F1 e ROC-AUC sao mais informativos que accuracy isolada.
- Nao removemos outliers automaticamente: cobrancas altas podem representar clientes reais de maior valor.