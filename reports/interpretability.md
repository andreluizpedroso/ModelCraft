# Interpretabilidade

## Variaveis mais importantes
| feature                     |   importance |
|:----------------------------|-------------:|
| Contract_Month-to-month     |     0.675607 |
| tenure                      |     0.673207 |
| InternetService_Fiber optic |     0.568699 |
| MonthlyCharges              |     0.39149  |
| Contract_One year           |     0.350678 |
| Contract_Two year           |     0.324922 |
| InternetService_DSL         |     0.324839 |
| InternetService_No          |     0.243853 |
| TechSupport_No              |     0.200338 |
| TechSupport_Yes             |     0.200331 |

## Como interpretar
Importancia alta nao prova causalidade. Ela indica que a variavel ajudou o modelo a separar clientes com e sem churn dentro dos dados historicos.

Em churn de telecom, variaveis como tipo de contrato, tempo como cliente, suporte tecnico e valor mensal costumam ser fortes porque representam atrito, comprometimento contratual e percepcao de valor.

## SHAP
SHAP nao foi gerado nesta execucao. Motivo: `ModuleNotFoundError: No module named 'shap'`. A interpretabilidade principal usa importancia de features/permutacao.
