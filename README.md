# Churn Prediction com Machine Learning

Projeto completo de Machine Learning em Python para prever churn de clientes de telecomunicacoes, com foco em qualidade de engenharia, reprodutibilidade, interpretabilidade e prevencao de data leakage.

## Problema

Empresas de assinatura perdem receita quando clientes cancelam sem que haja tempo para uma acao de retencao. Este projeto treina um modelo para ranquear clientes por risco de churn e ajudar times de negocio a priorizar contato, suporte ou ofertas.

O problema e de **classificacao binaria**:

- `0`: cliente nao cancelou.
- `1`: cliente cancelou.

## Decisao de negocio

A metrica tecnica primaria e **ROC-AUC**, pois o modelo sera usado para priorizar clientes por score. Para a operacao, **recall da classe churn** tambem e critico: falso negativo significa nao agir sobre um cliente que provavelmente cancelaria.

Custos dos erros:

- **Falso negativo**: perda de receita recorrente e custo futuro de reacquisicao.
- **Falso positivo**: desconto ou contato desnecessario, custo operacional e fadiga do cliente.

## Dados

O projeto usa o dataset publico Telco Customer Churn. O script tenta baixar a base automaticamente. Se o ambiente estiver sem rede, uma base sintetica semelhante e criada para manter o projeto executavel.

Principais variaveis:

- `tenure`: meses como cliente.
- `Contract`: tipo de contrato.
- `MonthlyCharges`: cobranca mensal.
- `TotalCharges`: cobranca total historica.
- `TechSupport`, `OnlineSecurity`, `InternetService`: servicos contratados.
- `Churn`: alvo.

## Arquitetura

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── images/
├── models/
├── notebooks/
│   ├── 01_end_to_end_churn_project.py   <- script com marcadores # %%
│   └── 01_end_to_end_churn_project.ipynb <- notebook Jupyter interativo
├── reports/
│   ├── model_card.md
│   └── project_plan.md
├── src/
│   ├── config.py
│   ├── data.py
│   ├── eda.py
│   ├── evaluation.py
│   ├── interpretability.py
│   ├── leakage_checks.py
│   ├── models.py
│   ├── preprocessing.py
│   └── utils.py
├── main.py
├── requirements.txt
└── README.md
```

## Boas praticas de ML Engineering

- Split treino/validacao/teste estratificado.
- Teste final intocado ate a avaliacao.
- `Pipeline` e `ColumnTransformer` para imputacao, encoding e scaler.
- Transformacoes ajustadas apenas no treino.
- Remocao de `customerID` para evitar memorizacao.
- Baseline com `DummyClassifier`.
- Comparacao com modelos robustos: Random Forest, HistGradientBoosting e, quando instalados, XGBoost, LightGBM e CatBoost.
- Validacao cruzada com `StratifiedKFold`.
- Otimizacao com `RandomizedSearchCV`.
- Artefatos salvos com `joblib`.
- Relatorios e imagens gerados automaticamente.

## Como executar

### Pipeline completo (terminal)

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Notebook interativo (VS Code)

1. Abra `notebooks/01_end_to_end_churn_project.ipynb` no VS Code.
2. Clique em **Select Kernel** no canto superior direito e escolha `venv\Scripts\python.exe`.
3. Use **Run Cell** (`Shift+Enter`) para executar celula a celula, ou **Run All** para rodar tudo.

Saidas principais:

- `models/churn_pipeline.joblib`: pipeline final treinado.
- `models/best_hyperparameters.json`: melhores hiperparametros.
- `reports/model_leaderboard.csv`: comparacao de modelos.
- `reports/test_metrics.json`: metricas finais no teste.
- `reports/business_report.md`: leitura executiva.
- `reports/interpretability.md`: interpretabilidade.
- `images/`: EDA, matriz de confusao, curva PR e importancia de features.

## Como carregar e prever

```python
import joblib
import pandas as pd

pipeline = joblib.load("models/churn_pipeline.joblib")

novo_cliente = pd.DataFrame([
    {
        "gender": "Female",
        "SeniorCitizen": "No",
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 8,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.5,
        "TotalCharges": 764.0,
    }
])

prob_churn = pipeline.predict_proba(novo_cliente)[:, 1][0]
print(f"Probabilidade de churn: {prob_churn:.2%}")
```

## Avaliacao

Execute `python main.py` para gerar as metricas reais do ambiente. O projeto calcula:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Average Precision
- Matriz de confusao
- Precision-Recall Curve

### Resultados obtidos

| Modelo | ROC-AUC (CV) | ROC-AUC (validacao) |
|---|---|---|
| **Logistic Regression** | **0.746** | **0.751** |
| Random Forest | 0.720 | 0.735 |
| CatBoost | 0.726 | 0.734 |
| XGBoost | 0.715 | 0.727 |
| HistGradientBoosting | 0.689 | 0.705 |
| LightGBM | 0.693 | 0.700 |
| Dummy Baseline | 0.500 | 0.500 |

**Metricas finais no teste (modelo: Logistic Regression):**

| Metrica | Valor |
|---|---|
| ROC-AUC | 0.736 |
| F1-score | 0.655 |
| Recall | 0.653 |
| Precision | 0.657 |
| Accuracy | 0.685 |

## Interpretabilidade

O projeto gera importancia de features para explicar quais variaveis mais influenciam a previsao. Quando `shap` funciona no ambiente, tambem gera `images/shap_beeswarm.png`.

Importante: interpretabilidade preditiva nao e causalidade. Uma feature importante mostra associacao aprendida pelo modelo, nao prova que mudar aquela variavel causara reducao de churn.

## Limitacoes

- O modelo nao entende eventos recentes fora da base historica.
- Pode falhar se houver mudanca de preco, produto, atendimento ou concorrencia.
- Pode reproduzir vieses historicos de campanhas e segmentos.
- Nao deve decidir descontos sozinho.
- Precisa de monitoramento de drift e recalibracao de limiar.
- Clientes com perfis raros podem ter previsoes menos confiaveis.

## Proximos passos

- Calibrar probabilidades com `CalibratedClassifierCV`.
- Otimizar limiar por custo esperado, nao apenas por metrica estatistica.
- Adicionar MLflow para tracking de experimentos.
- Criar API de inferencia com FastAPI.
- Adicionar testes automatizados em CI.
- Monitorar drift e performance pos-deploy.
