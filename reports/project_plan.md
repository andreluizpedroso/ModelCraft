# Plano detalhado do projeto

## 1. Problema escolhido
Prever churn de clientes de telecomunicacoes. O modelo recebe atributos de contrato, servicos contratados, tempo de relacionamento e cobrancas para estimar a probabilidade de cancelamento.

## 2. Tipo de problema
Classificacao binaria:

- Classe 0: cliente nao cancelou.
- Classe 1: cliente cancelou.

## 3. Impacto real
Uma empresa pode usar o score para priorizar clientes em campanhas de retencao. Isso reduz desperdicio de descontos em clientes estaveis e aumenta a chance de agir antes que clientes de alto risco cancelem.

## 4. Metrica de negocio
ROC-AUC e a metrica tecnica primaria porque mede ranking de risco. Em operacao, recall da classe churn e fundamental quando o custo de perder cliente e maior que o custo de uma abordagem comercial.

## 5. Erros mais graves
- Falso negativo: cliente que cancelaria nao e priorizado. Pode gerar perda de receita recorrente.
- Falso positivo: cliente sem risco recebe incentivo. Gera custo de campanha, desconto desnecessario e possivel fadiga.

## 6. Arquitetura
- `main.py`: orquestracao ponta a ponta.
- `src/data.py`: carga, limpeza minima e relatorio de dados.
- `src/eda.py`: graficos e insights exploratorios.
- `src/preprocessing.py`: split e transformacoes em `Pipeline`.
- `src/models.py`: baseline, modelos avancados, validacao cruzada e tuning.
- `src/evaluation.py`: metricas, matriz de confusao, curva PR e relatorio de negocio.
- `src/interpretability.py`: importancia de features e SHAP quando disponivel.
- `src/leakage_checks.py`: validacoes defensivas contra vazamento.

## 7. Decisoes anti-leakage
- O teste e separado antes de qualquer treino final.
- Imputacao, scaler e encoding ficam dentro do `Pipeline`.
- O pre-processador e ajustado somente nos folds de treino durante validacao cruzada.
- `customerID` e removido porque e identificador, nao sinal generalizavel.
- O target nao entra em nenhuma transformacao de features.

## 8. Riscos tecnicos
- Dataset remoto pode estar indisponivel; por isso ha fallback sintetico.
- Dependencias opcionais podem nao estar instaladas; o codigo treina os modelos disponiveis.
- SHAP pode falhar em alguns estimadores ou ambientes; ha fallback para importancia de features.
- A base historica pode refletir vieses comerciais e nao necessariamente causalidade.
- Mudancas de preco, produto ou atendimento podem gerar drift e degradar performance.
