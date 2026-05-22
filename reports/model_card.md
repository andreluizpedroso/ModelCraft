# Model Card - Churn Prediction

## Uso pretendido
Priorizar clientes para campanhas de retencao e investigacao comercial. O modelo deve apoiar analistas, nao substituir julgamento humano.

## Fora de escopo
- Decidir sozinho se um cliente deve receber desconto.
- Inferir motivos reais de cancelamento sem analise causal.
- Ser usado em outros mercados sem revalidacao.

## Dados
Base Telco Customer Churn, com informacoes contratuais, servicos, cobrancas e target de churn. Se o download falhar, o projeto gera uma base sintetica com estrutura semelhante para permitir execucao didatica.

## Metricas
As metricas finais sao geradas em `reports/test_metrics.json` ao executar `python main.py`.

## Riscos e vieses
O modelo aprende padroes historicos. Se segmentos especificos receberam atendimento pior, precos diferentes ou campanhas anteriores, esses efeitos podem aparecer como sinal preditivo sem serem uma causa justa ou acionavel.

## Monitoramento recomendado
- Acompanhar ROC-AUC, recall e precision mensalmente.
- Monitorar drift em `tenure`, `MonthlyCharges`, `Contract` e `PaymentMethod`.
- Recalibrar limiar de decisao conforme capacidade da equipe de retencao.
- Revalidar o modelo apos mudancas de produto, preco ou politica comercial.
