# EDA - principais achados

- Churn e uma classe minoritaria, mas relevante; otimizar apenas accuracy favoreceria previsoes conservadoras.
- Contratos mensais tendem a concentrar maior cancelamento, o que faz sentido porque a barreira de saida e menor.
- `tenure`, `MonthlyCharges` e `TotalCharges` devem ser avaliadas juntas: clientes novos com mensalidade alta costumam ser mais frageis.
- Valores ausentes em `TotalCharges` sao tratados como problema de qualidade, nao como informacao a ser removida antes do split.
- Outliers financeiros podem ser clientes premium; por isso usamos modelos robustos e avaliamos impacto em vez de excluir linhas cegamente.