# Probabilistic Window Forecast

Pacote Python para inferência bayesiana conjunta da janela útil, do modelo local
e do forecast de séries mensais.

Em vez de selecionar uma única janela por regras fixas, o pacote calcula:

- probabilidade de não ter ocorrido uma ruptura recente;
- distribuição posterior da data de início do regime;
- probabilidades dos modelos candidatos;
- forecast integrado sobre todas as combinações;
- intervalo preditivo que inclui incerteza de parâmetros, janela e modelo;
- decomposição da variância dentro e entre hipóteses;
- prior adaptada à geometria de cada hipótese, sem calibrar uma população de séries;
- sensibilidade automática da decisão à força dessa prior;
- indicação de revisão baseada em perda esperada e materialidade.

## Documentação

- [Overview matemático com equações renderizadas](notebooks/overview_matematico.ipynb)
- [Fonte Markdown do overview matemático](docs/overview_matematico.md)
- [Overview operacional, conceitos e funções Python](docs/overview.md)
- [Metodologia e fórmulas](docs/metodologia.md)
- [Tutorial executável: derivação passo a passo](notebooks/tutorial_inferencia_bayesiana.ipynb)
- [Exemplo executável](example.py)

## Instalação

```bash
python -m pip install -e .
```

## Uso básico

```python
from probabilistic_window_forecast import (
    BayesianConfig,
    BayesianForecastPipeline,
)

series = [
    (100.0, "2024-01"),
    (101.4, "2024-02"),
    (103.2, "2024-03"),
    # ...
]

config = BayesianConfig(
    max_lookback=60,
    min_before=8,
    min_after=8,
    forecast_horizon=6,
    prior_no_change=0.60,
    posterior_draws=12_000,
)

pipeline = BayesianForecastPipeline(config)
result = pipeline.run(series, series_id="produto_a")
```

## Resultados principais

```python
print(result.probability_no_change)
print(result.probability_change)
print(result.window_probabilities)
print(result.window_credible_set)
print(result.model_probabilities)
print(result.recommended_window_start_t)
print(result.recommended_model)
print(result.center, result.scale, result.g_value)
print(result.prior_sensitive)
print(result.g_sensitivity_results)
```

Cada ponto do forecast contém:

```python
for point in result.forecast:
    print(
        point.t,
        point.mean,
        point.median,
        point.lower,
        point.upper,
        point.within_hypothesis_variance,
        point.between_hypothesis_variance,
    )
```

`result.forecast` é o forecast automático integrado sobre todas as janelas e
modelos. `result.selected_forecast` é condicionado à janela e ao modelo
selecionados, permitindo comparação visual entre a projeção integrada e a
escolha oficial.

No nível da série, o pacote fornece dois índices adimensionais:

```python
print(result.predictive_uncertainty_index)  # ruído + parâmetros + seleção
print(result.selection_uncertainty_index)   # somente discordância entre hipóteses
```

O segundo representa a parcela que uma revisão da janela/modelo pode reduzir.
Os índices não recebem rótulos arbitrários de baixa/média/alta.

## Modelos

Por padrão, são usados os modelos com evidência analítica:

```text
constant
linear
logarithmic
```

O exponencial pode ser habilitado:

```python
config = BayesianConfig(
    center_series=False,
    candidate_models=(
        "constant",
        "linear",
        "logarithmic",
        "exponential",
    ),
)
```

O exponencial exige valores positivos, `center_series=False` e usa aproximação
de Laplace. Ele não participa da nova centralização robusta nem da g-prior e é
considerado experimental. Para o fluxo recomendado do MVP, mantenha apenas
constante, linear e logarítmico.

## Prior automática por série

Cada série é centralizada pela própria mediana e dividida por uma escala robusta
baseada no desvio absoluto mediano (MAD). Assim, `b0 = 0` significa apenas
"sem direção anterior após centralizar", e não "a série deve estar perto de
zero".

Nos modelos conjugados, a dispersão dos coeficientes não é mais uma diagonal
arbitrária como `[100, 1, ...]`. Para cada hipótese, o pacote usa a g-prior de
Zellner:

```text
V0 = g × (X'X)^-1
g = n × g_prior_multiplier
```

O padrão `g=n` é chamado de prior de informação unitária: a precisão da prior
equivale aproximadamente à informação de uma observação típica. Como `V0`
depende da matriz `X` da própria hipótese, a escala de uma inclinação linear e
de uma inclinação logarítmica é tratada de forma coerente.

Não é preciso agrupar séries nem estimar hiperparâmetros populacionais.

## Sensibilidade da prior

Por padrão, a inferência é repetida com `g=n/2`, `g=n` e `g=2n`:

```python
for item in result.g_sensitivity_results:
    print(
        item.g_multiplier,
        item.recommended_window_start_t,
        item.recommended_model,
        item.forecast_means,
    )

print(result.prior_sensitive)
print(result.max_g_sensitivity_forecast_spread)
```

`prior_sensitive=True` significa que a janela ou o modelo vencedor mudou nessa
faixa. Nesse caso, a série é marcada para revisão humana, salvo se o usuário já
tiver feito um override. Isso transforma uma escolha inevitável de prior em um
diagnóstico explícito, em vez de escondê-la.

## Priors dos modelos

Por padrão, os modelos elegíveis recebem pesos iguais. É possível configurar:

```python
config = BayesianConfig(
    center_series=False,  # necessario apenas porque o exemplo inclui exponencial
    model_priors={
        "constant": 0.35,
        "linear": 0.35,
        "logarithmic": 0.20,
        "exponential": 0.10,
    },
    candidate_models=(
        "constant",
        "linear",
        "logarithmic",
        "exponential",
    ),
)
```

Os pesos precisam ser positivos para todos os modelos elegíveis. O pacote os
normaliza automaticamente.

## Escolha humana

A inferência automática nunca é apagada por um override.

```python
reviewed = pipeline.run(
    series,
    window_start="2024-03",
    model_override="linear",
)
```

Para forçar ausência de ruptura:

```python
reviewed = pipeline.run(
    series,
    force_no_change=True,
    model_override="linear",
)
```

O resultado preserva:

```text
recommended_window_start_t   recomendação probabilística
selected_window_start_t      decisão efetivamente usada
recommended_model            modelo com maior probabilidade marginal
selected_model               modelo efetivamente usado
window_selection_source      automatic ou manual
model_selection_source       automatic ou manual
```

## Alternância visual por modelo

O pacote calcula um forecast condicionado a cada família de modelo, mas ainda
integrado sobre as janelas:

```python
linear_forecast = result.model_forecasts["linear"]
log_forecast = result.model_forecasts["logarithmic"]
```

A aplicação pode alternar essas projeções sem executar novamente o estimador.

Para comparar os modelos mantendo fixa a janela selecionada, use
`model_views`. Cada visão traz o ajuste dentro do regime corrente, o forecast e
a probabilidade do modelo condicionada àquela janela:

```python
view = result.model_views["linear"]

print(view.conditional_probability)
for point in view.fitted:
    print(point.t, point.value)
for point in view.forecast:
    print(point.t, point.mean, point.lower, point.upper)
```

Assim, a interface pode sobrepor os pontos observados, a curva ajustada e a
projeção de cada modelo sem alterar a recomendação automática. Se o usuário
confirmar uma alternativa, basta executar novamente com `window_start` e/ou
`model_override`.

## Intervenção humana

A decisão usa a discordância entre hipóteses:

```python
print(result.expected_decision_loss)
print(result.review_cost)
print(result.human_intervention_required)
```

A perda implementada no MVP é:

```text
materialidade × média da variância entre hipóteses / escala²
```

`review_cost` é uma política operacional configurável, não um parâmetro
estatístico. A revisão também é solicitada quando `prior_sensitive=True`. Um
override válido registra a intervenção como resolvida.

## Muitas séries

```python
results = pipeline.run_many({
    "serie_a": serie_a,
    "serie_b": serie_b,
})
```

Esta versão processa as séries independentemente. Isso é intencional: a g-prior
se adapta à matriz de desenho de cada hipótese e não pressupõe uma população
comparável de séries. Pooling hierárquico pode ser estudado no futuro, mas não é
necessário para operar o MVP.

## Entrada e serialização

São aceitas tuplas `(valor, t)`:

```python
[(100.0, "2024-01"), (101.0, "2024-02")]
```

ou dicionários:

```python
[
    {"t": "2024-01", "valor": 100.0},
    {"t": "2024-02", "value": 101.0},
]
```

As observações devem estar em ordem cronológica e ser igualmente espaçadas.

```python
payload = result.to_dict()
```

## Limitações

- uma única ruptura terminal;
- regime anterior representado por uma reta;
- erros aditivos, normais e homocedásticos;
- ausência de sazonalidade e autocorrelação;
- g-prior é uma convenção transparente, não conhecimento substantivo do domínio;
- sensibilidade testa uma vizinhança de g, mas não elimina risco de má especificação;
- exponencial por aproximação de Laplace;
- quantis da mistura obtidos por Monte Carlo;
- decisão humana baseada em uma função de perda inicial e simplificada.
