# Overview da metodologia e do pacote

Este documento apresenta a metodologia de forma operacional. O objetivo é
responder três perguntas antes de entrar nas derivações matriciais:

1. O que o método tenta descobrir?
2. Como os dados percorrem o algoritmo?
3. Quais funções e objetos Python representam cada etapa?

Para o overview matemático com equações renderizadas, consulte
[`overview_matematico.ipynb`](../notebooks/overview_matematico.ipynb). Para a
derivação matemática completa, consulte [`metodologia.md`](metodologia.md). Para
ver cada operação sendo executada, consulte o notebook
`tutorial_inferencia_bayesiana.ipynb`.

## 1. Problema que o método resolve

Recebemos uma única série mensal até a data atual. Não assumimos que todo o
histórico represente o presente: uma mudança de nível ou direção pode ter
iniciado um regime novo.

O método procura conjuntamente:

- a data mais plausível de início do regime corrente;
- a forma local mais plausível: constante, reta ou curva logarítmica suave;
- a projeção futura;
- a incerteza sobre coeficientes, ruído, janela e modelo;
- sinais de que a escolha precisa de inspeção humana.

Ele não precisa agrupar séries nem aprender uma prior populacional. Cada série
é processada de forma independente.

## 2. Visão do fluxo completo

```text
observações (valor, t)
        │
        ▼
validação e recorte do lookback
        │
        ▼
centralização pela mediana + escala MAD
        │
        ▼
enumeração das hipóteses (janela × modelo)
        │
        ▼
matriz X + g-prior + posterior + evidência
        │
        ▼
probabilidades de janela, modelo e hipótese
        │
        ├──────────────► sensibilidade em g = n/2, n, 2n
        │
        ▼
mistura das distribuições preditivas
        │
        ▼
forecast + intervalo + decomposição da incerteza
        │
        ▼
recomendação automática e flag de revisão humana
```

A ideia central é não tomar uma decisão prematura. Em vez de escolher primeiro
uma janela e depois agir como se ela fosse certa, o método mantém todas as
hipóteses elegíveis e atribui uma probabilidade a cada uma.

## 3. Conceitos fundamentais

### 3.1 Observação

Uma observação contém:

- `value`: valor numérico;
- `t`: rótulo temporal, por exemplo `"2025-03"`.

As observações devem estar ordenadas e igualmente espaçadas. O pacote não
preenche meses ausentes.

### 3.2 Lookback

É o trecho máximo do histórico analisado. Se há mais observações que
`max_lookback`, somente as mais recentes entram na inferência.

Lookback não é a janela final do modelo. Ele é o conjunto comum de dados que
todas as hipóteses precisam explicar.

### 3.3 Hipótese

Uma hipótese é uma combinação de:

- uma janela: sem mudança ou início do regime em uma data candidata;
- um modelo do regime corrente: `constant`, `linear` ou `logarithmic`.

Exemplo:

```text
hipótese = início em 2025-03 + modelo linear
```

Quando há mudança, a parte anterior ao regime corrente é representada por uma
reta auxiliar. Ela não é usada como dinâmica do presente; serve para que a
hipótese explique o mesmo histórico completo que as demais.

### 3.4 Regime corrente

É o segmento terminal, que necessariamente chega ao último dado observado. A
data de início desse segmento é a janela útil proposta para projetar o estado
atual.

O MVP admite no máximo uma ruptura terminal por hipótese. Ele pode trabalhar
com dados que tiveram mudanças antigas, mas não devolve uma segmentação completa
de todos os regimes históricos.

### 3.5 Centralização e escala robustas

O centro é a mediana da própria série. A escala principal é o MAD, isto é, a
mediana dos desvios absolutos em relação à mediana, multiplicada por `1,4826`.

A transformação interna é:

```text
z = (y - centro) / escala
```

Isso tem dois efeitos:

- séries com unidades muito diferentes passam a ter uma escala numérica
  comparável;
- o centro zero da prior significa “nível típico da série e ausência de direção”,
  em vez de significar que o valor real deveria ser zero.

O centro e a escala são iguais para todas as hipóteses daquela série.

### 3.6 Matriz de desenho `X`

`X` traduz uma forma funcional em uma regressão.

- constante: coluna `1`;
- linear: colunas `1` e `idade`;
- logarítmico: colunas `1` e `log(1 + idade)`.

Com ruptura, `X` possui um bloco para a reta histórica e outro bloco para o
regime corrente. Zeros indicam quais coeficientes não atuam em cada ponto.

### 3.7 g-prior

Os coeficientes têm uma prior normal centrada em zero. Sua dispersão é:

```text
V0 = g × inversa(X'X)
```

O padrão é `g=n`, sendo `n` o número de observações no lookback. Isso é chamado
de prior de informação unitária: sua precisão equivale aproximadamente à
informação de uma observação típica.

A principal vantagem prática é que a prior se adapta à geometria de cada
matriz `X`. Não é necessário inventar variâncias diferentes para interceptos,
inclinações lineares e inclinações logarítmicas.

### 3.8 Posterior

A posterior combina a informação da prior com a informação dos dados. Para os
modelos padrão, essa combinação possui solução analítica
Normal–Inverse-Gamma.

Da posterior obtemos:

- centro provável dos coeficientes;
- incerteza dos coeficientes;
- estimativa da variância residual;
- distribuição preditiva para pontos futuros.

### 3.9 Evidência

A evidência é a probabilidade dos dados depois de integrar os coeficientes e a
variância residual. Ela permite comparar hipóteses com números diferentes de
parâmetros sem olhar somente o erro de ajuste.

Uma hipótese flexível pode ajustar melhor e ainda perder, porque precisa
espalhar sua massa anterior por um espaço maior de parâmetros. Essa é a
penalização bayesiana de complexidade.

### 3.10 Probabilidade posterior da hipótese

Para cada combinação de janela e modelo:

```text
probabilidade posterior ∝ evidência × prior da janela × prior do modelo
```

Somando probabilidades, obtemos:

- `window_probabilities`: probabilidade de cada início de regime;
- `model_probabilities`: probabilidade de cada família de modelo;
- `probability_no_change`: probabilidade de nenhum início novo no lookback.

### 3.11 Forecast como mistura

O forecast automático não usa apenas a hipótese vencedora. Ele combina as
distribuições preditivas de todas as hipóteses, ponderadas pelas respectivas
probabilidades.

Isso permite que o intervalo final incorpore duas fontes distintas:

- incerteza dentro da hipótese: ruído e coeficientes;
- incerteza entre hipóteses: discordância sobre janela e modelo.

### 3.12 Sensibilidade a `g`

A inferência completa é repetida com:

```text
g = n/2, n e 2n
```

O objetivo não é selecionar o melhor `g` usando os mesmos dados. O objetivo é
verificar se a recomendação sobre janela e modelo é estável diante de uma
variação razoável da força da prior.

Se a recomendação mudar, `prior_sensitive=True`.

### 3.13 Condicionamento numérico

O número de condição de `X` mede quão difícil é separar o efeito de seus
coeficientes. Número muito alto sugere que combinações diferentes de
coeficientes produzem quase o mesmo ajuste.

O pacote:

- rejeita matrizes acima de `max_design_condition_number`;
- retorna o número de condição da hipótese selecionada;
- retorna o maior número de condição encontrado.

### 3.14 Revisão humana

A revisão é recomendada quando pelo menos uma destas condições ocorre:

1. a discordância futura entre hipóteses, ponderada pela materialidade, supera
   `review_cost`;
2. a janela ou o modelo vencedor muda na análise de sensibilidade de `g`.

O usuário pode escolher visualmente outra janela ou outro modelo e executar
novamente. A recomendação automática original permanece registrada.

## 4. Componentes públicos do pacote

### `Observation`

Representa um ponto da série:

```python
Observation(value=123.4, t="2025-03")
```

Também são aceitas tuplas `(valor, t)` e dicionários.

### `BayesianConfig`

Concentra parâmetros estatísticos e operacionais. Os principais são:

| Parâmetro | Papel |
|---|---|
| `max_lookback` | máximo de observações recentes analisadas |
| `min_before` | mínimo de pontos antes de uma ruptura candidata |
| `min_after` | mínimo de pontos no regime corrente |
| `forecast_horizon` | número de meses projetados |
| `candidate_models` | modelos comparados |
| `prior_no_change` | massa anterior da hipótese sem mudança |
| `g_prior_multiplier` | define `g = n × multiplicador` |
| `g_sensitivity_multipliers` | valores usados na análise de sensibilidade |
| `confidence` | cobertura nominal do intervalo preditivo |
| `materiality` | peso operacional da série na triagem humana |
| `review_cost` | limiar operacional para revisão |

### `BayesianForecastPipeline`

É a interface principal.

```python
pipeline = BayesianForecastPipeline(config)
```

#### `run(...)`

Executa a inferência de uma série:

```python
result = pipeline.run(
    observations,
    series_id="serie_a",
    materiality=1.0,
)
```

Overrides opcionais:

```python
reviewed = pipeline.run(
    observations,
    window_start="2025-03",
    model_override="linear",
)
```

Para declarar que não houve ruptura:

```python
reviewed = pipeline.run(observations, force_no_change=True)
```

#### `run_many(...)`

Executa várias séries de forma independente:

```python
results = pipeline.run_many({
    "serie_a": observations_a,
    "serie_b": observations_b,
})
```

Não há compartilhamento de parâmetros entre as séries.

### `BayesianResult`

É o objeto completo retornado por `run`. Os grupos mais importantes são:

#### Preparação

- `lookback_observations`;
- `center`;
- `scale`;
- `prior_strategy`;
- `g_value`.

#### Inferência automática

- `hypotheses`;
- `window_probabilities`;
- `window_credible_set`;
- `model_probabilities`;
- `probability_no_change`;
- `probability_change`;
- `recommended_window_start_t`;
- `recommended_model`.

#### Decisão efetivamente usada

- `selected_window_start_t`;
- `selected_model`;
- `window_selection_source`;
- `model_selection_source`;
- `selected_forecast`.

Esses campos diferem dos `recommended_*` quando existe override humano.

#### Projeções

- `forecast`: mistura automática sobre todas as hipóteses;
- `selected_forecast`: condicionado à decisão usada;
- `model_forecasts`: um forecast por modelo, misturando janelas;
- `model_views`: ajuste e forecast por modelo na janela selecionada.

#### Diagnósticos

- `predictive_uncertainty_index`;
- `selection_uncertainty_index`;
- `g_sensitivity_results`;
- `prior_sensitive`;
- `max_g_sensitivity_forecast_spread`;
- `selected_design_condition_number`;
- `maximum_design_condition_number`;
- `human_intervention_required`.

#### `to_dict()`

Converte todo o resultado em estruturas serializáveis:

```python
payload = result.to_dict()
```

### `PredictivePoint`

Cada mês projetado contém:

- `mean` e `median`;
- limites `lower` e `upper`;
- `within_hypothesis_variance`;
- `between_hypothesis_variance`;
- `total_variance`.

### `HypothesisResult`

Registra modelo, janela, evidência, prior, probabilidade posterior, parâmetros,
variância residual, `g` e número de condição de uma hipótese específica.

### `GSensitivityResult`

Registra, para um multiplicador de `g`, a janela vencedora, o modelo vencedor,
a probabilidade de não mudança e as médias projetadas.

### `ModelView`

Foi desenhado para interfaces visuais. Contém a curva ajustada e o forecast de
um modelo, mantendo fixa a janela selecionada.

## 5. Funções internas e responsabilidade

As funções abaixo são detalhes de implementação e podem mudar entre versões,
mas ajudam a localizar cada etapa no código.

| Função | Responsabilidade |
|---|---|
| `_validate` | valida quantidade e valores das observações |
| `_eligible_models` | decide quais famílias podem participar |
| `_enumerate_hypotheses` | cria todas as combinações de janela e modelo |
| `_design_matrix` | constrói `X` para hipótese segmentada ou sem mudança |
| `_basis` | constrói a base constante, linear ou logarítmica |
| `_fit_hypothesis` | calcula posterior e evidência conjugadas |
| `_normalize_hypotheses` | transforma log-pesos em probabilidades |
| `_window_probabilities` | marginaliza sobre modelos |
| `_model_probabilities` | marginaliza sobre janelas |
| `_g_sensitivity` | repete a inferência para diferentes valores de `g` |
| `_mixture_forecast` | combina distribuições preditivas |
| `_predictive_moments` | calcula média e variância condicionais |
| `_predictive_samples` | simula quantis da distribuição preditiva |
| `_selected_window_states` | aplica recomendação ou override humano |
| `_current_fitted` | gera a curva ajustada para visualização |

## 6. Exemplo mínimo de leitura do resultado

```python
result = pipeline.run(series, series_id="produto_a")

print("Regime recomendado:", result.recommended_window_start_t)
print("Modelo recomendado:", result.recommended_model)
print("Probabilidades dos modelos:", result.model_probabilities)
print("Sensível à prior:", result.prior_sensitive)
print("Requer revisão:", result.human_intervention_required)

for point in result.forecast:
    print(point.t, point.mean, point.lower, point.upper)
```

Uma leitura operacional possível é:

1. verificar `recommended_window_start_t` e `recommended_model`;
2. observar se a massa de `window_probabilities` está concentrada ou dispersa;
3. verificar `prior_sensitive`;
4. comparar variância dentro e entre hipóteses;
5. se `human_intervention_required=True`, mostrar `model_views` e datas
   candidatas ao usuário;
6. executar novamente com os overrides escolhidos.

## 7. O que o método não afirma

- Ele não prova que houve uma ruptura física no processo.
- Ele não recupera automaticamente todos os regimes históricos.
- Ele não resolve sazonalidade, autocorrelação ou variância variável.
- Ele não torna o forecast confiável indefinidamente.
- Ele não elimina a influência das priors; torna essa influência explícita e
  testável.
- Ele não substitui validação prospectiva quando novos meses estiverem
  disponíveis.

O resultado deve ser lido como uma comparação probabilística entre as
hipóteses que foram colocadas no sistema, não como certeza de que uma delas é a
descrição completa da realidade.
