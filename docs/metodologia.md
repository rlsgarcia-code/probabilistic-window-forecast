# Metodologia

## 1. Hipóteses comparadas

Cada hipótese é uma combinação:

\[
c=(m,\tau),
\]

em que \(m\) é o modelo do regime corrente e \(\tau\) é sua data inicial. A
hipótese \(\tau=\varnothing\) representa ausência de mudança.

Todas as hipóteses explicam o mesmo lookback completo. Quando existe ruptura, o
trecho anterior é representado por uma reta de apoio e o trecho terminal usa o
modelo candidato.

## 2. Centralização e escala robustas

Para que cada série possa ser tratada sem uma população comparável, calculamos:

\[
c_y=\operatorname{mediana}(y)
\]

e

\[
s_y=1{,}4826\operatorname{mediana}|y-c_y|.
\]

Se o MAD for praticamente zero, usa-se uma escala de segurança baseada no
desvio-padrão, no módulo do centro e em 1. A resposta interna é:

\[
z_t=\frac{y_t-c_y}{s_y}.
\]

Essa transformação mantém a evidência comparável, pois é comum a todas as
hipóteses da série. Ela também dá interpretação clara a \(\mathbf b_0=0\): após
retirar o nível típico da própria série, a prior não favorece direção positiva
ou negativa. As saídas são retornadas por \(y=c_y+s_yz\).

## 3. Modelos conjugados

Constante, linear e logarítmico podem ser escritos como:

\[
\mathbf z=\mathbf X_c\boldsymbol\beta_c+\boldsymbol\varepsilon,
\qquad
\boldsymbol\varepsilon\sim N(0,\sigma^2\mathbf I).
\]

A prior é:

\[
\boldsymbol\beta_c\mid\sigma^2
\sim N(\mathbf b_0,\sigma^2\mathbf V_0),
\]

\[
\sigma^2\sim\operatorname{InverseGamma}(a_0,d_0).
\]

Na implementação, \(\mathbf b_0=0\). O ponto decisivo é como obter
\(\mathbf V_0\) sem escolher manualmente uma variância para cada coeficiente.

### 3.1 De onde vem a g-prior

Para uma hipótese com matriz de desenho \(\mathbf X\), a informação trazida
pelos dados sobre os coeficientes é proporcional a:

\[
\mathbf X^T\mathbf X.
\]

Logo, a escala natural da incerteza dos coeficientes é proporcional à inversa:

\[
(\mathbf X^T\mathbf X)^{-1}.
\]

A prior de Zellner usa exatamente essa geometria:

\[
\boxed{\mathbf V_0=g(\mathbf X^T\mathbf X)^{-1}}.
\]

Invertendo essa relação:

\[
\mathbf V_0^{-1}=\frac{1}{g}\mathbf X^T\mathbf X.
\]

Portanto, a precisão da prior é uma fração \(1/g\) da precisão da amostra
inteira. O padrão do pacote é \(g=n\). Nesse caso, a prior carrega
aproximadamente a informação de uma observação média; por isso é chamada de
*unit-information prior*.

Essa construção resolve dois problemas do MVP:

1. não exige grupos de séries para estimar hiperparâmetros;
2. adapta automaticamente a unidade de cada coeficiente à base constante,
   linear, logarítmica e às matrizes segmentadas.

Ela não é uma verdade sobre o negócio. É uma convenção fraca, reproduzível e
diagnosticável. O pacote testa sua sensibilidade explicitamente.

## 4. Posterior conjugada

A log-posterior combina dois termos quadráticos: o desvio em relação à prior e
o erro dos dados. Ao expandi-los e agrupar os termos em
\(\boldsymbol\beta\), as precisões se somam:

\[
\mathbf V_n=(\mathbf V_0^{-1}+\mathbf X^T\mathbf X)^{-1},
\]

\[
\mathbf b_n=
\mathbf V_n(\mathbf V_0^{-1}\mathbf b_0+\mathbf X^T\mathbf z),
\]

Com \(\mathbf b_0=0\) e a g-prior:

\[
\mathbf V_n=
\left(\frac{1}{g}\mathbf X^T\mathbf X+\mathbf X^T\mathbf X\right)^{-1}
\]

e

\[
\mathbf b_n=\frac{g}{g+1}\hat{\boldsymbol\beta}_{OLS}.
\]

Assim, o efeito de \(g\) é transparente. Para \(g=n\), o estimador de mínimos
quadrados é multiplicado por \(n/(n+1)\): pouca contração quando há dados, mais
regularização quando a janela é curta.

\[
a_n=a_0+\frac n2,
\]

\[
d_n=d_0+\frac12
(\mathbf z^T\mathbf z+
\mathbf b_0^T\mathbf V_0^{-1}\mathbf b_0-
\mathbf b_n^T\mathbf V_n^{-1}\mathbf b_n).
\]

## 5. Evidência analítica

\[
p(\mathbf z\mid c)=
(2\pi)^{-n/2}
\frac{|\mathbf V_n|^{1/2}}{|\mathbf V_0|^{1/2}}
\frac{d_0^{a_0}}{d_n^{a_n}}
\frac{\Gamma(a_n)}{\Gamma(a_0)}.
\]

O cálculo é feito em logaritmos. A evidência recompensa ajuste, mas penaliza
automaticamente modelos e segmentações que ocupam volume excessivo no espaço de
parâmetros.

## 6. Priors de janela e modelo

\[
P(\tau=\varnothing)=\pi_0.
\]

Se houver \(K\) datas elegíveis:

\[
P(\tau=k)=\frac{1-\pi_0}{K}.
\]

Com prior de modelo \(P(m)\):

\[
P(m,\tau)=P(m)P(\tau).
\]

## 7. Sensibilidade à força da prior

O pacote repete toda a comparação de janela e modelo com:

\[
g\in\{n/2,n,2n\}.
\]

Para cada valor, são guardados modelo vencedor, janela vencedora, probabilidade
de não mudança e médias do forecast. `prior_sensitive` é verdadeiro quando o
modelo ou a janela vencedora muda. Também se retorna a maior amplitude das
médias previstas, normalizada por \(s_y\).

Essa análise não escolhe o melhor \(g\) usando os mesmos dados. Ela pergunta se
a conclusão é estável diante de escolhas razoavelmente próximas. Instabilidade
gera revisão humana.

## 8. Probabilidade posterior

\[
P(c\mid\mathbf z)=
\frac{p(\mathbf z\mid c)P(c)}
{\sum_{c'}p(\mathbf z\mid c')P(c')}.
\]

A normalização usa `logsumexp`.

As marginais são:

\[
P(\tau\mid\mathbf z)=\sum_mP(m,\tau\mid\mathbf z),
\]

\[
P(m\mid\mathbf z)=\sum_\tau P(m,\tau\mid\mathbf z).
\]

## 9. Exponencial experimental

O exponencial usa apenas o modo de escala (`center_series=False`), pois a forma
atual supõe resposta estritamente positiva. O modelo terminal é:

\[
z_t=\exp(\eta+bu_t)+\varepsilon_t.
\]

O pacote encontra o MAP e aproxima a posterior por:

\[
\theta\mid\mathbf z,c
\approx N(\hat\theta_{MAP},\mathbf H^{-1}),
\]

em que \(\mathbf H\) é a Hessiana negativa da log-posterior.

A evidência de Laplace é:

\[
\log p(\mathbf z\mid c)
\approx
\log p(\mathbf z,\hat\theta\mid c)
+\frac d2\log(2\pi)
-\frac12\log|\mathbf H|.
\]

Os coeficientes não lineares usam priors normais independentes da variância
residual. Essa escolha e a aproximação de Laplace fazem do exponencial um modelo
experimental nesta versão.

## 10. Preditiva conjugada

No ponto futuro \(\mathbf x_*\):

\[
z_*\mid\mathbf z,c
\sim t_{2a_n}
\left(
\mathbf x_*^T\mathbf b_n,
\frac{d_n}{a_n}
[1+\mathbf x_*^T\mathbf V_n\mathbf x_*]
\right).
\]

Para o exponencial, a preditiva é simulada a partir da aproximação normal dos
parâmetros, adicionando o ruído residual.

## 11. Mistura preditiva

\[
p(y_*\mid\mathbf y)=
\sum_c p(y_*\mid\mathbf y,c)P(c\mid\mathbf y).
\]

Os quantis são estimados por Monte Carlo. A semente é configurável para tornar
o resultado reproduzível.

## 12. Decomposição da variância

\[
\operatorname{Var}(Y_*)=
E_c[\operatorname{Var}(Y_*\mid c)]
+
\operatorname{Var}_c(E[Y_*\mid c]).
\]

O primeiro termo é retornado como `within_hypothesis_variance`. O segundo é
`between_hypothesis_variance`.

O índice `predictive_uncertainty_index` é a média, ao longo do horizonte, da
variância total dividida por (s_y^2). O índice
`selection_uncertainty_index` aplica a mesma normalização apenas à variância
entre hipóteses. Ambos são adimensionais; o segundo isola a incerteza que uma
escolha humana de janela ou modelo pode potencialmente reduzir.

## 13. Diagnóstico numérico

Para cada hipótese conjugada, o pacote calcula o número de condição da matriz
de desenho. Valores muito altos indicam que diferentes combinações de
coeficientes produzem quase o mesmo ajuste, tornando inversões e extrapolações
instáveis. Hipóteses acima de `max_design_condition_number` são rejeitadas; o
resultado expõe os valores da hipótese escolhida e o máximo observado.

## 14. Decisão de revisão

A implementação inicial usa:

\[
R=
M\times
\frac1H
\sum_{h=1}^H
\frac{V_{entre,h}}{s_y^2},
\]

em que \(M\) é a materialidade.

Solicita-se revisão quando:

\[
R>C_{review}.
\]

ou quando a recomendação é sensível à faixa \(g\in\{n/2,n,2n\}\).

`review_cost` é uma política de negócio. Ele não altera a inferência nem as
probabilidades posteriores.

## 15. Inspeção e decisão humana

O resultado separa três objetos:

- `forecast`: mistura sobre todas as janelas e modelos;
- `model_forecasts`: uma mistura sobre janelas condicionada a cada modelo;
- `model_views`: ajuste e forecast de cada modelo, condicionados à janela
  selecionada.

Um override humano não recalcula nem apaga as probabilidades automáticas. Ele
somente define a combinação usada em `selected_forecast`, mantendo recomendação
e decisão em campos separados para auditoria.

## 16. Limites de interpretação

As probabilidades são condicionais às famílias de modelos, às priors e às
hipóteses de erro. Elas não são garantia de que a realidade pertença ao conjunto
de candidatos.

Antes de uso decisório, recomenda-se:

- análise de sensibilidade à força da g-prior, já produzida pelo pacote;
- testes sintéticos de recuperação de regime;
- posterior predictive checks;
- validação prospectiva dos intervalos;
- avaliação separada por materialidade e horizonte.
