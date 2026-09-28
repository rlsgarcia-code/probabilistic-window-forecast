# Overview matemático da metodologia

> **Nota de visualização:** alguns visualizadores de Markdown exibem LaTeX como
> texto literal. Para ver as equações renderizadas, abra
> [`overview_matematico.ipynb`](../notebooks/overview_matematico.ipynb). Este
> arquivo `.md` é mantido como fonte textual pesquisável.

Este texto apresenta a matemática do método em uma única sequência lógica. A
intenção é mostrar **o que cada objeto representa**, **por que ele aparece** e
**como uma etapa alimenta a seguinte**.

O problema é inferir, a partir de uma série observada até hoje:

1. quando pode ter começado o regime corrente;
2. qual função simples descreve esse regime;
3. qual é a projeção futura;
4. quanta incerteza vem do ruído, dos coeficientes e da seleção de janela e
   modelo.

## 1. Notação básica

Temos $n$ observações mensais:

\[
\mathbf y=(y_1,y_2,\ldots,y_n)^T.
\]

Cada hipótese será indicada por:

\[
h=(m,\tau),
\]

em que:

- $m$ é a família funcional do regime corrente;
- \(\tau\) é a data de início desse regime;
- \(\tau=\varnothing\) significa que não ocorreu uma ruptura terminal.

O algoritmo compara várias hipóteses $h$, todas usando o mesmo conjunto de
$n$ observações.

## 2. Centralização e mudança de escala

Primeiro calculamos o centro robusto:

\[
c_y=\operatorname{mediana}(y_1,\ldots,y_n).
\]

Depois calculamos a escala robusta:

\[
s_y=1{,}4826\operatorname{mediana}(|y_t-c_y|).
\]

O fator $1{,}4826$ coloca o MAD aproximadamente na mesma escala do
desvio-padrão quando os dados são normais. Se o MAD for praticamente zero, a
implementação usa uma escala de segurança.

Trabalhamos internamente com:

\[
z_t=\frac{y_t-c_y}{s_y}.
\]

Essa transformação não escolhe janela nem modelo. O mesmo $c_y$ e o mesmo
$s_y$ são usados em todas as hipóteses da série.

Por que centralizar? Porque a prior dos coeficientes será centrada em zero.
Depois da transformação, zero significa o nível típico da própria série, não
um valor real igual a zero.

Ao final, voltamos à unidade original por:

\[
y_t=c_y+s_yz_t.
\]

## 3. Funções candidatas do regime corrente

Dentro de um regime, definimos $u=0,1,2,\ldots$ como a idade do ponto desde o
início daquele regime.

### Constante

\[
f_{\text{constante}}(u)=\alpha.
\]

Seu vetor de base é:

\[
\boldsymbol\phi_{\text{constante}}(u)=[1].
\]

### Linear

\[
f_{\text{linear}}(u)=\alpha+bu.
\]

Seu vetor de base é:

\[
\boldsymbol\phi_{\text{linear}}(u)=[1,u].
\]

### Logarítmica

\[
f_{\log}(u)=\alpha+b\log(1+u).
\]

Seu vetor de base é:

\[
\boldsymbol\phi_{\log}(u)=[1,\log(1+u)].
\]

A curva logarítmica permite crescimento ou queda que perde intensidade com o
tempo. Ela é uma curva simples e menos explosiva que um polinômio quadrático.

## 4. Como a hipótese vira uma matriz

### 4.1 Hipótese sem ruptura

Se \(\tau=\varnothing\), a mesma função descreve todo o lookback. A linha $t$
da matriz de desenho é:

\[
\mathbf x_t=\boldsymbol\phi_m(t-1).
\]

Empilhando todas as linhas:

\[
\mathbf X_h=
\begin{bmatrix}
\mathbf x_1\\
\mathbf x_2\\
\vdots\\
\mathbf x_n
\end{bmatrix}.
\]

### 4.2 Hipótese com ruptura em \(\tau=k\)

Os pontos anteriores a $k$ recebem uma reta auxiliar:

\[
g_{\text{anterior}}(t)=a_{pre}+b_{pre}t.
\]

O regime corrente recebe a função candidata $f_m(t-k)$. O vetor de
coeficientes passa a ser:

\[
\boldsymbol\beta_h=
\begin{bmatrix}
a_{pre} & b_{pre} & \boldsymbol\theta_m^T
\end{bmatrix}^T,
\]

em que \(\boldsymbol\theta_m\) contém os coeficientes do modelo atual.

A linha da matriz é construída em blocos:

\[
\mathbf x_t=
\begin{cases}
[1,t,\mathbf 0], & t<k,\\
[0,0,\boldsymbol\phi_m(t-k)], & t\ge k.
\end{cases}
\]

Os zeros desligam os coeficientes do regime ao qual aquele ponto não pertence.

Esse desenho é importante: uma janela curta não vence por simplesmente ignorar
o passado. Toda hipótese precisa explicar os mesmos $n$ pontos.

## 5. Modelo probabilístico dos dados

Para uma hipótese $h$, escrevemos:

\[
\mathbf z=\mathbf X_h\boldsymbol\beta_h+\boldsymbol\varepsilon,
\]

com:

\[
\boldsymbol\varepsilon\sim
N(\mathbf 0,\sigma^2\mathbf I).
\]

Equivalentemente:

\[
\mathbf z\mid\boldsymbol\beta_h,\sigma^2,h
\sim
N(\mathbf X_h\boldsymbol\beta_h,\sigma^2\mathbf I).
\]

Essa é a verossimilhança. Ela diz que os desvios em torno da função escolhida
são aditivos, normais, independentes e têm a mesma variância.

## 6. De onde vem \(\mathbf X^T\mathbf X\)

A soma dos erros quadráticos é:

\[
(\mathbf z-\mathbf X\boldsymbol\beta)^T
(\mathbf z-\mathbf X\boldsymbol\beta).
\]

Expandindo:

\[
\mathbf z^T\mathbf z
-2\boldsymbol\beta^T\mathbf X^T\mathbf z
+\boldsymbol\beta^T\mathbf X^T\mathbf X\boldsymbol\beta.
\]

Os dois objetos que carregam informação sobre os coeficientes são:

\[
\mathbf X^T\mathbf X
\quad\text{e}\quad
\mathbf X^T\mathbf z.
\]

$\mathbf X^T\mathbf X$ determina a curvatura da penalidade: quanto mais ela
cresce em uma direção, mais precisamente essa combinação de coeficientes foi
medida. Por isso ela é chamada de matriz de informação ou precisão dos dados.

$\mathbf X^T\mathbf z$ aponta a direção preferida pelos dados.

## 7. Prior dos coeficientes

Usamos:

\[
\boldsymbol\beta_h\mid\sigma^2,h
\sim
N(\mathbf b_0,\sigma^2\mathbf V_{0,h}),
\]

com:

\[
\mathbf b_0=\mathbf 0.
\]

O centro zero significa nível centralizado e ausência de tendência como ponto
de referência.

### 7.1 Construção da g-prior

Se \(\mathbf X^T\mathbf X\) mede precisão, sua inversa tem a geometria de uma
covariância. Portanto definimos:

\[
\boxed{
\mathbf V_{0,h}=g(\mathbf X_h^T\mathbf X_h)^{-1}
}
\]

e, consequentemente:

\[
\mathbf V_{0,h}^{-1}
=\frac{1}{g}\mathbf X_h^T\mathbf X_h.
\]

A precisão da prior é $1/g$ vezes a precisão da amostra completa.

O padrão é:

\[
g=n.
\]

Logo:

\[
\mathbf V_{0,h}^{-1}
=\frac{1}{n}\mathbf X_h^T\mathbf X_h,
\]

que corresponde aproximadamente à informação média de uma observação. Essa é
a interpretação de “prior de informação unitária”.

## 8. Prior da variância residual

Usamos:

\[
\sigma^2\sim\operatorname{InverseGamma}(a_0,d_0).
\]

Na parametrização do pacote, sua densidade é proporcional a:

\[
(\sigma^2)^{-(a_0+1)}
\exp\left(-\frac{d_0}{\sigma^2}\right).
\]

$a_0$ controla a forma e a quantidade de regularização. $d_0$ define a
escala anterior do ruído na série já padronizada.

## 9. Posterior dos coeficientes

A penalidade quadrática dos dados contém:

\[
\boldsymbol\beta^T\mathbf X^T\mathbf X\boldsymbol\beta.
\]

A penalidade quadrática da prior contém:

\[
(\boldsymbol\beta-\mathbf b_0)^T
\mathbf V_0^{-1}
(\boldsymbol\beta-\mathbf b_0).
\]

Multiplicar verossimilhança e prior equivale a somar seus logaritmos. Logo, as
precisões se somam:

\[
\mathbf V_n^{-1}
=\mathbf V_0^{-1}+\mathbf X^T\mathbf X.
\]

Portanto:

\[
\boxed{
\mathbf V_n=
(\mathbf V_0^{-1}+\mathbf X^T\mathbf X)^{-1}
}
\]

e o novo centro é:

\[
\boxed{
\mathbf b_n=
\mathbf V_n
(\mathbf V_0^{-1}\mathbf b_0+\mathbf X^T\mathbf z)
}.
\]

Aqui o segundo “+” representa soma matricial dentro dos parênteses; a expressão
equivalente em código é:

```python
bn = Vn @ (prior_precision @ b0 + X.T @ z)
```

Como \(\mathbf b_0=0\) e usamos a g-prior:

\[
\mathbf V_n^{-1}
=\left(1+\frac1g\right)\mathbf X^T\mathbf X.
\]

Se:

\[
\hat{\boldsymbol\beta}_{OLS}
=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf z,
\]

então:

\[
\boxed{
\mathbf b_n=\frac{g}{g+1}
\hat{\boldsymbol\beta}_{OLS}
}.
\]

Assim, o efeito de $g$ fica explícito. Com $g=n$, a estimativa de mínimos
quadrados é multiplicada por $n/(n+1)$.

## 10. Posterior da variância residual

A forma atualizada é:

\[
a_n=a_0+\frac n2.
\]

A escala atualizada é:

\[
d_n=d_0+\frac12
\left(
\mathbf z^T\mathbf z
+\mathbf b_0^T\mathbf V_0^{-1}\mathbf b_0
-\mathbf b_n^T\mathbf V_n^{-1}\mathbf b_n
\right).
\]

Como \(\mathbf b_0=0\), o termo central desaparece. A expressão entre
parênteses é o que resta da soma de quadrados depois de deslocar a distribuição
para o novo centro \(\mathbf b_n\).

A média posterior da variância, quando $a_n>1$, é:

\[
E[\sigma^2\mid\mathbf z,h]=\frac{d_n}{a_n-1}.
\]

## 11. Evidência de uma hipótese

Para comparar janela e modelo, não basta avaliar o erro no melhor coeficiente.
Integramos todos os valores possíveis de \(\boldsymbol\beta\) e \(\sigma^2\):

\[
p(\mathbf z\mid h)
=\int\int
p(\mathbf z\mid\boldsymbol\beta,\sigma^2,h)
p(\boldsymbol\beta\mid\sigma^2,h)
p(\sigma^2)
\,d\boldsymbol\beta\,d\sigma^2.
\]

Para a prior conjugada, o resultado é:

\[
\boxed{
p(\mathbf z\mid h)=
(2\pi)^{-n/2}
\frac{|\mathbf V_n|^{1/2}}{|\mathbf V_0|^{1/2}}
\frac{d_0^{a_0}}{d_n^{a_n}}
\frac{\Gamma(a_n)}{\Gamma(a_0)}
}.
\]

Interpretação dos termos:

- \(d_n\) reflete a qualidade do ajuste;
- o quociente de determinantes mede quanto o volume plausível dos coeficientes
  foi comprimido;
- as funções Gamma vêm da integração da variância residual;
- modelos flexíveis só vencem se o ganho de ajuste compensar o espaço adicional
  de parâmetros.

O cálculo computacional usa logaritmos para evitar estouro ou perda de
precisão.

## 12. Priors de janela e modelo

Se \(\pi_0\) é a probabilidade anterior de não haver ruptura:

\[
P(\tau=\varnothing)=\pi_0.
\]

Se existem $K$ datas de ruptura elegíveis:

\[
P(\tau=k)=\frac{1-\pi_0}{K}.
\]

Cada família também recebe uma prior $P(m)$. No padrão, os modelos recebem
pesos iguais.

A prior conjunta da hipótese é:

\[
P(h)=P(m)P(\tau).
\]

## 13. Probabilidade posterior de cada hipótese

Aplicando Bayes:

\[
\boxed{
P(h\mid\mathbf z)=
\frac{p(\mathbf z\mid h)P(h)}
{\sum_{h'}p(\mathbf z\mid h')P(h')}
}.
\]

Na implementação, calculamos primeiro:

\[
\ell_h=\log p(\mathbf z\mid h)+\log P(h)
\]

e normalizamos com `logsumexp`.

As probabilidades por janela são marginais:

\[
P(\tau\mid\mathbf z)
=\sum_m P(m,\tau\mid\mathbf z).
\]

As probabilidades por modelo também:

\[
P(m\mid\mathbf z)
=\sum_\tau P(m,\tau\mid\mathbf z).
\]

A janela e o modelo recomendados são os maiores valores dessas duas marginais.

## 14. Distribuição preditiva de uma hipótese

Considere a linha de desenho futura \(\mathbf x_*\). Se a hipótese tem ruptura,
essa linha utiliza somente o bloco do regime corrente.

Depois de integrar coeficientes e variância residual:

\[
z_*\mid\mathbf z,h
\sim
t_{2a_n}
\left(
\mu_h,
q_h^2
\right),
\]

em que:

\[
\mu_h=\mathbf x_*^T\mathbf b_n
\]

e

\[
q_h^2=
\frac{d_n}{a_n}
\left(1+\mathbf x_*^T\mathbf V_n\mathbf x_*\right).
\]

O termo $1$ representa um novo erro residual. O termo
\(\mathbf x_*^T\mathbf V_n\mathbf x_*\) representa a incerteza sobre os
coeficientes projetada naquele ponto futuro.

Voltando à unidade original:

\[
E[Y_*\mid\mathbf y,h]=c_y+s_y\mu_h.
\]

Se \(\nu=2a_n>2\), a variância condicional é:

\[
\operatorname{Var}(Y_*\mid\mathbf y,h)
=s_y^2q_h^2\frac{\nu}{\nu-2}.
\]

## 15. Mistura preditiva

Ainda existe incerteza sobre a própria hipótese. Logo, a preditiva final é:

\[
\boxed{
p(y_*\mid\mathbf y)
=\sum_h
p(y_*\mid\mathbf y,h)
P(h\mid\mathbf y)
}.
\]

Se $w_h=P(h\mid\mathbf y)$, \(\mu_h\) é a média na escala original e $v_h$
é a variância da hipótese, a média da mistura é:

\[
\mu=\sum_h w_h\mu_h.
\]

A variância total é:

\[
\boxed{
\operatorname{Var}(Y_*)
=
\underbrace{\sum_h w_hv_h}_{\text{dentro das hipóteses}}
+
\underbrace{\sum_h w_h(\mu_h-\mu)^2}_{\text{entre hipóteses}}
}.
\]

O primeiro termo contém ruído e incerteza dos coeficientes. O segundo mede a
discordância causada pela seleção de janela e modelo.

Os quantis do intervalo final são calculados por amostragem dessa mistura, pois
uma mistura de distribuições Student-t não é, em geral, outra Student-t.

## 16. Sensibilidade à força da prior

O pacote repete toda a sequência anterior para:

\[
g\in\left\{\frac n2,n,2n\right\}.
\]

Para cada $g$, recalculamos:

- \(\mathbf V_0\);
- posterior;
- evidências;
- pesos das hipóteses;
- janela e modelo vencedores;
- médias do forecast.

Definimos:

```text
prior_sensitive = verdadeiro
```

quando a janela ou o modelo recomendado muda nessa faixa.

Também medimos a maior amplitude das médias previstas:

\[
S_g=
\max_{1\le r\le H}
\frac{
\max_g \mu_{r,g}-\min_g \mu_{r,g}
}{s_y},
\]

em que $r$ identifica o horizonte projetado. O índice é adimensional por ser
dividido por $s_y$.

## 17. Índices de incerteza e revisão

Para horizonte $H$, o índice de incerteza preditiva é:

\[
U_{total}=
\frac1H\sum_{r=1}^H
\frac{V_{total,r}}{s_y^2}.
\]

O índice de seleção usa apenas a variância entre hipóteses:

\[
U_{seleção}=
\frac1H\sum_{r=1}^H
\frac{V_{entre,r}}{s_y^2}.
\]

Com materialidade $M$, a perda operacional inicial é:

\[
L=M\,U_{seleção}.
\]

A revisão humana é recomendada quando:

\[
L>C_{review}
\]

ou quando `prior_sensitive=True`.

`review_cost` não modifica a posterior. Ele apenas transforma a incerteza já
calculada em uma regra operacional de triagem.

## 18. Condicionamento da matriz

A g-prior exige inverter \(\mathbf X^T\mathbf X\). Isso só é seguro se as
colunas de \(\mathbf X\) forem suficientemente distintas.

O número de condição:

\[
\kappa(\mathbf X)
=\frac{s_{max}(\mathbf X)}{s_{min}(\mathbf X)}
\]

compara o maior e o menor valor singular. Se \(s_{min}\) é muito pequeno,
alguns coeficientes são quase indistinguíveis.

Hipóteses com \(\kappa(\mathbf X)\) acima do limite configurado são rejeitadas
para evitar inversões e extrapolações numericamente instáveis.

## 19. Mapa entre matemática e implementação

| Objeto matemático | Implementação |
|---|---|
| $c_y,s_y,\mathbf z$ | preparação feita em `run` |
| \(\boldsymbol\phi_m(u)\) | `_basis` |
| \(\mathbf X_h\) | `_design_matrix` |
| \(\mathbf V_0,\mathbf V_n,\mathbf b_n,a_n,d_n\) | `_fit_hypothesis` |
| \(p(\mathbf z\mid h)\) | `log_evidence` em `_fit_hypothesis` |
| \(P(h\mid\mathbf z)\) | `_normalize_hypotheses` |
| \(P(\tau\mid\mathbf z)\) | `_window_probabilities` |
| \(P(m\mid\mathbf z)\) | `_model_probabilities` |
| preditiva de uma hipótese | `_predictive_moments` e `_predictive_samples` |
| mistura preditiva | `_mixture_forecast` |
| sensibilidade em $g$ | `_g_sensitivity` |

## 20. Hipóteses estatísticas e limites

As probabilidades são condicionais a estas suposições:

- os modelos candidatos contêm aproximações úteis do regime atual;
- o erro é aditivo, normal, independente e homocedástico;
- existe no máximo uma ruptura terminal em cada hipótese;
- o trecho histórico anterior pode ser resumido por uma reta auxiliar;
- a prior de informação unitária é uma regularização adequada;
- o lookback contém informação suficiente para distinguir as hipóteses.

Portanto, uma probabilidade posterior alta significa “esta hipótese domina as
alternativas consideradas sob essas suposições”. Ela não prova que a realidade
seja exatamente constante, linear ou logarítmica.

## 21. A cadeia matemática em uma linha

\[
\boxed{
\mathbf y
\longrightarrow
\mathbf z
\longrightarrow
\{\mathbf X_h\}
\longrightarrow
\{\text{posterior}_h,\ p(\mathbf z\mid h)\}
\longrightarrow
\{P(h\mid\mathbf z)\}
\longrightarrow
\sum_h P(h\mid\mathbf z)p(y_*\mid\mathbf y,h)
}
\]

Essa é a metodologia completa: transformar os dados, construir hipóteses,
estimar cada hipótese, compará-las pela evidência e integrar sua incerteza no
forecast final.
