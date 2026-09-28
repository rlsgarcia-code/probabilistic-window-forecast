# Manual detalhado da modelagem probabilística de janela e forecast

## Da série observada à projeção, com derivação de cada objeto matemático

Versão 1.0 - setembro de 2026

Este manual explica, passo a passo, a metodologia implementada no pacote
`probabilistic-window-forecast`. O objetivo não é apresentar fórmulas prontas,
mas mostrar de onde cada expressão vem, o que ela mede e como é usada na etapa
seguinte.

O problema tratado é deliberadamente restrito: projetar uma única série mensal
`X`, identificar qual trecho final é compatível com o regime corrente e comparar
dinâmicas locais simples. A hierarquia de consolidação e as séries fornecidas
pelo usuário ficam fora deste MVP.

> Princípio central: dados históricos distantes não devem influenciar o modelo
> atual como se pertencessem automaticamente ao mesmo regime. Entretanto, eles
> também não devem ser descartados sem custo. Cada hipótese precisa explicar o
> lookback completo e justificar probabilisticamente onde o regime atual começa.

## Como ler este manual

O texto segue a mesma cadeia do notebook `overview_matematico.ipynb`:

1. preparar a série;
2. construir hipóteses de janela e dinâmica;
3. escrever o modelo probabilístico;
4. combinar prior e dados;
5. calcular a evidência de cada hipótese;
6. transformar evidências em probabilidades;
7. projetar cada hipótese;
8. combinar projeções e decompor a incerteza;
9. decidir se o resultado precisa de revisão humana.

Os quadros chamados **Interpretação** traduzem a álgebra. Os quadros chamados
**Na implementação** indicam onde o conceito aparece no código.

![Fluxo completo da metodologia](../tmp/pdfs/manual_fluxo.png)

## 1. Problema estatístico

Temos uma série mensal observada até o instante atual:

$$
\mathbf y=(y_1,y_2,\ldots,y_n)^T
$$

Queremos responder simultaneamente:

- houve uma mudança de regime dentro do lookback?
- se houve, em qual mês o regime atual começou?
- o regime corrente é mais compatível com nível constante, reta ou curva
  logarítmica?
- qual é a distribuição dos valores futuros?
- quanto da incerteza vem do ruído e quanto vem da dúvida sobre janela e modelo?

Não tratamos a janela como uma decisão tomada antes da modelagem. Janela e
modelo formam conjuntamente uma hipótese:

$$
h=(m,\tau)
$$

Aqui, `m` identifica a família funcional e `tau` identifica o início do regime
corrente. O símbolo `tau = vazio` representa a hipótese sem ruptura: uma única
dinâmica explica todo o lookback.

### 1.1 O que significa inferir a janela

Uma hipótese de janela curta não usa somente os pontos recentes e ignora o
restante. Ela usa uma função auxiliar para o período anterior e uma função para
o regime corrente. Assim, todas as hipóteses explicam os mesmos `n` pontos.

Isso impede uma comparação injusta. Se um modelo pudesse apagar observações
difíceis apenas escolhendo uma janela menor, o melhor resultado tenderia a ser a
menor janela permitida.

### 1.2 O que o método entrega

O resultado não é apenas um nome de modelo. Ele contém:

- probabilidade de não ter ocorrido mudança;
- distribuição de probabilidade sobre possíveis inícios de regime;
- probabilidade marginal de cada família funcional;
- forecast automático, integrando todas as hipóteses;
- forecast condicionado à janela e ao modelo selecionados;
- intervalos preditivos;
- variância dentro das hipóteses e entre hipóteses;
- sensibilidade à força da prior;
- diagnóstico numérico;
- indicação de revisão humana.

## 2. Lookback e hipóteses elegíveis

Se a série tem mais observações que `max_lookback`, apenas as últimas
`max_lookback` observações entram no procedimento. Esse limite é operacional:
ele evita comparar regimes extremamente antigos e controla o custo computacional.

Para uma ruptura candidata no índice `k`, são exigidos:

$$
k\geq n_{antes}
$$

e

$$
n-k\geq n_{depois}
$$

Os parâmetros `min_before` e `min_after` garantem dados mínimos nos dois lados.
Sem essa proteção, uma ruptura muito perto da borda poderia criar coeficientes
praticamente não identificáveis.

Se existem `K` datas elegíveis e `M` modelos candidatos, o número de hipóteses é:

$$
M(K+1)
$$

O termo adicional corresponde à hipótese sem ruptura.

![Série sintética com múltiplos regimes históricos e foco no regime terminal](../tmp/pdfs/manual_regimes.png)

> O procedimento admite no máximo uma ruptura terminal por hipótese. A série
> pode ter tido várias mudanças históricas, mas o bloco anterior é resumido por
> uma reta auxiliar. O alvo da inferência é o início do regime que chega até hoje.

## 3. Centralização e escala robustas

Modelos com unidades muito diferentes criam problemas numéricos e tornam uma
prior fixa impossível de interpretar. Por isso cada série é transformada antes
da estimação.

### 3.1 Centro robusto

Calculamos a mediana:

$$
c_y=\operatorname{mediana}(y_1,\ldots,y_n)
$$

A mediana é menos sensível a poucos valores extremos que a média.

### 3.2 Escala robusta

Primeiro calculamos o desvio absoluto mediano:

$$
MAD=\operatorname{mediana}(|y_t-c_y|)
$$

Depois aplicamos o fator de consistência normal:

$$
s_y=1{,}4826\,MAD
$$

Sob dados aproximadamente normais, `s_y` fica na escala do desvio-padrão. Se o
MAD for quase zero, a implementação usa uma escala de segurança baseada no
desvio-padrão, no nível da série ou em 1.

### 3.3 Série transformada

Cada observação passa a ser:

$$
z_t=\frac{y_t-c_y}{s_y}
$$

Em notação vetorial:

$$
\mathbf z=\frac{\mathbf y-c_y\mathbf 1}{s_y}
$$

O mesmo centro e a mesma escala são usados em todas as hipóteses. Isso é
essencial: evidências só são comparáveis quando representam a mesma variável.

Para retornar à unidade original:

$$
y_t=c_y+s_yz_t
$$

**Interpretação.** Depois da transformação, intercepto zero significa o nível
típico da própria série. Uma inclinação igual a 0,2 representa crescimento de
0,2 escalas robustas por mês, e não crescimento de 0,2 unidade original.

## 4. Dinâmicas candidatas para o regime corrente

Definimos `u = 0, 1, 2, ...` como a idade de cada ponto desde o início do regime
corrente. Usar a idade, e não o índice absoluto da série, torna os coeficientes
comparáveis entre janelas.

### 4.1 Modelo constante

$$
f_{const}(u)=\alpha
$$

Vetor de base:

$$
\boldsymbol\phi_{const}(u)=[1]
$$

O modelo representa um patamar com flutuações aleatórias.

### 4.2 Modelo linear

$$
f_{lin}(u)=\alpha+bu
$$

Vetor de base:

$$
\boldsymbol\phi_{lin}(u)=[1,u]
$$

O coeficiente `b` mede a mudança média por mês na escala padronizada.

### 4.3 Modelo logarítmico

$$
f_{log}(u)=\alpha+b\log(1+u)
$$

Vetor de base:

$$
\boldsymbol\phi_{log}(u)=[1,\log(1+u)]
$$

A derivada é:

$$
\frac{d f_{log}(u)}{du}=\frac{b}{1+u}
$$

Portanto, a direção é controlada pelo sinal de `b`, mas a intensidade diminui
com o tempo. Esse formato oferece curvatura sem a extrapolação explosiva de um
polinômio quadrático.

### 4.4 Por que não escolher visualmente uma família de antemão

Uma reta pode parecer adequada no histórico e ainda produzir projeção frágil
quando sua inclinação é mal estimada. O método compara as famílias pela
probabilidade dos dados depois de integrar a incerteza dos coeficientes. A
visualização continua importante, mas funciona como diagnóstico e possível
intervenção, não como substituto silencioso da inferência.

## 5. Da função à matriz de desenho

Uma regressão linear nos parâmetros pode ser escrita como:

$$
\mathbf z=\mathbf X_h\boldsymbol\beta_h+\boldsymbol\varepsilon
$$

Cada linha de `X_h` informa quais coeficientes participam da média de uma
observação. Cada coluna representa um coeficiente desconhecido.

### 5.1 Hipótese sem ruptura

Para um modelo linear com quatro observações e idades `0, 1, 2, 3`:

$$
\mathbf X=
\left[\begin{array}{cc}
1&0\\
1&1\\
1&2\\
1&3
\end{array}\right]
$$

e

$$
\boldsymbol\beta=[\alpha,b]^T
$$

O produto `X beta` produz:

$$
\mathbf X\boldsymbol\beta=[\alpha,\alpha+b,\alpha+2b,\alpha+3b]^T
$$

### 5.2 Hipótese com ruptura

Suponha uma ruptura em `k`. Antes dela usamos uma reta auxiliar:

$$
g_{pre}(t)=a_{pre}+b_{pre}t
$$

Depois dela usamos a família candidata do regime corrente. Para um regime
corrente linear, o vetor de coeficientes é:

$$
\boldsymbol\beta=[a_{pre},b_{pre},\alpha,b]^T
$$

Uma matriz ilustrativa com três pontos antes e três depois é:

$$
\mathbf X=
\left[\begin{array}{cccc}
1&0&0&0\\
1&1&0&0\\
1&2&0&0\\
0&0&1&0\\
0&0&1&1\\
0&0&1&2
\end{array}\right]
$$

Os zeros desligam o bloco que não pertence à observação. Não é imposta
continuidade na ruptura: uma mudança de nível pode ocorrer junto com uma mudança
de direção.

**Interpretação.** A reta anterior não é o objeto que queremos projetar. Ela é
um mecanismo para cobrar da hipótese uma explicação pelo passado sem forçar o
regime atual a reproduzi-lo.

## 6. Modelo probabilístico dos dados

Para uma hipótese `h`, assumimos:

$$
\mathbf z\mid\boldsymbol\beta_h,\sigma^2,h\sim
N(\mathbf X_h\boldsymbol\beta_h,\sigma^2\mathbf I)
$$

Isso equivale a escrever:

$$
\mathbf z=\mathbf X_h\boldsymbol\beta_h+\boldsymbol\varepsilon
$$

com

$$
\boldsymbol\varepsilon\sim N(\mathbf 0,\sigma^2\mathbf I)
$$

A matriz identidade expressa três suposições:

- mesma variância residual em todos os meses;
- ausência de correlação serial residual;
- erro aditivo na escala transformada.

A densidade é o produto de um fator de normalização,

$$
(2\pi\sigma^2)^{-n/2},
$$

e de um fator que diminui com o erro quadrático:

$$
\exp\left[-\frac{1}{2\sigma^2}
(\mathbf z-\mathbf X\boldsymbol\beta)^T
(\mathbf z-\mathbf X\boldsymbol\beta)\right].
$$

Portanto, a densidade completa é o produto desses dois fatores.

O expoente penaliza a soma de erros quadráticos. Quanto maior o erro, menor a
verossimilhança.

## 7. Origem de `X transposta X` e `X transposta z`

Começamos pela soma de quadrados:

$$
Q(\boldsymbol\beta)=
(\mathbf z-\mathbf X\boldsymbol\beta)^T
(\mathbf z-\mathbf X\boldsymbol\beta)
$$

Distribuindo o produto:

$$
Q(\boldsymbol\beta)=
\mathbf z^T\mathbf z
-\mathbf z^T\mathbf X\boldsymbol\beta
-\boldsymbol\beta^T\mathbf X^T\mathbf z
+\boldsymbol\beta^T\mathbf X^T\mathbf X\boldsymbol\beta
$$

O produto `z transposta X beta` é um escalar. A transposta de um escalar é o
próprio escalar, portanto:

$$
\mathbf z^T\mathbf X\boldsymbol\beta
=\boldsymbol\beta^T\mathbf X^T\mathbf z
$$

Assim:

$$
Q(\boldsymbol\beta)=
\mathbf z^T\mathbf z
-2\boldsymbol\beta^T\mathbf X^T\mathbf z
+\boldsymbol\beta^T\mathbf X^T\mathbf X\boldsymbol\beta
$$

Agora aparecem dois objetos:

$$
\mathbf X^T\mathbf X
\quad\mathrm{e}\quad
\mathbf X^T\mathbf z
$$

`X transposta X` controla a curvatura de `Q`. Direções com grande curvatura são
fortemente penalizadas quando se afastam do melhor ajuste e, por isso, são
estimadas com maior precisão. `X transposta z` aponta para a combinação de
coeficientes favorecida pelos dados.

Derivando `Q` em relação a `beta`:

$$
\frac{\partial Q}{\partial\boldsymbol\beta}
=-2\mathbf X^T\mathbf z+2\mathbf X^T\mathbf X\boldsymbol\beta
$$

No mínimo, a derivada vale zero:

$$
\mathbf X^T\mathbf X\widehat{\boldsymbol\beta}_{OLS}
=\mathbf X^T\mathbf z
$$

Se `X transposta X` for invertível:

$$
\widehat{\boldsymbol\beta}_{OLS}
=(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf z
$$

## 8. Prior normal dos coeficientes

Antes de observar os dados, usamos:

$$
\boldsymbol\beta\mid\sigma^2,h
\sim N(\mathbf b_0,\sigma^2\mathbf V_0)
$$

Há três objetos diferentes nessa expressão:

- `b0`: centro anterior dos coeficientes;
- `V0`: geometria e dispersão relativa da prior;
- `sigma quadrado`: escala global de ruído compartilhada com a verossimilhança.

No pacote:

$$
\mathbf b_0=\mathbf 0
$$

Depois da centralização, isso representa ausência de deslocamento e tendência
como referência fraca, não a afirmação de que a série real vale zero.

### 8.1 Por que `V0` não é escolhida diagonalmente

Uma variância fixa para cada coeficiente depende das unidades das colunas de
`X`. A coluna do intercepto e a coluna de tempo têm escalas diferentes. Além
disso, a escala da coluna temporal muda com o tamanho da janela.

A g-prior usa a geometria do próprio desenho:

$$
\mathbf V_0=g(\mathbf X^T\mathbf X)^{-1}
$$

Logo, sua precisão é:

$$
\mathbf V_0^{-1}=\frac{1}{g}\mathbf X^T\mathbf X
$$

Essa escolha adapta a prior à unidade e à correlação das colunas. O parâmetro
escalar `g` controla sua força.

### 8.2 Interpretação de `g`

Precisão maior significa distribuição mais concentrada. Como a precisão da
prior é `1/g` vezes a precisão total dos dados:

- `g` pequeno cria regularização mais forte;
- `g` grande cria prior mais difusa;
- `g = n` produz aproximadamente a informação média de uma observação.

O padrão do pacote é:

$$
g=n
$$

Essa convenção é chamada prior de informação unitária. Ela é reproduzível, mas
não é uma verdade sobre o negócio. Por isso sua sensibilidade é medida.

## 9. Prior da variância residual

Usamos uma distribuição Inverse-Gamma:

$$
\sigma^2\sim IG(a_0,d_0)
$$

Na parametrização do pacote:

$$
p(\sigma^2)\propto
(\sigma^2)^{-(a_0+1)}
\exp\left(-\frac{d_0}{\sigma^2}\right)
$$

`a0` controla a forma e a quantidade de regularização. `d0` determina a escala
anterior do ruído depois da padronização robusta. Os padrões são fracos, mas
evitam problemas em amostras curtas ou ajustes quase perfeitos.

## 10. Posterior dos coeficientes, sem saltos algébricos

Esta é a etapa em que a informação anterior e a informação dos dados são
combinadas.

A parte da log-verossimilhança que depende de `beta` é proporcional a:

$$
-\frac{1}{2\sigma^2}
(\mathbf z-\mathbf X\boldsymbol\beta)^T
(\mathbf z-\mathbf X\boldsymbol\beta)
$$

A parte da log-prior que depende de `beta` é proporcional a:

$$
-\frac{1}{2\sigma^2}
(\boldsymbol\beta-\mathbf b_0)^T
\mathbf V_0^{-1}
(\boldsymbol\beta-\mathbf b_0)
$$

Multiplicar densidades equivale a somar logaritmos. Portanto, somamos as duas
penalidades.

### 10.1 Expansão da penalidade dos dados

$$
(\mathbf z-\mathbf X\boldsymbol\beta)^T
(\mathbf z-\mathbf X\boldsymbol\beta)
=\mathbf z^T\mathbf z
-2\boldsymbol\beta^T\mathbf X^T\mathbf z
+\boldsymbol\beta^T\mathbf X^T\mathbf X\boldsymbol\beta
$$

### 10.2 Expansão da penalidade da prior

Como `V0 inversa` é simétrica:

$$
(\boldsymbol\beta-\mathbf b_0)^T
\mathbf V_0^{-1}
(\boldsymbol\beta-\mathbf b_0)
=\boldsymbol\beta^T\mathbf V_0^{-1}\boldsymbol\beta
-2\boldsymbol\beta^T\mathbf V_0^{-1}\mathbf b_0
+\mathbf b_0^T\mathbf V_0^{-1}\mathbf b_0
$$

### 10.3 Agrupamento dos termos

Somando as expansões e agrupando por potência de `beta`:

$$
\boldsymbol\beta^T
(\mathbf X^T\mathbf X+\mathbf V_0^{-1})
\boldsymbol\beta
-2\boldsymbol\beta^T
(\mathbf X^T\mathbf z+\mathbf V_0^{-1}\mathbf b_0)
+C
$$

`C` reúne os termos que não dependem de `beta`:

$$
C=\mathbf z^T\mathbf z+\mathbf b_0^T\mathbf V_0^{-1}\mathbf b_0
$$

Os termos em `beta` não se anulam. O termo quadrático recebe a soma das
precisões, e o termo linear recebe a soma das informações que determinam o
centro.

Definimos:

$$
\mathbf A=\mathbf X^T\mathbf X+\mathbf V_0^{-1}
$$

e

$$
\mathbf c=\mathbf X^T\mathbf z+\mathbf V_0^{-1}\mathbf b_0
$$

A expressão dependente de `beta` é:

$$
\boldsymbol\beta^T\mathbf A\boldsymbol\beta
-2\boldsymbol\beta^T\mathbf c
$$

### 10.4 Completação de quadrados

Queremos reescrever a expressão em torno de um novo centro `bn`. Expandimos:

$$
(\boldsymbol\beta-\mathbf b_n)^T
\mathbf A
(\boldsymbol\beta-\mathbf b_n)
$$

O resultado é:

$$
\boldsymbol\beta^T\mathbf A\boldsymbol\beta
-2\boldsymbol\beta^T\mathbf A\mathbf b_n
+\mathbf b_n^T\mathbf A\mathbf b_n
$$

Para o termo linear coincidir com `-2 beta transposta c`, precisamos de:

$$
\mathbf A\mathbf b_n=\mathbf c
$$

Logo:

$$
\mathbf b_n=\mathbf A^{-1}\mathbf c
$$

Substituindo as definições:

$$
\mathbf V_n=(\mathbf V_0^{-1}+\mathbf X^T\mathbf X)^{-1}
$$

e

$$
\mathbf b_n=\mathbf V_n
(\mathbf V_0^{-1}\mathbf b_0+\mathbf X^T\mathbf z)
$$

Portanto:

$$
\boldsymbol\beta\mid\sigma^2,\mathbf z,h
\sim N(\mathbf b_n,\sigma^2\mathbf V_n)
$$

> Não são as covariâncias que se somam. Somam-se as precisões. A covariância
> posterior é obtida invertendo a precisão total.

### 10.5 Simplificação com g-prior e `b0 = 0`

Com a g-prior:

$$
\mathbf V_0^{-1}=\frac{1}{g}\mathbf X^T\mathbf X
$$

Assim:

$$
\mathbf V_n^{-1}=\left(1+\frac{1}{g}\right)\mathbf X^T\mathbf X
$$

O centro posterior vira:

$$
\mathbf b_n=\frac{g}{g+1}
(\mathbf X^T\mathbf X)^{-1}\mathbf X^T\mathbf z
$$

Como o termo final é o estimador OLS:

$$
\mathbf b_n=\frac{g}{g+1}\widehat{\boldsymbol\beta}_{OLS}
$$

Para `g = n`, os coeficientes OLS são contraídos pelo fator `n/(n+1)`.

## 11. Exemplo numérico mínimo da posterior

Considere três observações padronizadas:

$$
\mathbf z=[0,1,2]^T
$$

e um modelo linear com:

$$
\mathbf X=
\left[\begin{array}{cc}
1&0\\
1&1\\
1&2
\end{array}\right]
$$

Primeiro:

$$
\mathbf X^T\mathbf X=
\left[\begin{array}{cc}
3&3\\
3&5
\end{array}\right]
$$

e

$$
\mathbf X^T\mathbf z=[3,5]^T
$$

O ajuste OLS resolve:

$$
\left[\begin{array}{cc}3&3\\3&5\end{array}\right]
\widehat{\boldsymbol\beta}_{OLS}
=[3,5]^T
$$

resultando em:

$$
\widehat{\boldsymbol\beta}_{OLS}=[0,1]^T
$$

Com `g = n = 3` e `b0 = 0`:

$$
\mathbf b_n=\frac{3}{4}[0,1]^T=[0,0{,}75]^T
$$

A inclinação é levemente puxada para zero. Se `g` fosse 99, o fator seria
`99/100`, quase sem regularização. Se `g` fosse 1, o fator seria `1/2`.

## 12. Posterior da variância residual

Depois de completar o quadrado em `beta`, a prior Inverse-Gamma também é
atualizada. O parâmetro de forma é:

$$
a_n=a_0+\frac{n}{2}
$$

Cada observação adiciona meia unidade porque a densidade normal contém a
potência `(sigma quadrado) elevado a -n/2`.

O parâmetro de escala é:

$$
d_n=d_0+\frac{1}{2}
\left(
\mathbf z^T\mathbf z
+\mathbf b_0^T\mathbf V_0^{-1}\mathbf b_0
-\mathbf b_n^T\mathbf V_n^{-1}\mathbf b_n
\right)
$$

O primeiro termo mede a energia bruta dos dados. O segundo incorpora a posição
da prior. O terceiro subtrai a parte explicada quando deslocamos a distribuição
para o centro posterior.

Quando `an > 1`, a média posterior é:

$$
E[\sigma^2\mid\mathbf z,h]=\frac{d_n}{a_n-1}
$$

Essa quantidade é retornada na unidade original após multiplicação por
`s_y` ao quadrado.

## 13. Evidência marginal de uma hipótese

Comparar apenas o erro no melhor ajuste favoreceria modelos flexíveis. A
evidência marginal avalia toda a região de parâmetros permitida pela prior:

$$
p(\mathbf z\mid h)=
\int\int
p(\mathbf z\mid\boldsymbol\beta,\sigma^2,h)
p(\boldsymbol\beta\mid\sigma^2,h)
p(\sigma^2)
\,d\boldsymbol\beta\,d\sigma^2
$$

### 13.1 Integração dos coeficientes

A completação de quadrados transforma a parte em `beta` numa densidade normal
posterior. Integrar essa densidade produz o fator:

$$
\frac{|\mathbf V_n|^{1/2}}{|\mathbf V_0|^{1/2}}
$$

O determinante mede volume. Se apenas uma região muito pequena do espaço
anterior explica os dados, há forte compressão de volume. Esse é um componente
da penalização automática de complexidade.

### 13.2 Integração da variância

Depois da integração de `beta`, resta o núcleo de uma Inverse-Gamma com
parâmetros `an` e `dn`. Sua integral produz:

$$
\frac{d_0^{a_0}}{d_n^{a_n}}
\frac{\Gamma(a_n)}{\Gamma(a_0)}
$$

Juntando os fatores, a evidência pode ser lida como o produto de duas partes. A
primeira vem da normal e da integração dos coeficientes:

$$
(2\pi)^{-n/2}
\frac{|\mathbf V_n|^{1/2}}{|\mathbf V_0|^{1/2}}.
$$

A segunda vem da integração da variância:

$$
\frac{d_0^{a_0}}{d_n^{a_n}}
\frac{\Gamma(a_n)}{\Gamma(a_0)}.
$$

Logo, `p(z | h)` é o produto dessas duas expressões.

**Interpretação.** Um modelo vence quando encontra bom compromisso entre ajuste
e volume de parâmetros plausíveis. Flexibilidade só ajuda se melhorar os dados
o suficiente para compensar a dispersão adicional.

Na implementação, a evidência é calculada em logaritmos:

$$
\log p(\mathbf z\mid h)
$$

Isso evita underflow ao multiplicar números muito pequenos.

## 14. Priors sobre janela e modelo

Além das priors dos parâmetros, precisamos de uma prior sobre as hipóteses.

Se `pi0` é a probabilidade anterior de não haver ruptura:

$$
P(\tau=\varnothing)=\pi_0
$$

Se há `K` datas elegíveis:

$$
P(\tau=k)=\frac{1-\pi_0}{K}
$$

Cada modelo recebe peso `P(m)`. A prior conjunta é:

$$
P(h)=P(m)P(\tau)
$$

O padrão distribui igualmente a massa entre as famílias e usa probabilidade
anterior de 0,60 para a hipótese sem mudança. Isso expressa parcimônia: uma
ruptura precisa de evidência para superar a alternativa mais simples.

## 15. Probabilidade posterior das hipóteses

Aplicando o teorema de Bayes:

$$
P(h\mid\mathbf z)=
\frac{p(\mathbf z\mid h)P(h)}
{\sum_{h'}p(\mathbf z\mid h')P(h')}
$$

Computacionalmente, definimos:

$$
\ell_h=\log p(\mathbf z\mid h)+\log P(h)
$$

e calculamos:

$$
P(h\mid\mathbf z)=
\exp\left(\ell_h-\operatorname{logsumexp}(\ell_1,\ldots,\ell_H)\right)
$$

O `logsumexp` normaliza sem exponenciar diretamente números extremos.

### 15.1 Probabilidade de uma janela

Uma mesma janela aparece combinada com vários modelos. Sua probabilidade é a
soma das hipóteses correspondentes:

$$
P(\tau\mid\mathbf z)=\sum_m P(m,\tau\mid\mathbf z)
$$

### 15.2 Probabilidade de um modelo

Analogamente:

$$
P(m\mid\mathbf z)=\sum_\tau P(m,\tau\mid\mathbf z)
$$

Essa marginalização é importante. A janela recomendada não precisa pertencer à
mesma hipótese conjunta que possui a maior probabilidade individual; ela é a
janela com maior suporte somado entre modelos.

### 15.3 Conjunto crível de janelas

Ordenamos as janelas por probabilidade e as acumulamos até atingir
`credible_mass`, cujo padrão é 90%. O conjunto resultante mostra quais datas de
início concentram a maior parte da plausibilidade posterior.

## 16. Distribuição preditiva de uma hipótese

Para um horizonte futuro, construímos a linha de desenho `x estrela`. Em uma
hipótese com ruptura, ela utiliza apenas o bloco do regime corrente; o bloco
anterior recebe zeros.

Condicionalmente a `beta` e `sigma quadrado`:

$$
z_*\mid\boldsymbol\beta,\sigma^2,h
\sim N(\mathbf x_*^T\boldsymbol\beta,\sigma^2)
$$

Integrar `beta` adiciona a incerteza dos coeficientes:

$$
z_*\mid\sigma^2,\mathbf z,h
\sim N\left(
\mathbf x_*^T\mathbf b_n,
\sigma^2(1+\mathbf x_*^T\mathbf V_n\mathbf x_*)
\right)
$$

O termo 1 representa um novo choque residual. O termo quadrático representa a
incerteza dos coeficientes projetada na direção do ponto futuro.

Integrar `sigma quadrado` produz uma Student-t:

$$
z_*\mid\mathbf z,h
\sim t_{2a_n}(\mu_h,q_h^2)
$$

com

$$
\mu_h=\mathbf x_*^T\mathbf b_n
$$

e

$$
q_h^2=\frac{d_n}{a_n}
(1+\mathbf x_*^T\mathbf V_n\mathbf x_*)
$$

Se `nu = 2 an` é maior que 2, a variância é:

$$
\operatorname{Var}(z_*\mid\mathbf z,h)
=q_h^2\frac{\nu}{\nu-2}
$$

Na unidade original:

$$
E[Y_*\mid\mathbf y,h]=c_y+s_y\mu_h
$$

e

$$
\operatorname{Var}(Y_*\mid\mathbf y,h)
=s_y^2\operatorname{Var}(z_*\mid\mathbf z,h)
$$

## 17. Forecast como mistura de hipóteses

Selecionar uma hipótese e fingir certeza descartaria informação. O forecast
automático integra janela e modelo:

$$
p(y_*\mid\mathbf y)=
\sum_h p(y_*\mid\mathbf y,h)P(h\mid\mathbf y)
$$

Se `w_h` é o peso posterior e `mu_h` é a média da hipótese:

$$
\mu=\sum_h w_h\mu_h
$$

A lei da variância total fornece:

$$
\operatorname{Var}(Y_*)=
\sum_h w_hv_h
+\sum_h w_h(\mu_h-\mu)^2
$$

O primeiro termo é a variância **dentro das hipóteses**. Ele inclui ruído futuro
e incerteza dos coeficientes. O segundo é a variância **entre hipóteses**. Ele
mede a discordância provocada por diferentes janelas e modelos.

![Decomposição conceitual da incerteza preditiva](../tmp/pdfs/manual_incerteza.png)

### 17.1 Por que os quantis são simulados

Uma média ponderada de distribuições Student-t geralmente não é outra Student-t.
Por isso o pacote:

1. sorteia quantas amostras virão de cada hipótese de acordo com seus pesos;
2. gera amostras da preditiva de cada hipótese;
3. concatena as amostras;
4. calcula mediana e quantis empíricos.

O intervalo é preditivo: ele inclui um novo erro residual. Portanto, é mais
largo que um intervalo apenas para a média estimada.

## 18. Sensibilidade à força da prior

Uma resposta automática não deve depender de uma escolha arbitrária não
inspecionada. O pacote repete a inferência para:

$$
g\in\{n/2,n,2n\}
$$

Para cada valor, recalcula prior, posterior, evidências, probabilidades e
forecast.

`prior_sensitive` é verdadeiro quando a janela ou o modelo recomendado muda.

A amplitude normalizada das médias projetadas é:

$$
S_g=\max_{1\leq r\leq H}
\frac{\max_g\mu_{r,g}-\min_g\mu_{r,g}}{s_y}
$$

Dividir por `s_y` permite comparar séries em unidades diferentes.

**Interpretação.** Sensibilidade não prova que a prior esteja errada. Ela mostra
que os dados disponíveis não dominam completamente decisões plausíveis sobre
regularização.

## 19. Índices de incerteza e revisão humana

Para horizonte `H`, o índice de incerteza preditiva é:

$$
U_{total}=\frac{1}{H}\sum_{r=1}^{H}
\frac{V_{total,r}}{s_y^2}
$$

O índice de incerteza de seleção usa apenas a parcela entre hipóteses:

$$
U_{sel}=\frac{1}{H}\sum_{r=1}^{H}
\frac{V_{entre,r}}{s_y^2}
$$

Com materialidade `M`:

$$
L=M U_{sel}
$$

A revisão é recomendada quando:

$$
L>C_{review}
$$

ou quando a seleção é sensível a `g`.

O custo de revisão não altera a posterior. Ele é uma regra operacional aplicada
depois da inferência.

### 19.1 O que o revisor pode fazer

O usuário pode:

- selecionar visualmente o início do regime;
- forçar a hipótese sem mudança;
- alternar entre modelos na janela selecionada;
- escolher explicitamente um modelo;
- executar novamente e comparar o forecast condicionado.

O resultado registra se janela e modelo vieram da decisão automática ou manual.

## 20. Condicionamento numérico

A g-prior exige inverter `X transposta X`. Se duas colunas de `X` forem quase
combinações uma da outra, alguns coeficientes não são separadamente identificados.

O número de condição é:

$$
\kappa(\mathbf X)=\frac{s_{max}(\mathbf X)}{s_{min}(\mathbf X)}
$$

Os termos `s max` e `s min` são o maior e o menor valor singular. Quando o menor
é quase zero, pequenas perturbações nos dados podem causar grandes mudanças nos
coeficientes.

Hipóteses acima de `max_design_condition_number` são rejeitadas. Isso não é um
critério estatístico de escolha do melhor modelo; é uma proteção contra uma
operação numericamente não confiável.

## 21. Algoritmo completo

```text
entrada: observações mensais (valor, t)

1. validar, ordenar conforme recebido e limitar o lookback
2. calcular mediana, MAD e série padronizada z
3. enumerar modelos candidatos
4. enumerar hipótese sem ruptura e todas as rupturas elegíveis
5. para cada hipótese h:
      construir X_h
      verificar condicionamento
      construir g-prior
      calcular Vn, bn, an e dn
      calcular log-evidência
6. adicionar log-prior de cada hipótese
7. normalizar com logsumexp
8. marginalizar probabilidades por janela e por modelo
9. repetir para g em {n/2, n, 2n}
10. para cada horizonte:
      calcular preditiva de cada hipótese
      combinar pela probabilidade posterior
      decompor variância dentro/entre hipóteses
      simular quantis da mistura
11. calcular índices de incerteza e perda operacional
12. emitir recomendação automática e flag de revisão
13. se houver override humano, condicionar e recalcular o forecast selecionado
```

## 22. Correspondência com o código Python

| Conceito | Implementação |
|---|---|
| observação | `Observation` |
| configuração | `BayesianConfig` |
| pipeline público | `BayesianForecastPipeline` |
| base funcional | `_basis` |
| matriz de desenho | `_design_matrix` |
| ajuste de uma hipótese | `_fit_hypothesis` |
| normalização posterior | `_normalize_hypotheses` |
| probabilidades de janela | `_window_probabilities` |
| probabilidades de modelo | `_model_probabilities` |
| conjunto crível | `_credible_set` |
| preditiva de uma hipótese | `_predictive_moments` e `_predictive_samples` |
| mistura | `_mixture_forecast` |
| sensibilidade | `_g_sensitivity` |
| resultado | `BayesianResult` |

### 22.1 Relação direta entre álgebra e NumPy

```python
b0 = np.zeros(design.shape[1])
data_precision = design.T @ design
g_value = len(z) * g_multiplier

v0 = g_value * np.linalg.inv(data_precision)
v0_inv = data_precision / g_value

vn = np.linalg.inv(v0_inv + data_precision)
bn = vn @ (v0_inv @ b0 + design.T @ z)

an = noise_shape + len(z) / 2.0
dn = noise_scale + 0.5 * (
    z @ z
    + b0 @ v0_inv @ b0
    - bn @ np.linalg.solve(vn, bn)
)
```

Linha por linha:

- `design.T @ design` calcula a precisão fornecida pelos dados;
- `v0_inv` é a precisão da g-prior;
- `v0_inv + data_precision` soma as precisões;
- sua inversa é a covariância relativa posterior `vn`;
- `design.T @ z` é a informação linear dos dados;
- `v0_inv @ b0` é a informação linear da prior;
- `bn` resolve o novo centro;
- `an` e `dn` atualizam a distribuição do ruído.

## 23. Como executar o pacote

```python
from probabilistic_window_forecast import (
    BayesianConfig,
    BayesianForecastPipeline,
)

observations = [
    (102.0, "2024-01"),
    (101.5, "2024-02"),
    (103.2, "2024-03"),
    # ...
]

config = BayesianConfig(
    max_lookback=60,
    min_before=8,
    min_after=8,
    forecast_horizon=6,
)

pipeline = BayesianForecastPipeline(config)
result = pipeline.run(observations, series_id="serie_x")
```

Principais leituras:

```python
result.recommended_window_start_t
result.window_credible_set
result.recommended_model
result.model_probabilities
result.forecast
result.selection_uncertainty_index
result.prior_sensitive
result.human_intervention_required
```

### 23.1 Reexecutar com janela escolhida pelo usuário

```python
adjusted = pipeline.run(
    observations,
    series_id="serie_x",
    window_start="2025-01",
)
```

### 23.2 Forçar ausência de ruptura

```python
adjusted = pipeline.run(
    observations,
    series_id="serie_x",
    force_no_change=True,
)
```

### 23.3 Escolher modelo para inspeção

```python
adjusted = pipeline.run(
    observations,
    series_id="serie_x",
    window_start="2025-01",
    model_override="linear",
)
```

## 24. Como interpretar a saída

### 24.1 Forecast automático versus selecionado

`forecast` integra todas as hipóteses com seus pesos posteriores. É a previsão
probabilística oficial antes de decisão humana.

`selected_forecast` condiciona na janela e no modelo efetivamente selecionados.
Ele é útil para visualizar o efeito de um override, mas elimina deliberadamente
parte da incerteza de seleção.

### 24.2 Intervalos

Para cada horizonte, `PredictivePoint` contém:

- `mean`: média da mistura;
- `median`: mediana simulada;
- `lower` e `upper`: limites do intervalo preditivo;
- `within_hypothesis_variance`: ruído e coeficientes;
- `between_hypothesis_variance`: discordância de seleção;
- `total_variance`: soma das duas parcelas.

### 24.3 Confiança

O pacote evita reduzir toda a qualidade a um único score arbitrário. A confiança
deve ser lida por meio de:

- concentração das probabilidades de janela;
- concentração das probabilidades de modelo;
- largura do conjunto crível de janelas;
- proporção da variância entre hipóteses;
- sensibilidade a `g`;
- condicionamento da matriz;
- largura dos intervalos preditivos.

## 25. Validação quando não há futuro observado

No instante da decisão, não existem valores futuros para medir erro de forecast.
Logo, o método não usa um backtest inexistente para escolher a janela atual.

A validação disponível naquele momento é interna e estrutural:

- coerência probabilística das hipóteses;
- evidência marginal no mesmo conjunto de dados;
- concentração ou dispersão da posterior;
- estabilidade a mudanças moderadas da prior;
- diagnóstico numérico;
- inspeção visual das alternativas materiais.

Quando novos meses forem observados, passam a existir avaliações prospectivas.
É recomendável registrar cada forecast emitido sem alterá-lo e, posteriormente,
comparar intervalo, cobertura, viés e erro realizado. Essa avaliação futura é
diferente da inferência usada hoje.

## 26. Limitações e sinais de alerta

As probabilidades são condicionais ao conjunto de modelos e às suposições. Uma
probabilidade alta não prova que a realidade seja exatamente constante, linear
ou logarítmica.

Exigem atenção:

- sazonalidade mensal não modelada;
- resíduos autocorrelacionados;
- variância que cresce com o nível;
- outliers ou dados revisados;
- mais de uma mudança recente importante;
- janela corrente menor que `min_after`;
- choque conhecido que ainda não aparece nos dados;
- projeção em horizonte muito maior que o regime observado;
- alta sensibilidade a `g`;
- intervalo estreito causado por premissas inadequadas, e não por informação.

O método deve ser ampliado ou suspenso quando essas condições forem materiais.

## 27. Por que o método é menos heurístico que uma regra de janela

Uma regra como “usar sempre os últimos 12 meses” impõe a janela sem quantificar
a dúvida. Um change point escolhido apenas pelo menor erro tende a premiar
segmentos curtos. Uma inspeção puramente visual é útil, mas difícil de reproduzir
em muitas séries.

Nesta formulação:

- todas as janelas competem no mesmo conjunto de observações;
- a complexidade é penalizada pela integração bayesiana;
- a preferência por ausência de ruptura é explícita;
- janela e modelo permanecem incertos;
- a incerteza de seleção chega ao forecast;
- escolhas anteriores são submetidas a análise de sensibilidade;
- a revisão humana é acionada por impacto, não por exceção informal.

Ainda existem escolhas de modelagem. A vantagem não é eliminar julgamento, mas
expô-lo, parametrizá-lo e mostrar quando ele influencia a decisão.

## 28. Glossário

**Coeficiente.** Parâmetro que determina nível, inclinação ou forma da função.

**Covariância.** Matriz que mede dispersão conjunta. Na posterior condicional,
ela é `sigma quadrado vezes Vn`.

**Precisão.** Inversa da covariância. Informações independentes aparecem como
somas de precisões.

**Prior.** Distribuição que representa regularização e informação antes dos
dados atuais.

**Verossimilhança.** Densidade dos dados como função dos parâmetros.

**Posterior.** Distribuição dos parâmetros depois de combinar prior e dados.

**Evidência marginal.** Probabilidade dos dados sob uma hipótese depois de
integrar seus parâmetros.

**Hipótese.** Combinação de família funcional e início de regime.

**Mistura preditiva.** Soma ponderada das preditivas das hipóteses.

**Incerteza de seleção.** Variância decorrente da discordância entre hipóteses.

**Intervalo preditivo.** Faixa para uma nova observação, incluindo ruído futuro.

**Conjunto crível de janelas.** Menor conjunto construído por ordenação que
acumula a massa posterior configurada.

## 29. Fórmulas essenciais em sequência

Padronização:

$$
z_t=\frac{y_t-c_y}{s_y}
$$

Modelo dos dados:

$$
\mathbf z=\mathbf X_h\boldsymbol\beta_h+\boldsymbol\varepsilon
$$

g-prior:

$$
\mathbf V_0=g(\mathbf X^T\mathbf X)^{-1}
$$

Precisão posterior:

$$
\mathbf V_n^{-1}=\mathbf V_0^{-1}+\mathbf X^T\mathbf X
$$

Centro posterior:

$$
\mathbf b_n=\mathbf V_n
(\mathbf V_0^{-1}\mathbf b_0+\mathbf X^T\mathbf z)
$$

Variância residual:

$$
a_n=a_0+n/2
$$

$$
d_n=d_0+\frac{1}{2}
(\mathbf z^T\mathbf z+\mathbf b_0^T\mathbf V_0^{-1}\mathbf b_0
-\mathbf b_n^T\mathbf V_n^{-1}\mathbf b_n)
$$

Posterior da hipótese:

$$
P(h\mid\mathbf z)\propto p(\mathbf z\mid h)P(h)
$$

Preditiva de uma hipótese:

$$
z_*\mid\mathbf z,h\sim t_{2a_n}
(\mathbf x_*^T\mathbf b_n,q_h^2)
$$

Mistura final:

$$
p(y_*\mid\mathbf y)=\sum_h
p(y_*\mid\mathbf y,h)P(h\mid\mathbf y)
$$

Decomposição da variância:

$$
V_{total}=V_{dentro}+V_{entre}
$$

## 30. Encadeamento final

$$
\mathbf y\rightarrow\mathbf z\rightarrow
\{\mathbf X_h\}\rightarrow
\{\mathrm{posterior}_h,\mathrm{evidencia}_h\}\rightarrow
\{P(h\mid\mathbf z)\}\rightarrow
\mathrm{mistura\ preditiva}\rightarrow
\mathrm{revisao\ se\ material}
$$

O resultado final preserva três formas de incerteza: ruído de uma nova
observação, incerteza sobre os coeficientes e incerteza sobre qual janela e qual
modelo representam o presente. Essa separação é o principal ganho conceitual do
procedimento.
