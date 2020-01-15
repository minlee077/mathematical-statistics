# 다차원 확률변수의 확률분포
## 이산형 이차원 확률 변수
### 정의
> 두 확률변수 X, Y의 순서쌍인 (X,Y)가 가질 수 있는 순서쌍들의 집합이
$$\{(X_j,Y_k)|j=1,2,...,\space k=1,2,...\}$$
일 때, 각 순서쌍에 그 순서쌍을 가질 확률을 대응시키는 함수
$$f(x_j,y_k)=P(X=x_j,Y=y_k)$$
로 정의된 함수 f를 이산형 bivariate random vector $(X,Y)$의 확률 밀도함수라고 한다.

### 이산형 pdf의 정의
> (a) $f(x,y)\ge0 \space\forall\space x,y:-\infty<x<\infty, \space -\infty<y<\infty \newline f(x,y)=0 \space \forall(x,y)\cancel{=}(x_j,y_k) \space (j=1,2,..., \space k=1,2,...)$
(b)$\displaystyle\sum_{x}\displaystyle\sum_{y}f(x,y)=\displaystyle\sum_{x=1}^{\infty}\displaystyle\sum_{y=1}^\infty f(x_j,y_k)=1$
(c)$ \displaystyle\sum_{x:a\le x\le b}\displaystyle\sum_{y:c\le y\le d} f(x,y)=P(a\le X\le b, c\le Y\le d)$

## 연속형 이차원 확률 변수
### 정의
> 두 개의 확률변수 X, Y가 모두 실수 구간의 값을 가질 수 있고 그에 관한 확률이 적분으로 주어질 때, 두확률변수의 순서쌍인 (X, Y)를 연속형의 bivariate random vector라고 하며 (X, Y)에 관한 확률을 정해주는 함수, 즉 $$\int_{x:a\le x\le b}\int_{y:c\le y\le d}f(x,y)dydx =P(a\le X\le b, c\le Y\le d)\space (a<b,c<d)$$인 함수 $f$를 확률밀도함수라고 한다.

### 연속형 pdf의 정의
>(a) $f(x,y)\ge0 \space\forall\space x,y:-\infty<x<\infty, \space -\infty<y<\infty$
(b) $\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty} f(x,y)dydx=1$
(c) $\int_{a}^{b}\int_{c}^{d}f(x,y)dydx = P(a\le X\le b,\space c\le Y\le d)\space (a<b,\space c<d) $

## Marginal PD , Joint PD
두 확률변수 X, Y가 주어지면, 이들의 순서쌍(X, Y)의 분포를 Joint PD라고 하고,
 X의 분포와 Y의 분포는 각 확률변수의 Marginal PD라고 한다.

> Joint PD로 부터 Marginal PD를 얻는 방법은 다음과 같다.
$\begin{aligned}  
P(a\le X\le b)&=P(a\le X\le b, -\infty<Y<+\infty)\\
&=\begin{cases}
\sum_{x:a\le x\le b}\sum_{y}f(x,y) \space (if,(X,Y) \space are\space  discrete )\\
\\\int_{a}^{b}\int_{-\infty}^{+\infty}f(x,y)dydx \space(if, (X,Y) \space are\space continuous)
\end{cases}
 \end{aligned} $
 따라서,
$$f_1(x)=\begin{cases} \sum_{y}f(x,y)\\\\ \int_{-\infty}^{+\infty}f(x,y)dy\end{cases}$$
로 정의된 함수 $f_1$에 대하여
$$P(a\le X \le b)=
\begin{cases}
\sum_{x:a\le x\le b}f_1(x) \space (if,(X,Y) \space are\space  discrete )
\\\\
\int_{a}^{b}f_1(x)dx \space(if, (X,Y) \space are\space continuous)
\end{cases}$$
y에 대한 함수도 같은 방식으로 $f_2(y)$를 통해 정의할 수 있다.

## X, Y의 Joint CDF
>$F(x,y)=P(X\le x, Y\le y)$, $-\infty<x<+\infty$, $-\infty<y<+\infty$

>(이산형)
X, Y가 가질 수있는 값들이 각각 $x_1<x_2<...<x_m<...$, $y_1<y_2<...<y_n<...$일 때,
$$F(x_m,y_n)=P(X\le x_m,Y\le y_n)=\sum_{j=1}^{m}\sum_{k=1}^{n}f(x_j,y_k)$$
$$\begin{aligned}
f(x_j,y_k)&=P(X=x_j, Y=j_k)\\
&=\{F(x_j,y_k)-F(x_{j-1},y_k)\}-\{F(x_j,y_{k-1})-F(X_{j-1},y_{k-1})\}
\end{aligned}$$

>(연속형)
$$F(x,y)=P(X\le x, Y\le y)=\int_{-\infty}^x \int_{-\infty}^yf(t,u)dudt$$
f가 연속인 점 (x,y)에서 $f(x,y)=\frac{\partial ^2}{\partial x \partial y}F(x,y)$

> Marginal CDF도 Joint CDF를 통해 얻을 수 있다.
$F_1(x)=\lim\limits_{y->\infty+}F(x,y)$
$F_2(y)=\lim\limits_{x->\infty+}F(x,y)$

#### Joint CDF의 성질
>오른쪽 연속성
$\lim\limits_{h->0+}F(x+h,y)=F(x,y)$
$\lim\limits_{k->0+}F(x,y+k)=F(x,y)$


# 결합확률분포의 특성치

## Joint PD에 대한 기댓값
>$$E[g(X,Y)]=\begin{cases}
\sum_{x}\sum_{y}g(x,y)f(x,y)
\\\\
\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}g(x,y)f(x,y)dydx
\end{cases}$$
가 실수로 정의되면 그 값을 $g(X,Y)$의 기댓값이라 하고, 기호로는 $E[g(X,Y)]$로 나타낸다.

## 기댓값의 성질
>(1) $$E[c_1g_1(X,Y)+c_2g_2(X,Y)] = c_1E[g_1(X,Y)]+c_2E[g_2(X,Y)]$$
(2) $$g_1(X,Y)\le g_2(X,Y)$$이면,  $$E[g_1(X,Y)] \le E[g_2(X,Y)]$$

> 기댓값의 정의를 통해 하나의 확률변수에만 의존하는 함수의 기댓값은 Marginal PD에 대한 기댓값임을 알 수 있다.
$$E[g(X)]=\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}g(x)f(x,y)=\int_{-\infty}^{+\infty}g(x)\{\int_{-\infty}^{+\infty}f(x,y)dy\}dx$$
이때, X의 Marginal PDF는 $f_1(x)=\int_{-\infty}^{+\infty}f(x,y)dy$이므로, g(X)의 기댓값은 Marginal PD에 대한 기댓값으로 주어진다.
$$E[g(X)]=\int_{-\infty}^{+\infty}g(x){\int_{-\infty}^{+\infty}f(x,y)dy}dx=\int_{-\infty}^{+\infty}g(x)f_1(x)dx$$

## Covariance와 Correlation coefficient
>(Covariance)
확률변수 X의 평균과 표준편차를 각각 $\mu_1$과 $\sigma_1$, Y의 평균과 표준편차를 각각 $\mu_2$$\sigma_2$라고 할 때, $(X-\mu_1)(Y-\mu_2)$의 기댓값을 X와 Y의 Covariance라고 하고, $Cov(X,Y)$, $\sigma_{X,Y}$,$\sigma_{1,2}$ 로 나타낸다.
$$Cov(X,Y)=E[(X-\mu1)(X-\mu_2)]$$
>(Correlation coefficient) $\rho$
두 확률변수 X, Y의 표준편차 $\sigma_1$, $\sigma_2$가 0이 아닐 때, X와 Y의 Covariance를 두 표준편차의 곱으로 나눈 수를 X와 Y의 Correlation coefficient라고 한다.
$$Corr(X,Y)=\frac{Cov(X,Y)}{\sigma_1 \sigma_2}$$
>> 의미는 다음과 같다.
Covariance는 두 변수가 각각의 기준인 평균으로부터 변화하는 방향과 변화하는 양에 대한 정보를 주는 특성치이다.
한 변수가 그 평균보다 커질 때 다른 변수도 평균보다 커지는 확률이 많다면 $(X-\mu_1)(Y-\mu_2)$의 값이 양수로서 큰 값이 될 것을 기대할 수 있다.
즉, 공분산의 값이 큰 양수일 것이고, 서로가 반대 방향으로 변화하는 확률이 클 경우에는 공분산의 값이 음수로 주어질 것이다.
이와 달리 각각의 변화가 서로에게 영향을 주지않는다면, 0에 가까운 값이 될 것이다.
>> Correlation coefficient는 Covariance가 두 변수의 변화 관계의 방향과 크기를 표현해주지만, 수치의 크기가 측정단위에 의존하는 문제가 있어 측정 단위에 기인한 규모의 차이인지 변화량의 크기에 기인한 것인지를 구분하기 어려운 문제가 있다.
따라서, 각 변수의 측정 단위로 인한 규모 변화를 방지하고자, 표준편차로 나누어 줌으로써 표준화된 Covariance를 의미한다.

## Covariance의 성질과 계산 공식

>(1) $Cov(X,Y) = Cov(Y,X), Cov(X,X)=Var(X)$   
(2) $Cov(aX+b,cY+d)=acCov(X,Y), (a,b,c,d \space are\space  constants)$
(3) $Cov(X,Y)=E(XY)-E(X)E(Y)$

>(2)에 대한 증명
$$V=aX+b, W=cY+d$$라고 하면
$$V-E(V)=a(X-\mu_1),W-E(W)=c(Y-\mu_2)$$
$$\begin{aligned}
Cov(X,Y)&=E[\{V-E(V)\}\{W-E(W)\}]\\
&=E[a(X-\mu_1)c(Y-\mu_2)]\\
&=acE[(X-\mu_1)(X-\mu_2)]
\end{aligned}$$

## Correlation coefficient($\rho$)의 성질
>(1)$$Var(\frac{Y-\mu_2}{\sigma_2}-\rho\frac{X-\mu_1}{\sigma_1})=1-\rho^2$$
(2) $-1\le \rho \le 1$
(3) $
\begin{cases}
\rho = 1 \lrArr P(\frac{Y-\mu_2}{\sigma_2}=\frac{X-\mu_1}{\sigma_1})=1\\\\
\rho = -1 \lrArr P(\frac{Y-\mu_2}{\sigma_2}=\frac{X-\mu_1}{\sigma_1})=1
\end{cases}
$

>(1)은 $var(Z)=E(Z^2)=1$임을 통해 증명이 되며, (1)을 통해 (2)가 성립됨을 알 수 있고, (3)은 $Var(X)=0$인 경우로써, $P(X=\mu)=1$을 통해 증명이 가능하다.
>>위의 성질을 통해 Correlation coefficient의 절댓값이 커질수록 (X,Y)의 분포는 직선 $\frac{y-\mu_2}{\sigma_2}=\rho\frac{x-\mu_1}{\sigma_1}$주위에 가깝게 분포되어 나타날 것이다. (비례관계)
이러한 뜻에서 상관계수는 두 변수 사이의 직선 관계를 나타내는 특성치로 해석할 수 있다.


## Joint moment
>Joint monet
$E(|X^rY^s|)<+\infty$일 때,
$$E(X^rY^s)=
\begin{cases}
\sum_{x}\sum_{y}x^ry^sf(x,y)
\\\\
\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}x^ry^sf(x,y)dydx
\end{cases}$$
를 (X,Y)의 (r+s)차의 (r,s)번째 joint moment라고 한다.
기호 : $m_{r,s}(X,Y)$또는 $m_r,s$
>>직선 관계의 특성을 나타내는 Covariance, Correlation coefficient처럼 Joint PD에 대한 특성을 나타낸다.

>하나의 PD에서와 같이 Joint moment generating function은 유사한 방식으로 전개된다.
$$e^{t_1X}e^{t_2Y}=\sum_{r=0}^{\infty}\frac{(t_1X)^r}{r!}\sum_{s=0}^{\infty}\frac{(t_2Y)^s}{s!}$$
$$E(e^{t_1X+t_2Y})=\sum_{r=0}^{\infty}\sum_{s=0}^{\infty}\frac{E(X^rY)^s}{r!s!}t_1^rt_2^s$$
Joint MGF는 Joing Moment를 생성하며, Joint PD를 결정하는 성질을 갖는다.

## Joint MGF 정의
>0을 포함하는 열린구간들의 $t_1,t_2$값에 대하여 $E(e^{t_1X+t_2Y})$가 실수일 때, 함수
$$M(t_1,t_2)=E(e^{t_1X+t_2Y}), -h_1<t_1<h_1, -h_2<t_2<h_2 (\exist h_1>0,h_2>0)$$
를 Joint MGF라고 한다.

## Joint MGF 성질
>(1) 이차원 확률변수 (X,Y)의 Joint MGF가 존재하면, 즉
$$M(t_1,t_2)=E(e^{t_1X+t_2Y})<+\infty \newline
-h_1<t_1<h_1, -h_2<t_2<h_2(\exist h_1>0,h_2>0)
$$
이면, (X,Y)의 모든 Joint moment가 존재하고,
$$
E(X^rY^s)=[\frac{\partial^{r+s}}{\partial t_1^r \partial t_2^s}M(t_1,t_2)]_{t_1=0,t_2=0}\newline
E(e^{t_1X+t_2Y})=M(t_1,t_2)=\sum_{r=0}^\infty\sum_{s=0}^\infty\frac{E(X^rY^s)}{r!s!}t_1^rt_2^s\newline
-h_1<t_1<h_1, -h_2<t_2<h_2(\exist h_1>0,h_2>0)
$$
(2) (분포 결정성)
두 이차원 확률변수 (X_1,x_2)와 (Y_1,Y_2)의 Joint MGF가 0을 포함하는 열린구간들에서 일치 즉, $$M_{X_1,X_2}(t_1,t_2)=M_{Y_1,Y_2}(t_1,t_2)\forall t_i: -h_i<t_i<h_i (\exist h_i>0)(i=1,2)$$
이면,$(X_1,X_2)$와 $(Y_1,Y_2)$의 PD가 일치한다. 즉, Joint PDF와 Joint CDF가 일치한다.

## Joint CGF 정의
> Joint MGF가 $M(t_1,t_2)$가 존재하면,
$$C(t_1,t_2)=logM(t_1,t_2)=logE(e^{t_1X+t_2Y}), -h_i<t_i<h_i(\exist h_i>0)(i=1,2)$$
를 Joint CGF(cumuland generating function)라고 한다.
$$
C(t_1,t_2)=\sum_{r=0,r+s\ge 1}^{\infty}\sum_{s=0}^{\infty}\frac{C^{(r,s)}(0,0)}{r!s!}t_1^rt_2^s\newline
C^{(r,s)}(0,0)=[\frac{\partial^{r+s}}{\partial t_1^r\partial t_2^s}]_{t_1=0,t_2=0}
$$
$C^{(r,s)}(0,0)$을 (X,Y)의 (r+s)차의 (r,s)번째 joint cumlant,
기호: $c_{r,s}(X,Y)$또는 $c_{r,s}$

> joint cumlant와 joint moment의 관계를 통해
$$
\begin{aligned}
c_{1,0}&=m_{1,0}=E(X) \\
c_{0,1}&=m_{0,1}=E(Y) \\
c_{2,0}&=m_{2,0}-(m_{1,0})^2=Var(X) \\
c_{1,1}&=m_{1,1}-m_{1,0}m_{0,1}=Cov(X,Y)\\
c_{0,2}&=m_{0,2}-(m_{0,1})^2=E(Y^2)-[E(Y)]^2=Var(Y)\\
\end{aligned}
$$
## Marginal MGF, CGF

$$
\begin{aligned}
\\
M_X(s)&=M_{X,Y}(s,0), C_X(s)=C_{X,Y}(s,0)\\
M_Y(t)&=M_{X,Y}(0,t), C_Y(t)=C_{X,Y}(0,r)
\\\\
\end{aligned}
$$

# 조건부분포와  조건부기댓값

## Conditional PDF의 정의
> 확률변수 X,Y가 모두 이산형일 때, X=x인 조건에서 Y에 관한 CD가
$$P(Y=y|X=x)=\frac{P(X=x,Y=x)}{P(X=x)}$$
로 정의되기 위해서는 $P(X=x)>0$이어야 한다. 즉 이산형의 경우에는 변수가 가질 수 있는 값이 주어진 조건에서의 CD는 언제나 정의된다.

>이차원 확률변수 (X,Y)가 이산형으로서 (X,Y)의 Joint probabiltiy가 $f_{1,2}(x,y)$이고, X의 Marginal pmf가 $f_1(x)$일 때, $X=x$인 condition에서 Y의 가능한 값에 conditional probabiltiy를 대응시키는 함수
$$f_{1,2}=\frac{f_{1,2}(x,y)}{f_1(x)}=\frac{P(X=x,Y=y)}{P(X=x)}=P(Y=y|X=x)$$
를 $X=x$인 조건에서 Y의 conditional pdf라고 한다.

>이산형인 경우의 conditional pdf
$$f_{2|1}(y|x)=\frac{f_{1,2}(x,y)}{f_1(x)}=P(Y=y|X=x) \space \space (x:f_1(x)>0)$$
(1) $f_{2|1}(y|x)\ge0 \space \forall y: -\infty < y<+\infty $
 $f_{2|1}(y|x)=0 \space \forall y:y\not= y_k(k=1,2,...)$
(2) $\displaystyle\sum_y f_{2|1}(y|x)=1$
>>$(X=x)$ event로 구성된 sample space에서 y의 분포로 해석할 수 있다.

>연속형인 경우의 conditional pdf
$$f_{2|1}(y|x)=\frac{f_{1,2}(x,y)}{f_1(x)} \space \space (x:f_1(x)>0)$$
$$P(c\le Y\le d|X=x)=\lim_{h->0+} P(c\le Y\le d|x\le X\le x+h)=\int_c^df_{2|1}(y|x)dy$$
(1) $f_{2|1}(y|x)\ge0 \space \forall y: -\infty < y<+\infty $
(2) $\int_{-\infty}{+\infty}f_{2|1}(y|x)dy=1$
(3) $P(c\le Y \le d|X=x)=\int_c^df_{2|1}(y|x)dy$
>> $P(c\le Y \le d, X=x)=0, P(X=x)=0$이므로, 양수 h에 대한 범위 극한을 통해 정의한다.

## conditional pdf의 성질
>$$P(a\le X\le b, c\le Y\le d)=
\begin{cases}
\displaystyle\sum_{a\le x \le b}P(c\le Y\le d|X=x)f_1(x) \space\space(if\space discrete)
\\\\
\int_{a}^{b}P(c\le Y\le d|X=x)f_1(x) \space\space (if \space continuous)
\end{cases}$$
>>실직선 위의 구간에 분포되는 것을 $X=x$인 조건에서 Y의 conditional pdf는 간략히 조건부분포라고 한다.

## Conditional Mean
>$X=x$인 조건에서 Y의 CDF가 $f_{2|1}(y|x)$일 때,
$$\mu_{2|1}(x)=E(Y|X=x)=
\begin{cases}
\displaystyle\sum_yyf_{2|1}(y|x) \space\space (if \space discrete)\\
\displaystyle\int_{-\infty}^{+\infty}yf_{2|1}(y|x)dy \space\space (if \space continuous)
\end{cases}$$
우항이 실수로 정의될 때, 좌항을  $X=x$일때, Y의 conditional mean이라 한다.

## Conditional Expectation
>$$E(g(X,Y|X=x)=\begin{cases}\sum_yg(x,y)f_{2|1}(y|x)\space \space (if \space discrete) \\ \int_{-\infty}^{+\infty}g(x,y)f_{2|1}(y|x)dy \space\space (if \space continuous)\end{cases}$$
conditional expected value of $Y$, given $X=x$

## Condtional Variance
>$$
\begin{aligned}
Var(Y|X=x)&=E[(Y-\mu_{2|1}(x))^2|X=x)]\\
&=\begin{cases} \displaystyle\sum_y(y-\mu_{2|1}(x))^2f_{2|1}(y|x) \\
\int_{-\infty}^{+\infty}(y-\mu_{2|1}(x))^2f_{2|1}(y|x)dy
 \end{cases}
\end{aligned}
$$

## Condtional Expectation의 성질 및 유도식
>(1) $E[c_1g_1(Y)+c_2g_2(Y)|X=x]=c_1E[g_1(Y)|X=x]+c_2E[g_2(Y)|X=x]$
(2) $E[c(X)g(Y)|X=x]=c(X)E[g(Y)|X=x]$
(3) $g_1(Y)\le g_2(Y)이면, E[g_1(Y)|X=x]\le E[g_2(Y)|X=x]$
$$Var(Y|X=x)=E(Y^2|X=x)-\{E(Y|X=x)\}^2$$
>conditional mean $\mu_{2|1}(x)=E(Y|X=x)=E(Y|X)$는 확률변수 X의 값 x에 따라 그 값이 정해지는 함수이다. 따라서, $\mu_{2|1}(x)$는 X의 분포에 따라 $\mu_{2|1}(x)$ 값들을 가지는 새로운 확률변수 이다. 이러한 확률변수를 X가 주어진 조건에서 Y의 Condtional mean이라 한다.
마찬가지로 conditional variance는 $\sigma_{2|1}^2(x)=Var(Y|X=x)=Var(Y|X)$로, X의 분포에 따라 결정되는 확률변수이다.

## 확률변수로서의 Conditional Expectation $E[g(Y)|X]$
>X=x인 조건에서 g(Y)의 conditional Expectation을
$$h(x)=E[g(Y)|X=x]=E[g(Y)|X]=h(X)$$
라고 할 때, X의 분포에 따라 $h(x)$ 값들을 가지는 확률변수를 X가 주어진 조건에서 g(Y)의 Condtional Expectation이라 한다.

## Conditional Expectation의 성질
> (1)$E[E(Y|X)]=E(Y), \space E(Y)=E[E[Y|X]]$
(2)$Cov(Y-E(Y|X),v(X))=0, \space\space \forall v(X)$

>$Proof$
$$\begin{aligned}
E[E(Y|X)]&=\int_{-\infty}^{+\infty}E(Y|X)f_1(x)dx \space\space \because E(Y|X)=g(X)\\
&=\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}yf_{2|1}(y|x)dyf_1(x)dx\\
&=\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}yf(x,y)dydx\\
&=E(Y) \space\space \because y=g(y), \space vice \space versa
\\\\
\end{aligned}
$$
$$
let, Z=Y-E(Y|X)\newline
E(Z|X)=E(Y|X)-E\{E(Y|X)|X\}=E(Y|X) - E(Y|X)=0 \newline \because E(Y|X)=g(X), \space E[g(X)|X] = E[g(X)]=E[Y|X]\newline
E(Z)=E[E(Z|X)]=E(0)=0
$$
$$\begin{aligned}
Cov(Z,v(X))&=E(Zv(X))-E(Z)E(v(X)) \\
&=E\{E(Zv(X)|X)\} \space \because E(Z)=0, \space\space E(f(Z,X)=Y)=E\{E(f(Z,X)=Y|X)\}...(1)\\
&=E\{v(X)E(Z|X)\}\\
&=0=Cov(Y-E(Y|X),v(X))
\end{aligned}$$

## 회귀함수
> Condtional Expectation $E(Y|X)$는 두 확률변수 Y와 X 사이의 관계를 설명하는 데에 특별한 의미를 가진다. 특히나, $E(Y|X)$는 Y를 예측하는 가장 좋은 X의 함수를 의미하는 회귀함수라고 한다.
$least \space square \space predictor$
확률변수 X의 함수 $u(X)$로서 $E[(Y-u(X))^2]$를 최소로 하는 함수는 E(Y|X)이다. 즉,
$$E[(Y-E(Y|X))^2]\le E[(Y-u(X))^2], \space \forall u(X) $$
이고, mean squared prediction error인 $E[(Y-u(X))^2]$의 최소값은
$$E[(Y-E(Y|X))^2]=E\{E[(Y-E(Y|X))^2|X]\} = E[Var(Y|X)] \newline
\because E[(Y-\mu_{2|1}(x))^2|X=x]=E[Y|X=x]$$

## 분산의 분해
>$$Var(Y)=E[Var(Y|X)]+Var[E(Y|X)]$$
$\because Y-\mu = (Y-E(Y|X))\oplus(E(Y|X)-\mu) $
$ E[(Y-\mu)^2]=E[(Y-E(Y|X))^2]-2Cov(Y-E(Y|X),E(Y|X))+E[(E(Y|X)-\mu)^2]=E[Var(X|Y)]+Var(E(Y|X))$
$Cov(Y-E(Y|X,E(Y|X)))=0$

# 확률변수의 독립성
>정의
확률변수 X가 어떠한 범위의 값을 갖든 확률변수 Y에 관한 사건의 가능성에 아무런 영향을 주지 않는 경우. 즉 a,b,c,d값에 상관없이
$$P(C\le Y \;e d|a \le X\le b)=P(c \le Y \le d)$$
$$P(a\le X\le b,c\le Y \le d)= P(a\le X\le b)P(c\le Y \le d) \forall a,b,c,d$$
가 성립하는 경우로서, X에 관한 어떠한 사건도 Y에 관한 사건과 서로 독립이다. 두 확률변수 x와 Y가 $mutally\space independent$라고 한다.

## Probabiltiy Function에서의 독립성
>(1)Joint CDF $cdf_{1,2}(x,y)=cdf_1(x)cdf_2(y)$
(2)Joint PDF $pdf_{1,2}(x,y)=pdf_1(x)pdf_2(y)$
(3)Joint mgf $mgf_{1,2}(x,y)=mgf_1(x)mgf_2(y)$
(4)Probabiltiy Measure $P(X\in A,Y\in B)=P(X\in A)P(Y\in B),\space \forall A,B$*
*Probabiltiy Measure$X\in A$은 $\{s:X(s)\in A\}$, 즉 역사상을 의미한다.

>Joint PDF를 통해 독립성을 밝힐때, 반드시 Marginal PDF를 구할필요는 없다.
즉,
$$pdf_{1,2}(x,y)=g_1(x)g_2(y) \forall x,y \exist g_1, g_2$$
를 만족하는 $g_1(x),g_2(y)$가 존재하는 것을 증명하기만 하면 된다.
$$g_1(x)g_2(y)=\frac{g_1(x)}{\int_{-\infty}^{+\infty}g_1(x)dx}\frac{g_2(y)}{\int_{-\infty}^{+\infty}g_2(y)dy}=pdf_1(x)pdf_2(y)$$
이므로. ($\because 1=\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}pdf_{1,2}(x,y)dydx=\int_{-\infty}^{+\infty}\int_{-\infty}^{+\infty}g_1(x)g_2(y)dydx=\int_{-\infty}^{+\infty}g_1(x)dx\int_{-\infty}^{+\infty}g_2(y)dy$)

>$(e.g)$
$f_{1,2}(x,y)=2e^{-x-2y}I_{(x\ge 0,y\ge 0)}=2e^{-x}I_{(x\ge 0)}e^{-2y}I_{(y \ge 0)}=g_1(x)g_2(y)$
$\therefore X \space and \space B \space are \space mutually \space independent $

> 같은 방식으로 위의 (1)과 (3)의 Marginal cdf, Marginal mgf를 구하지 않아도 독립성을 판단할 수 있다.

## 독립인 확률변수들의 함수
>확률변수 X, Y 가 서로 독립이면 각각의 함수인 $g_1(X), g_2(Y)$도 서로 독립이다.
>>$Proof$
$g$의 inverse를 $g^{-1}(A)=\{x:g(x)\in A\}$로 나타내면
$$
(g_1(X)\in A, g_2(Y)\in B)=(X\in g_1^{-1}(A),Y\in g_2^{-1}(B))\newline
(g_1(X)\in X)=(X\in g_1^{-1}(A)),\space (g_2(Y)\in B)= (Y\in g_2^{-1}(B))
$$
이때, X, Y가 독립이면,
$$P(X\in g_1^{-1}(A),Y\in g_2^{-1}(B))=P(X\in g_1^{-1}(A))P(Y\in g_2^{-1}(B))$$
이므로,
$$\therefore P(g_1(X)\in A, g_2(Y) \in B) = P(g_1(X))P(g_2(Y))$$

## Expectation Variance의 독립성
>X, Y가 서로 독립이면,
$$E[g_1(X)g_2(Y)]=E[g_1(X)]E[g_2(Y)]$$
$$Cov(X,Y)=0 \space \space \space *$$
*역은 성립하지 않는다. ($E[XY]=E[X]=E[Y]=0$인 경우가 반례)

>$$Var(X+Y)=Var(X)+Var(Y)+2Cov(X,Y)\newline
if, \space X,\space Y \space are \space independent\newline
Var(X+Y)=Var(X)+Var(Y)
$$
