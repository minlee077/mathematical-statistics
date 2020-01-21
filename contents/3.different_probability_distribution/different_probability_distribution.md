# 여러 가지 확률분포

---

##hypergeometric distribution
> 초기하분포

###정의
모집단에서 $n$개를 단순랜덤추출하여 얻은 표본 중 1(참)의 개수를 $X$라고 할 때, $X$의 확률 밀도함수는 아래의 식과 같고,
이러한 $X$의 분포를 hypergeometric distribution라고 한다. 기호는 $X~H(n;N,D)$이다.

<center>

$f(x) = \Rho (X=x) = {D \choose x} {N-D \choose n-x} / {N \choose n}, \space 0 \le x \le D, \space 0 \le n-x \le N-D$
</center>

### hypergeometric distribution 평균
>기호 $H(n;N,D)$

<center>

$E(X) = np$
</center>

####유도
<center>

$E(X) = \sum_x {D \choose x} {N-D \choose n-x} / {N \choose n} \hspace{7em}$
$= D\sum_x {D-1 \choose x-1} {N-1-(D-1) \choose n-1 - (x-1)} / {N \choose n}$
$= D{N-1 \choose n-1} /{N \choose n} \hspace{6.97em}$
$= nD/N \hspace{9.5em}$
</center>

- 위 유도과정중 3번째 등식은 아래의 다항식에서 $t^{n-1}$의 계수를 비교하여 밝힐 수 있다.
<center>

$(1+t)^{D-1}(1+t)^{N-1-(D-1)} = (1+t)^{N - 1}$
</center>

### hypergeometric distribution 분산
<center>

$Var(X) = {N-n \over N-1}np(1-p)$
</center>
#### 유도
<center>

$E[X(X-1)] = D(D-1){N-2 \choose n-2}/{N \choose n} = n(n-1)D(D-1)/N(N-1)$
$Var(X) = E(X^2) - (E(X))^2 \hspace{10.2em}$
$= E[X(X-1)] + E(X) - (E(X))^2$
$= {N-n \over N-1}n{D \over N}(1-{D \over N}) \hspace{7.57em}$
</center>

### $N$이 $n$보다 충분히 큰 hypergeometric distribution
> 전체크기가 표본크기보다 충분히 큰 초기하분포

모집단의 크기 $N$이 표본 크기 $n$에 비해 충분히 큰경우 다름과 같이 초기하분포 확률에 대한 근사계산이 가능하다.

<center>

${D \choose x}{N-D \choose n-x}/{N \choose n} = {D! \over x!(D-x)!} {(N-D)! \over (n-x)!}{n!(N-n)! \over N!} \hspace{14.23em}$
$= {n \choose x}{D(D-1) \cdots (D-x+1) \over N(N-1)\cdots (N-x+1)} {(N-D)\cdots(N-D-n+x+1) \over (N-x) \cdots (N-n+1)}$
$\fallingdotseq {n \choose x} ({D \over N})^x (1 - {D \over N})^{n-x} \hspace{8.4em}$
</center>

위와 같은 계산은 복원추출에 의한 확률과 같다.
여기에서 모집단의 크기 N이 커짐에 따라 비복원추출의 효과가 없어지고 마치 복원추출하는 것과 같게 된다는 것을 알 수 있다.

---

## binomial distribution와 multinomial distribution
### binomial distribution의 일반적 정의
>이항 분포, 기호: __$X \sim B(n,p)$__

모집단을 구성하는 각 개체의 특성이 '0' 또는 '1'로 분류되어 있고 '1'의 비율이  $p$일 때, 모집단에서 한 개씩 동일 확률의 복원추출에 의해 뽑은 $n$개 중에서 '1'의 개수를 $X$라고 하면 $X$의 확률밀도함수는 다음과 같다. 이때의 분포를 __binomial distribution__ 라고 한다.

### binomial distribution의 평균

$X~B(n,p) 이면$

<center>

$E(X)=np$
</center>

#### 유도
>이항정리 사용

<center>

$E(X)=\sum_{x=0}^n x {n \choose x} p^x (1-p)^{n-x} = \sum_{x=1}^n n {n-1 \choose x-1} p^{(x-1)+1}(1-p)^{n-1-(x-1)}$
$= np\sum_{k=0}^{n-1}{n-1 \choose k} p^k (1-p)^{n-1-k} = np(p+(1+p))^{n-1}=np $
</center>

### binomial distribution의 분산
$X~B(n,p)$이면

<center>

$Var(X)=np(1-p)$
</center>


#### 유도
먼저 $E[X(X-1)]$에 대한 값을 다음과 같이 구한다.

<center>

$E[X(X-1)] = n(n-1)p^2$
</center>

그후 위를 이용하여 분산을 구하면 다음과 같다.


<center>

$Var(X) = E(X^2) - (E(X))^2 = E[X(X-1)] + E(X) - (E(X))^2$
$= np(1-p) \hspace{15.65em}$
</center>

### Bernoulli distribution
> 베르누이 분포, 기호: __$Z_i \sim Bernoulli(p)$__

각각의 복원추출이 독립인 추출결과를 $Z_1, Z_2, \cdots , Z_n, \cdots$이라고 하자.
각각의 분포가 아래와 같이 같을 때, 모집단의 분포($Z_i$의 분포)를 Bernoulli distribution라고 한다.


<center>

$P(Z_i = 1) = p, P(Z_i = 0) = 1 - p \enspace (i=1,2, \cdots, n)$
</center>


### Bernoulli trial
> 베르누이시행

두 가지로 분류된 모집단을 관측하는 것을 이른다.
- 서로 독립인 베르누이시행이란 서로 독립이고 베르누이분포를 따르는 확률변수들을 관측하는 것을 뜻한다.
- '1'과 '0'을 각각 '성공'과 '실패'로 부르고, 이항분포를 서로 독립인 Bernoulli trial에서 나오는 성공횟수의 분포라고도 한다.

#### binomial distribution의 대의적 정의
<center>

$X \sim B(n,p) \Leftrightarrow X \stackrel{d}{\equiv} Z_1+\cdots +Z_n, Z_i \stackrel{iid}{\sim} Bernoulli(p)(i=1, \cdots, n)$
</center>

- Bernoulli trial적용한 이항분포의 정의이다.

### binomial distribution의 성질
1. $X \sim B(n,p)$이면 그 moment generating function은 다음과 같다.

<center>

$mgf_X(t) = (pe^t + q)^n, - \infty < t < + \infty$
</center>

2. $X_1 \sim B(n,p), X_2 \sim B(n_x,p)$이고 $X_1, X_2$가 서로 독립이면 다음관계가 성립한다.
<center>

$X_1 + X_2 \sim B(n_1 + n_2, p)$
</center>

### multinomial distribution
> 다항분포, 기호: __$X=(X_1,X_2, \cdots, X_k)^t \sim Multi(n, (p_1,p_2, \cdots, p_k)^t)$__

모집단을 구성하는 각 개체의 특성이 3개 이상의 유형으로 분리되는 경우, 각 유형의 비율이 $p_1, p_2, \cdots, p_k$일 때를 생각해보자.
이때 동일 확률로 복원추출한 n개의 표본의 유형의 개수를 $X_1, X_2, \cdots ,X_k$라고 하면 $X=(X_1, X_2, \cdots, X_k)^t$의 결함확률 밀도는 다음과 같고, 이 분포를 mutinomial distribution이라고 한다.
<center>

$f(x_1, x_2, \cdots, x_k) = {n \choose x_1x_2\cdots x_k}p_1^{x_1}p_2^{x2}\cdots p_k^{x_k}$
$x_i=0,\cdots , n(i=1,2,\cdots, k) x_1 + x_2 + \cdots + x_k = n$
</center>

### multinomial trial
>다항시행

여러 가지 유형으로 분류되는 모집단에서 한 개씩 추출하여 관측하는 것.
- 복원 추출하는 것은 서로 독립인 multinomial trial을 뜻한다.
<center>

$X=(X_1,X_2, \cdots, X_k)^t \sim Multi(n,(p_1,p_2, \cdots, p_k)^t)$
$\Leftrightarrow X  \stackrel{d}{\equiv} Z_1+\cdots +Z_n, Z_i = (Z_{i1}, Z_{i2}, \cdots, Z_{ik})^t \stackrel{iid}{\sim}Multi(1,(p_1, p_2, \cdots, p_k)^t)(i=1, \cdots, n)$
</center>


### multinomial distribution의 성질

1. $X=(X_1,X_2,\cdots, X_k)^t \sim Multi(n,(p_1, p_2, \cdots, p_k)^t)$이면
<center>

$E(X_i) = np_l(l = 1, \cdots, k)$
$Var(X_l)=np_l(1-p_l), Cov(X_l) = -np_lp_m(l {=}\mathllap{/\,}m, l,m=1,\cdots, k)$
</center>

2. $X=(X_1,X_2,\cdots, X_k)^t \sim Multi(n,(p_1, p_2, \cdots, p_k)^t)$이면 moment generating function은

<center>

$mgf_X(t) = (p_1e^{t_1}+\cdots + p_ke^{t_k})^n, \> -\infty < t_1 < +\infty \> (l=1,\cdots, k)$
</center>

## geometric distribution
### geometric distribution의 일반적 정의
> 기하분포, 기호: __$W_1 \sim Geo(p)$__

서로 독립이고 성공률이 p인 bernoulli trial $X_1, \cdots, X_n, \cdots$을 관측할 때, 첫번째 성공을 관측할 때까지의 시행횟수를 $W_1$라고 하면 이때의 확률은 다음과 같고, 이러한 확률분포를 geometric distribution이라고 한다.
<center>

$P(W_1 = x) = (1-p)^{x-1}p, x=1,2,\cdots$
</center>

### geometric distribution의 성질
1. $W_1 \sim Geo(p)$이면 그 moment generating function은
<center>

$mgf_{W_1}(t) = (1-qe^t)^{-1}e^tp, \enspace t < -\log q \space (q = 1-p) $
</centeR>

2. $W_1 \sim Geo(p)$ 이면
<center>

$E(W_1) = 1/p, Var(w_1) = q/p^2 \space (q=1-p)$
</centeR>

## negative binomial distribution
### negative binomial distribution의 일반적 정의
> 음이항분포, 기호: __$W_r \sim Negbin(r,p)$__

서로 독립이고 성공률이 $p$인 Bernoulli trial $X_1, \cdots, X_n, \cdots$을 과나측할 때 $r$번째 성공까지의 시행횟수를 $W_r$이라고 하면 다음이 성립하고, 이 분포를 negative binomial distribution이라고 한다.

<center>

$P(W_r = x) = {x-1 \choose r-1}p^{r-1}(1-p)^{(x-1)-r-1}p \hspace{9.75em}$
$={x-1\choose r-1}p^r(1-p)^x-r, \enspace x=r, r+1, \cdots$
</center>

### negative binomial distribution의 대의적 정의

- 아이디어
<center>

$P(W_1 = x_1, W_2 - W_1 = x_2, \cdots, W_r - W_{r-1} = x_r) \hspace{8em}$
$ P(연속된 (x_i)번의 실패 후 성공, i=1,\cdots, r) \hspace{11.7em}$
$={(1-p)^{x_i - 1}p}{(1-p)^{x_2-1}p} \cdots {(1-p)^{x_r-1}p}, \space x_i = 1,2,\cdots \space (i=1,\cdots, r)$
</center>


- 정의
<center>

$X \sim Negbin(r,p) \Leftrightarrow X \stackrel{d}{\equiv} Z_1 + \cdots +Z_r, Z_i \stackrel{iid}{\sim} Geo(p)(i=1,\cdots, r)$
</center>

### negative binomial distribution의 성질
1. $X \sim Negbin(r,p)$이면 $E(X)=r/p, \space Var(X)= rq/p^2$
2. $X \sim Negbin(r,p)$이면 moment generating function은
<center>

$mgf_X(t)={\{pe^t(1-qe^t)^{-1}\}}^r, t < -\log q \space (q=1-p)$
</center>

3. $X_1 \sim Negbin(r_1,p) X_2 \sim Negbin(r_2,p)$이고 $X_1, X_2$가 서로 독립이면
<center>

$X_1 + X_2 \sim Negbin(r_1 + r_2, p)$
</center>


## Poisson distribution
### Poisson approximation
> 푸아송 근사

binomial distribution $B(n,p)$에서 시행횟수 $n$이 크고 확률 $p$가 작은 경우는 아래 식1과 같은 근사식이 성립하고,
이를 아래식 2로 나타낸 것을 binomial probability의 __Poisson approximation__ 이라고 한다.

<center>

1. ${n \choose x}p^x(1-p)^{n-x} = n(n-1) \cdots (n-x+1)p^x(1-p)^{n-x}/x! \hspace{8.35em}$
$= {n \over n}(1 - {1 \over n}) \cdots (1 - {x-1 \over n})(np)^x(1-{np \over n})^{n-x}/x!$
$= (np)^xe^{-np}/x! \hspace{12.65em}$

2. $\lim_{\substack{n\rarr \infty \\ np_n \rarr \lambda}} {n \choose x}p_n^x(1-p_n)^{n-x} = e^{-\lambda}/x! \space (\lambda > 0, \lambda = np)$
</center>

### Poisson distribution's probability density function
> 푸아송 분포의 확률밀도함수, 기호: __$X \sim Poisson(\lambda)$__

<center>

$f(x)=e^{-lambda} \lambda^x/x!, \space x=0,1,2,\cdots \enspace (\lambda > 0)$
</center>

#### Poisson distribution의 성질
1. $X \sim Poisson(\lambda)$ 이면 그 moment generating function은
<center>

$mgf_X(t)=e^{-\lambda + \lambda e^t}, -\infty < t < + \infty$
</center>

2. $X \sim Poisson(\lambda)$ 이면
<center>

$E(x)=\lambda, Var(X)=\lambda$
</center>

3. $X_1 \sim Poisson(\lambda_1),X_2 \sim Poisson(\lambda_2)$이고 $X_1, X_2$가 서로 독립이면
<center>

$X_1 + X_2 \sim Poisson(\lambda_1 + \lambda_2)$
</center>

### Poisson process
> 푸아송 과정

시각 0에서 $t$까지 특정한 현상이 발생하는 횟수를 $N_t$이라고 할 때,
다음의 조건들이 만족되면 $\{N_t:t \ge 0\}$를 occurrence rate(발생률) $\lambda$인 __Poisson process__ (포아송과정)이라고 한다.

1. Stationarity(정상성)
    현상이 발생하는 횟수의 분포는 시작하는 시각에 관계없다.
    즉, $N_t$의 분포와 $N_{s+t} - N_s$의 분포가 같고, $N_0=0$이다.

2. Independent Increment(독립증분성)
    시각 0부터 $t$까지 현상이 발생하는 횟수와 시각 $t$ 후부터 $1+h(h>0)$까지 발생하는 횟수는 서로 독립이다.
    즉, $N_t$와 $N_{t+h} - N_t$는 서로 독립니다.

3. Proportionality(비례성)
    짧은 시간 동안에 현상이 한 번 발생할 확률은 시간에 비례한다.
    즉, $P=(N_h = 1) = \lambda h + o(h), h \rarr 0$ 이 성립한다.
    위에서 $\lambda$는 양수의 비례상수이며 $o(h)$의 의미는 $\lim_{h \rarr 0}o(h)/h = 0$을 말한다.

4. Rareness(희귀성)
    짧은 시간 동안에 현상이 두 번 이상 발생할 확률은 매우 작다.
    즉, $P(N_h \ge 2) = o(h), h \rarr 0$

#### Poisson process 에서 발생횟수의 분포

occurrence rate이 $\lambda$인 Poisson process $\{N_t : t \ge 0\}$에서 시각 t까지 발생횟수 $N_t$의 분포는 평균이 $\lambda t$인 Poisson distribution이다.

<center>

$N_t \sim Poisson(\lambda t)$
</center>


## exponential distribution
### exponential distribution의 정의
> 지수분포, 기호:__$W_1 \sim Exp(1/\lambda) \space (\lambda > 0)$__

occurrence rate이 $\lambda$인 Poisson process ${N_t:t\ge 0}$에서 첫 번째 현상이 시각 $t$후에 발생한다고 하자.
이때 까지의 시간을 $W_1$라고 하면 다음과 같은 등식이 성립하고
<center>

$(W_1 > t) = (N_t = 0)$
</center>

이를 이용해 다음과 같은 cumulative distribution funciton을 구할 수 있다.
<center>

$P(W_1 > t )= P(N_t = 0) = e^{-\lambda t}, t \ge 0$
$P(W_1 \le t) = \begin{cases}1-e^{\lambda t}, t \ge 0 \\ 0, \hspace{2.3em} t<0\end{cases}$
</center>

따라서 첫 번째 현상이 발생할 때 까지의 시간 $W_1$의 확률밀도함수는 다음과 같고, 이를 __exponential distribution__ 이라고 한다.
<center>

$f(x) = \lambda e^{-lambda x}I_{(x\ge 0)}$
</center>

### exponential distribution의 성질
1. $W_1 \sim Exp(1/\lambda) \space (\lambda > 0)$이면 그 moment generating funciton은
<center>

$mgf_{W_1}(t) = (1-t/\lambda)$
</center>

2. $W_1 \sim Exp(1/\lambda) \space (\lambda > 0)$이면
<center>

$E(W_1)=1/\lambda, \space Var(W_1) = 1.\lambda ^2$
</center>

## gamma distribution
### gamma distribution의 정의
>감마분포

occurrence rate이 $\lambda$인 Poisson process ${N_t : t\ge 0}$ 에서 $r$ 번째 현상이 시각 $t$후에 발생한다고 하자.
이때까지의 시간을 $W_r$라하면 다음 관계가 성립한다.
<center>

$(W_r > t) = (N_r \le r-1)$
$P(W_r \le t ) = 1-P(W_r > t) = 1-P(N_t \le r-1)=1-\sum_{k=0}^{r-1} e^{-\lambda t}(\lambda t)^k / k!$
$t \ge 0$
</center>

이로부터 양변을 미분하여 $W_r$의 probability density function을 구하면 다음과 같고, 이를 gamma distribution이라고 부른다.
<center>

$pdf_{W_r}(t) = {d\over dt}cdf_{W_r}(t)$
$\hspace{17.85em}= -\sum_{k=0}^{r-1}\{(-\lambda)e^{\lambda t}(\lambda t )^k / k! + e^{-\lambda t } k \lambda(\lambda t)^{k-1}\}$
$\hspace{19.3em}=\lambda e^{-\lambda t}\{\sum_{k=0}^{r-1}(\lambda t )^k/k! - \sum_{k=1}^{r-1}(\lambda t )^{k-1}/(k-1)!\}$
$\hspace{10.49em}=\lambda^rt^{r-1}e^{-\lambda t}/(r-1)!, \space t > 0$
</center>

###shape parameter
>형상모수, $\Gamma (\alpha)$

위의 감마분포
<center>

$\lambda^rt^{r-1}e^{-\lambda t}/(r-1)!, \space t > 0$
</center>

에서 $r$의 값에 따라서 분포의 형태가 바뀐다. 따라서 $r$을 shape parameter라고 부른다.
일반적으로는 양수일 수 있으며, 이러한 경우에 흔히 $\alpha$로 나타내기도 한다.
이와 관련해
<center>

$\Gamma(\alpha) = \int_0^{+\infty} x^{\alpha - 1}e^{-x}dx, \space \alpha > 0$
</center>

처럼 정의되는 감마함수를 이용해 $Gamma(\alpha, (\beta))$분포의 pdf를 다음과 같이 나타낸다.
<center>

$f(x)= {1 \over \Gamma(\alpha)\beta^\alpha}x^{\alpha -1}e^{-x/\beta}I_{(x>0)} \space (\alpha >0, \beta > 0)$
</center>

- 위의 $\beta$는 scale parameter(척도모수)라고 하며 occurrence rate의 역수인 $1/\lambda$이다.

### gamma distribution의 성질
1. $X \sim Gamma(\alpha, \beta)$이면
<center>

$E(X) = \alpha \beta, Var(X)=\alpha \beta^2$
</center>

2. $X \sim Gamma(\alpha, \beta)$ 이면 그 moment generating function은
<center>

$mgf_X(t)=(1-\beta t)^{-alpha}, t<1/\beta$
</center>

3. $X_1 \sim Gamma(\alpha_1, \beta), X_2 \sim Gamma(\alpha_2, \beta)$이고 $X_1, X_2$가 서로 독립이면
<center>

$X_1 + X_2 \sim Gamma(\alpha_1 + \alpha_2, \beta)$
</center>

### gamma distribution의 대의적 정의
 exponential distribution은 shape parameter가 1인 gamma distribution이다.
 이것으로부터 gamma distribution을 다음처럼 정의 할 수 있다.

 shape parameter r이 자연수인 경우에
 <center>

 $X \sim Gamma(r,\beta) \Leftrightarrow X \stackrel{d}{\equiv}Z_1 + \cdots + Z_r, Z_i \stackrel{iid}{\sim}Exp(\beta)$
 </center>

## normal distribution
### standard normal distribution
> 표준정규분포

binomial distribution의 cumulative probability를 적분으로 나타내는 근사식은 아래와 같다.
<center>

$\sum_{x:a \le {x-np \over \sqrt{np(1-p)}} \le b} {n \choose x} p^x (1-p)^{n-x} \sim \int_a^b{1 \over \sqrt{2 \pi}}e^{-{1\over 2}{z^2}}dz, \enspace n \rarr \infty$
</center>

이때, 아래의 함수는 그 적분 값이 1이 되는 함수로서 standard normal distribution의 pdf라고 한다.
<center>

$\phi(z) = {1\over \sqrt{2\pi}}e^{-{1\over 2}{z^2}}, \enspace -\infty < z < +\infty$
</center>

### normal distribution
> 정규분포, __$N(\mu, \sigma ^2)$__

일반적으로 아래의 식을 normal distribution의 pdf라고 한다.
<center>

${1\over \sigma} \phi({x - \mu \over \sigma})={1\over \sqrt{2 \pi}\sigma}e^{-{1\over 2}{(x-\mu)^2 \over \sigma^2}}, \enspace -\infty < x < +\infty$
$(\mu는\space실수,\space\sigma는\space양수)$
</center>

### normal distribution의 성질
1. $X \sim N(\mu, \sigma^2)$ 이면
<center>

$E(X) = \mu, Var(X)=\sigma^2$
</center>

2. $X \sim N(\mu, \sigma^2)$이면 그 mgf는
<center>

$mgf_X(t)=e^{\mu t + {1\over 2}\sigma^2 t^2}, \enspace - \infty < t < +\infty$
</center>

3. $X_1 \sim N(\mu_1, \sigma_1^2), X_2 \sim N(\mu_2, \sigma_2^2)$이고, $X_1, X_2$가 서로 독립이면
<center>

$X_1 + X_2 \sim N(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$
</center>

###normal distribution의 대의적정의
1.  $X \sim N(\mu, \sigma^2)$ 이면 상수 $a,b$에 대하여
<center>

$aX + b \sim N(a\mu + b, a^2\sigma^2)$
</center>

2.  $X\sim N(\mu, \sigma^2) \Leftrightarrow {X-\mu \over \sigma}\sim N(0,1) \Leftrightarrow X \stackrel{d}{\equiv}\sigma Z + \mu, \space Z \sim N(0,1)$

#### normal distribution의 cumulative probability
> 정규분포의 누적확률

아래와 같은 standard normal distribution의 cumulative distribution function는 아래와 같다.
<center>

$\Phi (x) = \int_{-\infty}^{x} {1\over \sqrt{2\pi}} e^{-e^2/2}dz$
</centeR>

이때, 일반적인 normal distribution $N(\mu, \sigma ^2)$를 따르는 $X$의 cumulative distribution은 다음과 같다.
<center>

$P(X \le x) = P({X-\mu \over \sigma} \le {x-\mu \over \sigma}) = \Phi({x-\mu \over \sigma})$
</centeR>

###quantile
> 분위수

$Z \sim N(0,1)$ 일때 아래의 식을 만족하는 값 $z_\alpha$를
standard normal distribution의 $upper\space \alpha \space quantile$이라고 한다.
<center>

$P(Z > z_\alpha) = \alpha(0<\alpha <1)$
</center>

---

# 기타 필요 정의
## 모집단
> population

통계 조사에서 관심의 대상이 N개의 개체일 때 이들 중에서 n개를 '랜덤'하게 택하여 조사한 후 전체에 대한 추측을 한다고 하자.
이때, 조사와 추측의 대상이 되는 전체를 모집단이라한다.

## 비복원추출
> sampling without replacement

축자적으로 한개씩 동일한 확률로 뽑아나가며 한 번 뽑힌 것은 되돌려 넣지 않는 추출 방법

## 단순랜덤추출
> sample random sampling

N개의 개체로 구성된 모집단에서 '랜덤'하게 n개를 비복원추출방식으로 추출하는 것

## 랜덤 표본
> random sample

단순랜덤추출 로 추출된 n개를 지칭한다. 간단히 표본(sample)이라고도 한다.

## 모비율
> population proportion

각 객체의 특성에 대한 분류를 0또는 1의 두가지 분류로 나타낼 때, 조사와 추측 대상이 되는 전체에 0과 1이 각각 $N-D$개, $D$개 있다고 하자. 이 때 1의 비율인 $p=D/N$를 모비율이라고 한다. 이때 모집단분포는 1과 0에 각각 $p$와 $1-p$를 대응시키는 분포이다.

간단히 말해서 전체 $N$개중에 값이 1(혹은 참)인 $D$개의 비율 $D/N$을 모비율이라고 한다.

## $X \stackrel{d}{\equiv} Y$의 의미
확률변수 X와 Y가 같은 분포를 갖는다.

## $iid$의 의미
> independent and identically distributed

위 영어 문장의 약어로, 서로 독립이고 같은 분포를 같는다는 뜻이다.
