# Probability
- 상대도수 추상화
- 가능성을 수량화

### 확률의 공리를 통해 구체적인 정의 도출
>Sample space : 가능한 모든 관측 결과들의 집합
Event : sample space의 부분 집합인 특정 결과들의 집합
Probability : 각 event의 가능성 수량화 사건 A의 Probability P(A)


이때, P(A)에 대한 공리는 다음과 같다.
>(1) (확률의 범위) 각 event A에 대해 $P(A)\ge0$
(2) (전체의 확률) sample space S에 대해 $P(S)=1$
(3) (countable additivity) event $A_1, A_2, …에 대해 A_i∩A_j=Ø(i≠j)$이면, $$P(A_1∪A_2∪…) = P(A_1) + P(A_2) + …$$
>* (3)의 공리는 사건의 확률을 구할 때, 사건을 공통의 원소를 갖지 않는 사건들로 나누어 구할 수 있음을 뜻함.

---------------------------------------


### 공리로 부터 얻어지는 기본 성질 - (1)
>(1) 각 사건 A에 대해 $0<=P(A)<=1$ 이고 P(Ø)=0
(2) (여사건 확률) $P(A^c) = 1 - P(A)$
(3) (단조성) $A⊆B$ 이면 $P(A)≤P(B)$


------------

#### 확률모형 예시
- 베르누이 시행
동전 던지기 처럼 관측 결과(event)가 Binary 형태인 실험을 베르누이 시행.
두 관측 결과를 흔히 success와 fail로 부름.
$S=\lbrace s,f \rbrace,\space P\lgroup \lbrace {s}\rbrace \rgroup = p, P \lbrace \lbrace f \rbrace\rbrace = 1-p \space (0<=p<=1)$

 ----

### 공리 부터 얻어지는 기본성질 -(2) 합사건의 확률
>$
P(A_1∪A_2) = P(A_1)+P(A_2)-P(A_1∩A_2) \newline
P(A_1∪A_2∪A_3) = P(A_1)+P(A_2)+P(A_3) - P(A_1∩A_2)-P(A_2∩A_3)-P(A_1∩A_3) +P(A_1∩A_2∩A_3)\newline
…\newline
P(A_1∪A_2∪…) = P(A_1)+P(A_2)+…+P(A_n)-P(A_1∩A_2)-…-P(A_{n-1}∩A_n)+…+(-1)^{(n-1)}P(A_1∩A_2∩…∩A_n)\newline
(Countable\space subdadditivitiy)\newline
P(A_1∪A_2∪…)<=P(A_1)+P(A_2)+…\newline$
$[Proof] \space Countable \space Additivity$ + 귀납법을 이용하여 유도
$$A_1∪A_2 =  (A_1-A_2)∪(A_2-A_1)∪(A_1∩A_2) \newline
A_1 = (A_1-A_2) ∪ (A_1∩A_2)\newline A_2 = (A_2-A_1) ∪ (A_1∩A_2)\newline$$
*\*n번째 사건에 대한 것은 위의 사건과 n번째 식의 귀납법으로 유도*


----------

### Continuity of Probability Measure
> (1) $A_1 \subseteq A_2 \subseteq ... \subseteq A_n \subseteq ...\space$이면 $P(\bigcup\limits_{n=1}^\infty A_n)=\lim\limits_{n\to\infty}P(A_n)$
 (2) $A_1 \supseteq A_2 \supseteq ... \supseteq A_n \supseteq ...\space$이면 $P(\bigcap\limits_{n=1}^\infty A_n)=\lim\limits_{n\to\infty}P(A_n)$
 >
 >*\*countable addtivity를 이용하여 구한다.*

$$ P(\lim\limits_{n \to \infty}A_n) = \lim\limits_{n \to \infty}P(A_n)
$$
**포함관계가 커지거나 작아지는 사건들에 대해서는 연속인 실수 함수와 같이 확률의 계산이 가능하다는 의미.**
#### 연속 분포일때 한점을 정의하는 방법
> 표본공간이 $S=[0,1]$이고, $P((a,b])=b-a (b<a)$ 일때, 한 점의 확률
$P(\lbrace b \rbrace)P(\bigcap\limits_{n=1}^{\infty}(b-\frac{1}{n},b])=\lim\limits_{n\to\infty}P((b-\frac{1}{n},b])=\lim\limits_{n\to\infty}\frac{1}{n}=0$

------

# Conditional Probability

>Conditional Probability:
event A 가 주어진 경우에 event B의 조건부확률은
$P(B|A)=\frac{P(A\cap B)}{P(B)}$ (단, $P(A)>0$)

**한 사건 A가 일어났다는 전제하에서 사건 B가 일어날 가능성을 생각하는 경우의 확률**
**주어진 사건 A를 새로운 표본공간과 같이 간주하여 축소된 실험에서의 가능성을 의미**

---------------------
## Conditional Probability의 성질

> Multiplicity  : $P(A\cap B)= P(B|A)P(A)$
Total Probability : $A_1, A_2,A_3...$이 subspace S를 partition하고 $P(A_j)>0$이면,
$$P(B) = P(B|A_1)P(A_1)+P(B|A_2)P(A_2)+...$$


-----

### 예제
> 번호가 0번부터 N번까지 붙여진 N+1개의 상자가 있고, k(k=0,1,...)번 상자에는 빨간 공 k개와 흰 공 N-k개가 들어 있다.
이제 한 상자를 랜덤하게 택한 후에 그 상자에서 공을 한 개 꺼내어 색깔을 확인하고 도로 넣는 작업을 n번 반복하였더니 n번 모두 빨간 공이었다.
같은 상자에서 공을 한번 더 꺼내어 확인할 때 또 다시 빨간 공이 뽑힐 확률을 구하고, N이 큰 경우에 확률의 근사공식도 구하여라. (p.16)
> 라플라스의 해 뜰 확률 (오늘까지 해가 뜬 전제로 내일 해가 뜰 확률)
------------

## Bayes theorem
> Event $A_1, A_2, ...$이 Sample space S를 공통부분이 없게 partitioning하고, $P(A_i)>0$일 때, $P(B)>0$이면,
$$P(A_j|B) \varpropto P(B|A_j)P(A_j) \space (j=1,2,...)$$
이고, 비례상수는 좌변의 합이 1(sample space를 partition)임으로 부터 결정
즉, $A_j$가 sample space를 파티셔닝 할때, prior인 $P(A_j)$로 $P(A_j|B)$ 추론  

>$[Proof]$
>$
\begin{aligned}
 	P(A_j|B)P(B)&= P(B\cap A)\\
	&=P(B|A_j)P(A_j) \space (j=1,2,..) \\
\end{aligned}
$
>$
\begin{aligned}
 	P(A_j|B)&= \frac{P(B|A_j)P(A_j)}{P(B)}\\
	&=\frac{P(B|A_j)P(A_j)}{P(B|A_1)P(A_1)+P(B|A_2)P(A_2)+...}  \\
  &=P(A)
\end{aligned}
$

**베이지안 추론의 근본 $P(A_1), P(A_2), ...$는 여러 모형의 가능성을 의미하고,
 $P(A_1|B),P(A_2|B),...$는 실험 결과를 뜻하는 B의 관측 후에 각 모형의 가능성을 뜻함**
 >**$P(A_j)$: 사전확률 (Prior)**
 **$P(A_j|B)$: 사후확률 (Posterior)**
 베이즈 정리는 이 둘 사이의 관계를 의미
 실험을 통해 B라는 sample space에서 A event를 관찰하고 P(A)를 추론함
-----

## Independece
> Event A의 관측 여부가 Event B가 일어날 가능성에 아무런 영향을 주지않는 경우
$$P(B|A)=P(B)\newline
P(A\cap B) = P(A)P(B)
$$
A, B, C Event에 대해 A,B 각 Event와 C와의 독립이 $A\cap B$와 C의 독립을 보장하지 않는다.

>예시
A= {첫 번째 주사위의 눈이 홀수}
B= {두 번째 주사위의 눈이 짝수}
C= {두 주사위의 눈의 합이 홀수}
C와 A, B 각각은 독립이지만, C와 $A\cap B$는 독립이 아니다.

>세 개 이상의 Event의 독립성은 다음과 같고, 독립이 아닐시에는 mutally Independent라고 함.
(1) 두 사건이 서로 독립
  $$P(A\cap B)= P(A)P(B)$$
(2) 세 사건이 서로 독립
$$P(A\cap B)= P(A)P(B),\space P(B\cap C)= P(B)P(C),\space P(C\cap A)= P(C)P(A) \\
P(A\cap B\cap C) = P(A)P(B)P(C)
$$
(3) n개 사건이 서로 독립
$$P(A_i\cap A_j)=P(A_i)P(A_j)\space (1\le i \lt j \le n) \\
P(A_i \cap A_j \cap A_k)=P(A_i)P(A_j)P(A_k)\\
... \\
P(A_1\cap ... \cap A_n)=P(A_1)...P(A_n)
$$

---

#Random Variable Probability distribution
---
---
## 랜덤 실험
> 일반적으로, 여러 가지의 결과가 가능하고 그 가능성을 확률로 나타낼 수 있는 실험
---
## 확률 변수
> 랜덤 실험의 모든 가능한 결과의 집합인 표본공간에서 정의된 실수 값 함수
---
## 확률 분포표
> 확률 변수 X의 값을 도표로 나타낸 것
 확률변수가 가질 수 있는 값이 유한개인 경우, 확률 변수를 나타내기 위해 사용한다.

- EX)

| X의 값     | 0     | 1    | 2    |
| :------------- | :------------- | :------------- | :------------- |
| 확  률     | 1/4      | 1/2       |1/4       |

---

## 확률질량함수(Probability mass function)
> 확률밀도함수(Probability density function)라고도 불린다.

### 정의
> 이산형

아래의 식과 같이 이산형 확률 변수를 나타내는 함수를 말한다.
<center>

$
f(x_k) = \Rho(X=x_k) \space (k=1, 2, 3, \cdots)
$
</center>

> 연속형

아래의 식과 같이 연속형(실수개) 확률 변수를 나타내는 함수를 말한다.
<center>

$
\int_a^bf(x)dx = \Rho(a \le x \le b) \space (- \infty \lt a \lt b \lt + \infty)
$
</center>

### 성질

> 이산형 확률변수의 확률밀도함수 성질:

  - $ f(x) \ge 0 \hspace{1em} \forall x: - \infty \lt x \lt + \infty $
  - $ \sum_{x}f(x) = \sum_{k=1}^\infty f(x_k) = 1 $
  - $ \sum_{x \space : \space a \le x \le b} f(x) = \Rho(a \le X \le b) $

> 연속형 확률변수의 확률밀도함수 성질:

  - $ f(x) \ge 0 \hspace{1em} \forall x: - \infty \lt x \lt + \infty $
  - $ \int_{- \infty}^{+ \infty} f(x) dx = 1$
  - $ \int_a^b f(x) dx = \Rho (a \le X \le b) \hspace{1em} (- \infty \lt a \lt b \lt + \infty) $

### 지표함수
>연속형 확률밀도함수의 범위를 가리키는 함수

<center>

$
I_A(x) = \begin{cases}
             1 \enspace x \in A
             \\
             0 \enspace x \notin A
         \end{cases}
$
</center>

>  다른 표기

<center>

$ I_A(x) == I_{(x \in A)}$
</center>

### 확률 분포
> 확률밀도함수에 의해 정해지는 실직선 위에서의 확률의 분포, 아래와 같이 나타낸다.

<center>

$
\Rho(a \le X \le b) \space (- \infty \lt a \lt b \lt + \infty)
$
</center>

> 기호
<center>

$
 X \thicksim f(pdf)
$
</center>

---

#확률분포의 특성치

---
---

## 평균
> 분포의 위치를 나타내는 값, 기댓값(expeected value) (이)라고도 한다.

### 정의
>확률 변수 $X$의 확률밀도함수가 f일때, 실수 값 함수 $g(x)$에 대하여 아래 식이

<center>

$
\begin{cases}
\sum_x g(x)f(x) \hspace{2.1em} (X가 이산형일 때)
\\
\int_{- \infty}^{+ \infty} g(x)f(x)dx \enspace (X가 연속형일 때)
\end{cases}
$

</center>

>실수로 정의되면 그 값을 $ g(x) $의 기댓값이라 하고, 기호로는 $ E[g(X)] $로 나타낸다.
그 표기는 다음과 같다.
<center>

$
E[g(X)] = \mu = \begin{cases}
          \sum_x g(x)f(x) \hspace{2.1em} (X가 이산형일 때)
          \\
          \int_{- \infty}^{+ \infty} g(x)f(x)dx \enspace (X가 연속형일 때)
          \end{cases}
$
</center>

+ 일반적인 경우 위의 $ g(x) = x$ 이다.
- <span style="color:red"> 실수로 정의</span>된다는 의미는 위의 식에서 합 혹은 적분이 <span style="color:red">수렴</span>한다는 것이다.


### 성질
>선형성

<center>

$\space E(aX + b) = aE(X)+b \enspace (a, b는 \space 확률변수가 \space 아닌 \space 상수)$
</center>

>선형성
<center>

$\space E[c_1g_1(X) + c_2g_2(X)] = c_1E[g_1(X)] + c_2E[g_2(X)] \enspace (c_1, c_2는 \space 상수)$
</center>

>단조성
<center>

$\space g_1(X) \le g_2(X)이면 \space E[g_1(X)] \le E[g_2(X)]$
</center>

---

## 확률분포의 분산과 표준편차
### 정의
> 분산 : 분포가 평균값을 기준으로 어떻게 산포되어 있는가를 나타내는 특성치, $ (X-\mu)^2$
표준편차 : 분산의 제곱근

>확률변수 $X$의 확률밀도함수가 $ f $ 이고 평균이 $ \mu $일 때, X의 확률분포의 분산과 표준편차는 각각
<center>

$
Var(X) = E[(X-\mu)^2] =
\begin{cases}
\sum_x (x-\mu)^2 f(x) \hspace{2.1em} (X가 이산형일 때)
\\
\int_{- \infty}^{+ \infty} (x-\mu)^2f(x)dx \enspace (X가 연속형일 때)
\end{cases}
$

$
Sd(X) = \sigma = \sqrt{Var(X)}
$
</center>

###성질
- $Var(aX+b)=a^2Var(X) \enspace (a,b는 \space 상수)$

- $Var(X) = E(X^2) - \{E(X)\}^2$

### 표준화(standardized)
>평균이 $ \mu $이고 표준편차가 $\sigma$이고 $\sigma \gt 0$일 때

<center>

$
Z = {X-E(X) \over \sqrt{Var(X)}} = {X-\mu \over \sigma}
$
</center>

>일 때, 다음을 표준화된 확률변수 X라고 한다.

<center>

$
E(Z) = (E(X)-\mu)/\sigma = 0, \space Var(Z) = Var(X) / \sigma^2 = 1
$
</center>

---
#누적붙포함수와 누율생성함수
---
---
## 누적분포함수(cumulative distribution function)
### 정의
> 이산형 확률변수의 누적분포함수


  - 이산형 확률변수 X의 확률밀도함수가
  <center>

  $ f(x_k) = \Rho (X=x_k) \space (k = 1,2, \cdots), \space f(x) = 0 \enspace \forall x : x \neq x_k(k = 1, 2, \cdots)$
  </center>
  &emsp; &nbsp;일 때, X의 누적 확률을 나타내는 함수
  <center>

  $ F(x) = \Rho (X \le x) = \sum_{t \space : \space t \le x} f(t)$
  </center>

  - $ x_1 \lt x_2 \lt \cdots \lt x_{n-1} \lt x_n \lt \cdots $ 인 경우
  <center>

  $ F(x_n) = \sum_{k=1}^n f(x_k) = F(x_n) - F(X_{n-1})$
  </center>

> 연속형 확률변수의 누적분포함수

  - 연속형 확률변수 X의 확률밀도함수가 $f$일 때, $X$의 누적 확률을 나타내는 함수
  <center>

  $
  F(x) = \Rho(X \le x) = \int_{- \infty}^x f(t)dt
  $
  </center>

  &emsp; &nbsp;를 X의 누적분포함수라고 한다. 특히, $f$가 연속인 $x$에 대하여는
  <center>

  $
  {d \over dx}F(x) = f(x)
  $
  </center>

> 기호
<center>

$ X \thicksim F(cdf)$
</center>

### 성질
> 증가성
<center>

$
x_1 \lt x_2이면 F(x_1) \le F(x_2)
$
</center>

>전체 변동

<center>

$
\lim_{x \rarr -\infty}F(x) = 0, \enspace \lim_{x \rarr + \infty}F(x) = 1
$
</center>

>오른쪽 연속성

<center>

$
\lim_{\substack{h \rarr 0 \\ h \gt 0}}F(x+h) = F(x)
$
</center>

### 표준지수분포(standard exponential distribution)

> 확률변수 X의 확률밀도함수가 $f(x) = e^{-x}I_{(x \ge 0)}$ 일 때, X의 누적분포함수는 다음과 같다.

<center>

$
F(x) = \int_{- \infty}^x e^{-t}I_{(x \ge 0)}dt
     = \begin{cases}
     0, \hspace{8.1em} x \lt 0
     \\
     \int_0^x e^{-t}dt = 1-e^{-x}, \enspace x\ge 0
     \end{cases}
$
</center>

---
##확률생성함수
> 음이 아닌 정수의 값을 가질 수 있는 이산형 확률변수 X의 확률생성수는 아래와 같은 형태를 가진다.

<center>

$G(s) = E(s^X) = \sum_{x=0}^\infty s^x \Rho(X=x), \enspace -1 \lt x \lt 1$
</center>

> 위 확률생성함수를 이용해 생성되는 확률밀도함수는 아래와 같다.

<center>

$f(k) = \Rho(X=k) = G^{(k)}(0)/k!, \enspace k = 0, 1, 2, \cdots$
</center>

---

## 적률 생성함수

### 적률(moment)
> $E(| X |^k) \lt + \infty$ 일떄 아래와 같은 식을 확률변수 x의 k차 적률이라고 한다.

<center>

$m_k = m_k(X) = E(X^k) =
\begin{cases}
\sum_x x^kf(x) \hspace{3.5em} (X가 \space 이산형 \space 일때)
\\
\int_{-\infty}^{+\infty} x^kf(x)dx \hspace{2em} (X가 \space 연속형 \space 일때)
\end{cases}$
</center>



### 적률생성함수
> 0을 포함하는 열린구간 t 값에 대하여 $E(e^{tX})$가 실수일 때, 아래와 같은 함수를 확률 변수 X의 적률생성함수라고 한다.

<center>

$M(t) = E(e^{tX}), \enspace -h \lt t \lt h \enspace (\exist h \gt 0)$
</center>

### 성질
> 적률생성 성질

 확률 변수 X의 적률생성함수가 존재하면, 즉 아래의 1번식이 참이면, X의 모든 적률이 존재하고 2번식이 참이된다.
 <center>

 1. $
 M(t) = E(e^{tX}) \lt + \infty \enspace \forall t : -h \lt t \lt h \space (\exist h \gt 0)
 $
 2. $
 E(X^k) = M^{(k)}(0), M(t) = \sum_{k=0}^\infty{E(X^k) \over k!}t^k, \space -\epsilon \lt t \lt \epsilon \space (\exist \epsilon \gt 0)
 $

 </center>

- 확률변수의 모든 적률이 존재한다고 해서 적률생성함수가 존재하는 것은 아니다.

 > 분포 결정성

 두 확률변수 $X, Y$의 적률생성함수 $M_X(t), M_Y(t)$가 존재하고 0을 포함하는 열린구간에서 일치한다면,
 즉 아래의 식이 참이면, $X$와 $Y$의 확률분포가 일치한다.

 <center>

 $M_X(t)=M_Y(t) \enspace \forall t : -h \lt t \lt h \space (\exist h \gt 0)$
 </center>

---

## 누율생성함수

### 누율생성함수(cumulant generating function)
> 적률생성함수 $M(t)$가 존재할 때, 로그를 취하여 얻어지는 함수

<center>

$C(t) = logM(t) = logE(e^tX), \space -h \lt t \lt h \space (\exist h \gt 0)$
</center>

### 누율(cumulant)
> 아래의 누율생성함수의 멱급수 전개식에서 $C(0)=0$이고 $t^r / r!$의 계수인 $C^{(r)}(0)$를 X의 r차 누율이라고 한다.
기호는 $c_r(X)$, $c_r$(이)다.

<center>

$C(t) = \sum_{r=0}^\infty {C^{(r)}(0) \over r!}t^r, \space -h \lt t \lt h(\exist h \gt 0), r=1,2,\cdots$

$C(t) = \sum_{r=1}^\infty{c_r(X) \over r!}t^r, \space -h \lt t \lt h(\exist h \gt 0)$
</center>

### 누율과 적률의 관계 및 누율의 특징
> 누율과 적률의 관계는 아래의 식으로부터 얻을 수 있다.

<center>

$
log(1+\sum_{k=1}^\infty{m_k \over k!}t^k) = \sum_{r=1}^\infty{c_r \over r!}t^k, \space -h \lt t \lt h(\exist h \gt 0)
$
</center>

> 위 식을 아래 1번의 로그함수의 멱급수 전개식을 이용하여 전개하고, 아래 2번처럼 t의 오름차순으로 결과를 정리하면,
아래 3번처럼 일차, 이차 누율이 각각 평균과 분산임을 알 수 있다.

<center>

1. $log(1+A) = A - A^2/2 + A^3/3 - \cdots, \space (-1 \lt A \lt)$
2. $A=m_1t + m2t^2/2! + m_3t^3/3! + \cdots$
3. $c_1 = m_1, c_2=2!{m_2/2! - (m_1)^2/2} = m_2 - (m_1)^2$
</center>

---
#여러가지 부등식
---
---
##젠셴(Jensen)의 부등식
> 수직선 위의 구간 I에서의 값을 갖는 확률변수 X의 기댓값이 존재하면,
구간 I에서 볼록한 함수 $\phi$에 대하여 다음 부등식이 성립한다.

<center>

$
\phi(E(X)) \le E(\phi(X))
$
</center>

##리아푸노프의 부등식
> 확률변수 X에 대하여 $E(|X|^s) \lt \infty$이면 $0 \lt r \lt s$인 $r$에 대해 다음 부등식이 성립한다.

<center>

$(E(|X|^r))^{1/r} \le (E(|X|^s))^{1/s}$
</center>

## 마코프 부등식과 체비셰프 부등식
>마코프 부등식

확률변수 Z에 대하여 $E(|Z|^r) \lt \infty (r\gt 0)$이면, 임의의 양수 $k$에 대하여 아래의 부등식이 성립한다.
<center>

$\Rho(|Z| \ge k) \le E(|Z|^r)/k^r$
</center>

>체비셰프의 부등식

확률변수 X에 대해 $Var(X) \lt \infty$이면, 임의의 양수 $k$에 대해 아래의 부등식이 성립한다.
<center>

$\Rho(|X-E(X)| \ge k) \le Var(X)/k^2$
</center>

---
---

# Extra
## 연속형 확률밀도함수로 정해지는 확률
> 연속형 확률밀도함수로 정해지는 확률은 근사값임을 설명하는 수식이다.-

<center>

 $ \Rho (X = a) = \int_a^a f(x)dx = 0 $

 $\therefore \Delta x \fallingdotseq 0, \Rho (a \le X \le a + \Delta x) \fallingdotseq f(a) \Delta x$
</center>

## 지수 함수의 멱급수 전개식

>적률과 관련하여 사용된다.

<center>

$
e^a = 1 + {a \over 1!} + {a^2 \over 2!} +{a^3 \over 3!}+ \cdots + {a^k \over k!} + \cdots
$
</center>

## 테일러 정리
> 적률생성함수의 성질 중 적률생성 성질을 증명하는데 사용된다.
젠셴의 부등식 증명에서도 사용한다.

// 추후 정리하겠음.
## 볼록함수
>다음 부등식을 만족하는 함수
<center>

$
\phi(\lambda x + (1-\lambda)y) \le \lambda \phi(x) + (1-\lambda) \phi(y) \space \forall \lambda : 0 \le \lambda \le 1, \space \forall x \in I, \space \forall y \in I
$
</center>

## 분산 0의 의미
>확률변수 X의 평균이 $\mu = E(x)$이고 $Var(x) = 0$이면 $P(X=\mu) = 1$(이)다.
