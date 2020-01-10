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

# Random Variable Probability distribution

>TBD
