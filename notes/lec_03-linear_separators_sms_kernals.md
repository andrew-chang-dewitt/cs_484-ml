---
title: "ML: linear separators & sms kernals"
description: "Introducing linear separators & sms kernals & related topics/algos; e.g. Perceptron algorithm, Lagrangian Duality, etc."
keywords:
  - "linear separators"
  - "perceptron algorithm"
  - "soft margin classification"
  - "machine learning"
  - "data structures"
  - "algorithms"
  - "lecture notes"
  - "computer science"
  - "cs 484"
  - "illinois tech"
meta:
  byline: Andrew Chang-DeWitt
  published: "2025-09-10T00:00-06:00"
  updated: "2025-09-23T00:00-06:00"
---

:::aside{.right-aside.after#toc}

## contents

- linear separator
- perceptron algorithm
- support vector machines

:::

## linear separator

_**def:** linearly separable_&mdash;a characteristic of two datasets that is
true when all values in one set can be differentiated from all values in the
other by drawing a vector between the two

More formally, for two datasets of $n$ pairs, $P$ & $Q$, they are _linearly
separable_ if and only if there is a set of $n + 1$ real numbers that
satisfies the following:

$$
\begin{aligned}
    && \because
       && P \land Q &\models {(x_1,y_1),(x_2,y_2),...,(x_n,y_n)} && \textit{data are sets of pairs} \\
    && && \exists W &= {w_1,w_2,...,w_n,k} \\
(2) && && wx &< k \forall w \in W[1,n], x \in P \\
(3) && && wx &> k \forall w \in W[1,n], x \in Q \\
    && \therefore
       && (2) \land &(3) \iff LinSep(P,Q) \\
\end{aligned}
$$

_binary classification $y_i \in \{-1,1\}$_&mdash;the act of classifying
values into one of two values&mdash; can be modeled using linear separation. by
placing values in feature space, if a linear separator can be found, then it
can be used as a hypothesis to divide the entire feature space, thus useful for
prediction and other ml tasks.

- _hypothesis class_ of linear _decision surfaces_
  is $f(x_i) = sign(w^T x_i + b)$
- if we assume $b \equiv 0$, simplifies
  to $f(x_i) = sign(w^T x_i)$
- when doing superivsed learning, our output for each data point ( $y_i$ ) is
  known; thus we can say $y_i(w^T x_i) > 0 \iff x_i$ is correctly classified
  because if $y_i$ & $w^T x_i$ are both of the same sign, then we know our
  training found the correct _linear separation_ vector model

## perceptron algorithm

one of the oldest machine learning algorithms, created in the 1950s. not
typically used today, but good for introducing how a linear separator is used.

english description:

1. start @ $T = 1$ w/ $w_1$ as a _zero vector_
2. given some example $x$, predict $Positive(y) \iff w_1 \cdot x >= 0$
3. On a mistake, update vector weights as follows:
   - if should have been positive, then update $w_{t + 1} \gets w_t + x$
   - if should have been negative, then update $w_{t + 1} \gets w_t - x$

ex:

given data:

| i   | 1     | 2     | 3     | 4     |
| --- | ----- | ----- | ----- | ----- |
| x   | (1,2) | (2,3) | (2,1) | (3,0) |
| y   | +     | +     | -     | -     |

iterations:

$$
\begin{aligned}
(1) && \because
       && p(i) &= \Bigg\lbrace\begin{aligned}
                    +, & \text{ if } w^T \cdot x_i >= 0 \\
                    -, & \text{ otherwise}
                  \end{aligned}, &&
    \textit{$p(i)$ predicts classification of $x_i$} \\
(2) && && q(i) &= \Bigg\lbrace\begin{aligned}
                    w^{T - 1} + x_i, & \text{ if } y_i \equiv + \\
                    w^{T - 1} - x_i, & \text{ if } y_i \equiv - \\
                  \end{aligned} &&
    \textit{$q(i)$ gives new $w^T$ value from $x_i$ \& $y_i$} \\
\\
    && &&   T &:= 1 \\
(3) && && w^T &:= \begin{bmatrix} 0 & 0 \end{bmatrix} &&
    \textit{init $T$ counter \& $w$ vector} \\
\\
    && && y_1 &\equiv p(1) &&
    \textit{check prediction of $x_1$} \\
    && &&     &\equiv + \iff \begin{bmatrix} 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 \end{bmatrix} >= 0 &&
    \textit{\{1,3\}} \\
    && &&     &\equiv + \iff (0 * 1) + (0 * 2) >= 0 \\
    && &&     &\equiv + \iff 0 >= 0 \\
    && &&   + &\equiv + &&
    \textit{prediction is correct, continue} \\
\\
    && && y_2 &\equiv p(2) &&
    \textit{check prediction of $x_2$} \\
    && &&     &\equiv + \iff \begin{bmatrix} 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} 2 & 3 \end{bmatrix} >= 0 &&
    \textit{\{1,3\}} \\
    && &&     &\equiv + \iff (0 * 2) + (0 * 3) >= 0 \\
    && &&     &\equiv + \iff 0 >= 0 \\
    && &&   + &\equiv + &&
    \textit{prediction is correct, continue} \\
\\
    && && y_3 &\equiv p(3) &&
    \textit{check prediction of $x_3$} \\
    && &&     &\equiv + \iff \begin{bmatrix} 0 & 0 \end{bmatrix} \cdot \begin{bmatrix} 2 & 1 \end{bmatrix} >= 0 &&
    \textit{\{1,3\}} \\
    && &&     &\equiv + \iff (0 * 2) + (0 * 1) >= 0 \\
    && &&     &\equiv + \iff 0 >= 0 \\
    && &&   - &\equiv + \implies \bot &&
    \textit{prediction is incorrect, update $w^T$} \\
\\
    && &&   T &:= 2 \\
    && && w^T &:= q(3) \\
    && &&     &= \begin{bmatrix} 0 & 0 \end{bmatrix} - \begin{bmatrix} 2 & 1 \end{bmatrix} &&
    \textit{\{2,3\}} \\
(4) && && w^T &= \begin{bmatrix} -2 & -1 \end{bmatrix} \\
\\
    && && y_4 &\equiv p(4) &&
    \textit{check prediction of $x_4$} \\
    && &&     &\equiv + \iff \begin{bmatrix} -2 & -1 \end{bmatrix} \cdot \begin{bmatrix} 3 & 0 \end{bmatrix} >= 0 &&
    \textit{\{1,4\}} \\
    && &&     &\equiv + \iff (-2 * 3) + (-1 * 0) >= 0 \\
    && &&     &\equiv + \iff -6 >= 0 \\
    && &&   - &\equiv - &&
    \textit{prediction is correct, go back to $x_1$} \\
\\
    && && y_1 &\equiv p(1) &&
    \textit{check prediction of $x_1$} \\
    && &&     &\equiv + \iff \begin{bmatrix} -2 & -1 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 \end{bmatrix} >= 0 &&
    \textit{\{1,4\}} \\
    && &&     &\equiv + \iff (-2 * 1) + (-1 * 2) >= 0 \\
    && &&     &\equiv + \iff -4 >= 0 \\
    && &&   + &\equiv - \implies \bot &&
    \textit{prediction is incorrect, update $w^T$} \\
\\
    && &&   T &:= 3 \\
    && && w^T &:= q(1) \\
    && &&     &= \begin{bmatrix} -2 & -1 \end{bmatrix} + \begin{bmatrix} 1 & 2 \end{bmatrix} &&
    \textit{\{2,4\}} \\
(5) && && w^T &= \begin{bmatrix} -1 & 1 \end{bmatrix} \\
\\
    && && y_2 &\equiv p(2) &&
    \textit{check prediction of $x_2$} \\
    && &&     &\equiv + \iff \begin{bmatrix} -1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 2 & 3 \end{bmatrix} >= 0 &&
    \textit{\{1\}} \\
    && &&     &\equiv + \iff (-1 * 2) + (1 * 3) >= 0 \\
    && &&     &\equiv + \iff 1 >= 0 \\
    && &&   + &\equiv + &&
    \textit{prediction is correct, continue} \\
\\
    && && y_3 &\equiv p(3) &&
    \textit{check prediction of $x_3$} \\
    && &&     &\equiv + \iff \begin{bmatrix} -1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 2 & 1 \end{bmatrix} >= 0 &&
    \textit{\{1,5\}} \\
    && &&     &\equiv + \iff (-1 * 2) + (1 * 1) >= 0 \\
    && &&     &\equiv + \iff -1 >= 0 \\
    && &&   - &\equiv - &&
    \textit{prediction is correct, continue} \\
\\
    && && y_4 &\equiv p(4) &&
    \textit{check prediction of $x_4$} \\
    && &&     &\equiv + \iff \begin{bmatrix} -1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 3 & 0 \end{bmatrix} >= 0 &&
    \textit{\{1,5\}} \\
    && &&     &\equiv + \iff (-1 * 3) + (1 * 0) >= 0 \\
    && &&     &\equiv + \iff -3 >= 0 \\
    && &&   - &\equiv - &&
    \textit{prediction is correct, go back to $x_1$} \\
\\
    && && y_1 &\equiv p(1) &&
    \textit{check prediction of $x_1$} \\
    && &&     &\equiv + \iff \begin{bmatrix} -1 & 1 \end{bmatrix} \cdot \begin{bmatrix} 1 & 2 \end{bmatrix} >= 0 &&
    \textit{\{1\}} \\
    && &&     &\equiv + \iff (-1 * 1) + (1 * 2) >= 0 \\
    && &&     &\equiv + \iff 1 >= 0 \\
    && &&   + &\equiv + &&
    \textit{prediction is now correct for all training values} \\
\\
    && && p(i) &\text{ correct with } w^3 \space\forall \space x_i \\
    && \therefore
       && LinSep &\equiv w^3 \cdot x \\
    && &&        &\equiv \begin{bmatrix} -1 & 1 \end{bmatrix} \cdot \begin{bmatrix} x & y \end{bmatrix} &&
    \textit{\{5\} slides use $\begin{bmatrix} x_1 & x_2 \end{bmatrix}$} \\
    && && LinSep &\equiv -x + y = 0 \\
    && && LinSep &\equiv y = x \quad\blacksquare \\
\end{aligned}
$$

### geometric margin

_**def:** geometric margin $\gamma$_&mdash;the distance from some linear
separator, $w$, to some example datapoint $x$.

used to define an upper bound on number of allowable mistakes in linear
separator model. by allowing some mistakes, we can achieve better training time
complexity while still allowing for a well-defined degree of accuracy.

> [!IMPORTANT]
>
> #### theorem
>
> if data has a margin $\gamma$ and all pionts lie inside a ball
> of radius $R$, then Perceptron algorithm makes $<= \frac{R}{\gamma^2}$
> mistakes.

## support vector machine (svm)

a more sophisticated model utilizing linear separators.

_**def:** support vector machine_&mdash;a supervised _max-margin_ maodel w/
associated learning algorithms.

why use it?

- good at generalizing in theory & in practice
- works well w/ small training datasets
- finds the globally best model
- efficient algorithms
- amenable to the _kernal trick_ (more below)

### support vectors

_**def**_&mdash;the closest datapoints to the linear separator.

used to allow finding the _optimal_ linear separator (i.e. vector w/ widest
_margin_) by only checking the closest datapoints.

### optimizing $\rho$

given some training set, $T$, separated by a _hyperplane_, $w$, with a margin
, $\rho$, then the following holds for each example in the training set:

$$
\begin{aligned}
\because  && T &:= \{(x_i,y_i) \big| i \in [1,n], x_i \in \R^d, y_i \in \{-1,1\}\} \\
\\
\therefore && y_i &(w^T x_i + b) >= 1 \iff \Bigg\lbrace\begin{aligned}
                                             w^T + x_i >= 1, & \text{ if } y_i \equiv 1 \\
                                             w^T + x_i <= -1, & \text{ if } y_i \equiv -1
                                           \end{aligned} \\
\end{aligned}
$$

which then allows defining $\rho$ as:

$$
\begin{aligned}
\rho &:= \frac{2}{\big\Vert w \big\Vert}
\end{aligned}
$$

to find the most optimal $\rho$ & thus must optimal linear separator, we
consider the above as a _quadratic optimization problem_:

$$
\begin{aligned}
&& \text{f}&\text{ind $w$ \& $b$ such that} \\
&& \rho &= \frac{2}{\big\Vert w \big\Vert} \text{ is maximized} \\
&& \text{a}&\text{nd } \forall (x_i,y_i) \in T: y_i(w^T x_i + b) >= 1 \text{ holds}
\end{aligned}
$$

which can be rewritten as:

$$
\begin{aligned}
&& \forall &(x_i,y_i) \in T \text{ find $w$ \& $b$ such that} \\
&& Q&(w) = \frac{1}{2}\big\Vert w \big\Vert^2 = \frac{1}{2} w^T w \text{ is minimized} \\
&& \text{a}&\text{nd } y_i(w^T x_i + b) >= 1 \text{ holds}
\end{aligned}
$$

however, that still requires knowing all the weights & biases for
every $t_i \in T$ at each $t_i$, which becomes computationally large for large
datasets. luckily this can be further simplified.

### Lagrangian duality

by constructing a dual problem wherein _Lagrange multipliers_, $\alpha_i$, are
associated w/ all inequality constraints $y_i(w^T x_i + b) >= 1$ in original

$$
\begin{aligned}
    && \forall &i \in [1,n] \text{ find } \alpha_i \text{ such that} \\
(6) && Q&(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j \text{ is maximized} \\
    && \text{a}&\text{nd } \alpha_i >= 0 \land \sum_i \alpha_i y_i \equiv 0 \text{ holds}
\end{aligned}
$$

how is this definition still the same problem?

to find out, we minimize $\frac{1}{2} w^T w$ s.t. $1 - y_i(w^T x_i + b) <= 0, \forall (x_i,y_i) \in T$
by constructing the Lagrangian $\mathcal{L} = \frac{1}{2} w^T w + \sum_i \alpha_i(1 - y_i(w^T x_i + b))$
then taking it's derivative w/ respect to $w$ setting it equal to $0$ & solving for $w$.

$$
\begin{aligned}
\because
&& \frac{\partial \mathcal{L}}{\partial w} &= 0 \\
&&                                       0 &= w + \sum_i \alpha_i (-y_i) x_i \\
&&                                       w &= \sum_i \alpha_i y_i x_i \\
\\
\land
&& \frac{\partial \mathcal{L}}{\partial b} &= 0 \\
&&                                       0 &= \sum_i \alpha_i y_i \\
\\
\therefore
&& \mathcal{L} &= \frac{1}{2} \sum_i \alpha_i y_i x_i^T \sum_j \alpha_j y_j x_j
                  + \sum_i \alpha_i(1 - y_i(\sum_j \alpha_j y_j x_j^T x_j + b)) \\
&&             &= \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j
                  + \sum_i \alpha_i
                  - \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_j^T x_i + b(\sum_i \alpha_i y_i) \\
&& \mathcal{L} &= - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j
                  + \sum_i \alpha_i \quad\blacksquare
\end{aligned}
$$

### svm training

using this optimized formula for maximizing $\rho$, we can train an SVM model by:

1. solving the optimization problem $(6)\space Q(\alpha)$ to obtain $\alpha_1...\alpha_n$,
   including computing the _inner products_ $x_i^T x_j$ between _all_ training points
2. then solving the original problem with $w = \sum \alpha_i y_i x_i$ & $b = y_k - \sum \alpha_i y_i x_i^T x_k$

### svm testing

at this point, we know that any $a_i$ that is not zero is a support vector & that we should have the smallest number of support vectors possible. because of this, classifying a new datapoint (for validation or for prediction purposes) is simply a function of that new datapoint compared to _only_ the support vectors:

$$
f(x) = w^T x + b = \sum \alpha_i y_i x_i^T x + b
$$

## soft margin svm

when dealing w/ a training set that is not entirely linearly separable (e.g.
has some noisy datapoints that resist classification), a _slack variable_
$\Im_i can be used to account for this. by allowing for some missclafication
(w/ limits to minimize errors), we can then make an otherwise unseparable
dataset be linearly separable.

this slack variable is defined as:

$$
\Im_i = \text{max} \{0,1 - y_i(w^T x_i + b)\}
$$

introducing this slack margin to our SVM from before gives us the following:

$$
\begin{aligned}
    && \forall &(x_i,y_i) \in T \text{ find $w$ \& $b$ such that} \\
(7) && Q&(w, \Im_i) = \frac{1}{2} w^T w + C \sum_i \Im_i \text{ is minimized} \\
    && \text{a}&\text{nd } \Im_i >= 0 \land y_i(w^T x_i + b) >= 1 \text{ holds}
\end{aligned}
$$

but what's $C$ in this definition of $Q$?

we use the _margin parameter_ $C$ to control how many datapoints are allowed to
be misclassified, where a smaller value for $C$ prioritizes a wider margin at
the expense of more misclassifications while a larger $C$ means less

finally, using the same process in finding our Lagrangian duality-based
solution to a hard margin SVM from above, we can optimize the soft margin

$$
\begin{aligned}
    && \forall &i \in [1,n] \text{ find } \alpha_i \text{ such that} \\
(8) && Q&(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_j \text{ is maximized} \\
    && \text{a}&\text{nd } 0 <= \alpha_i <= C \land \sum_i \alpha_i y_i \equiv 0 \text{ holds}
\end{aligned}
$$

which gives us the following solutions for $w$ & $b$:

$$
\begin{aligned}
        && w &= \sum \alpha_i y_i x_i \\
\land   && b &= y_k(1 - \Im_i) - \sum \alpha_i y_i x_i^T x_k \\
\forall && a_k &\in {a_k | a_1...a_n, a_k > 0}
\end{aligned}
$$

## non-linear svm

what about data that resists linear separation? for example, how to separate
the $x$s from the $o$s in either of the following:

```
|
| o o o x x x x o o o
|
+------------------------
```

```
| o       o
|
|  o     o
|   x   x
|     x
+--------------
```

w/ a linear function? impossible. but recall we began referring to our
separator as a _hyperplane_ above. what if we reimagine the data as
3-dimensional instead of 2-dimensional, then separate it w/ a _plane_ insteadof a line.

entre the _kernal trick_.

## kernal method

by mapping input vectors to a higher-dimensional space, we can make the
maximum-margin separator plane much easier to compute. so long as we're careful
about how we do that mapping, we can achieve this separable dataset in
higher-order space w/out paying an increased price when computing inner
products between datapoints. the _kernal trick_ is how we do this.

- the problem: matrix multiplication between higher-dimensional matrices is vvv
  expensive.
- the solution: find some other function that is equivalent to the matrix
  multiplication, but is much easier to solve.

in the end, we'll have a decision function that look like this:

$$
\begin{aligned}
   f(x) &= w^T \varphi (x) + b, \textit{where} \\
\varphi(x) &= \langle \varphi (x_1),...,\varphi (x_n) \rangle &&
\textit{nonlinear mapping function}\\
\end{aligned}
$$

but when computing the multiplication $\varphi (x_i) \cdot \varphi (x_j)$ while
optimizing $w^T$ as we have above, we'll instead use our _Kernal
function_ $K(x_i,x_j) \equiv \varphi (x_i)^T \varphi(x_j)$.

### kernal functions

Below are some known Kernal functions for some given features space types. I've
elided the proofs here for brevity/sanity's sake.

$$
\begin{aligned}
\textit{linear} && K(x_i,x_j) &= x_i \cdot x_j \\
\\
\textit{polynomial} && K(x_i,x_j) &= (1 + x_i \cdot x_j)^p \\
\\
\textit{gaussian} && K(x_i,x_j) &= e^{\Bigg\{\frac{-\big\Vert x_i - x_j \big\Vert^2}{\normalsize 2 \alpha^2}\Bigg\}} \\
\\
\textit{laplace} && K(x_i,x_j) &= e^{\Bigg\{\frac{-\big\Vert x_i - x_j \big\Vert}{\normalsize 2 \alpha^2}\Bigg\}} \\
\\
\textit{sigmoid} && K(x_i,x_j) &= tanh(\mathcal K (x_i \cdot x_j) + \delta) \\
\end{aligned}
$$

there are also two types of kernal functions: _local_ & _global_.

1. local
   - only affects nearby datapoints; not the entire domain
   - has higher learning ability, but not as good at generalizing
   - used only when there's no prior knowledge about dataset
2. global
   - affects _all_ datapoints across entire domain
   - better at generalizing

#### validating kernal functions

you can also create your own kernal function, but you need to be able to
validate that your function actually is a proper kernal before you use it.

$$
\begin{aligned}
     && \because
        && K &:= \begin{bmatrix}
                   K(x_1,x_1) & \cdots     & K(x_1,x_n) \\
                   \vdots     & K(x_i,x_j) & \vdots     \\
                   K(x_n,x_1) & \cdots     & K(x_n,x_n) \\
                 \end{bmatrix} &&
    \textit{matrix of kernal fn on all pairs} \\
\\
(9)  && && K &\equiv K^T &&
    \textit{K is symmetric} \\
(10) && && c^T K c &>= 0, \text{ where } c \in \R &&
    \textit{K is positive semi-definite} \\
\\
     && \therefore
        && (9) &\land (10) \iff K: \varphi \times \varphi \text{ is a valid kernal} &&
    \textit{Mercer's theorem}\\
\end{aligned}
$$

#### kernal closures

new kernals can be constructed easily using existing ones. some common properties used to combine kernals include:

$$
\begin{aligned}
\because && K_1(\cdot,\cdot) &\land K_2(\cdot,\cdot) \text{ are kernals}, \textit{ and} \\
         && c_1 >= 0 &\land c_2 >= 0 \\
\\
\therefore && \text{the foll} & \text{owing are valid kernals:} \\
           && K &= c_1 K_1 \\
           && K &= c_1 K_1 + c_2 K_2 \\
           && K &= K_1 K_2 \\
\end{aligned}
$$

### non-linear svm definition & solution

we can define a non-linear svm using a kernal function $K$ by substituting it in $Q$ like so:

$$
\begin{aligned}
    && \forall &i \in [1,n] \text{ find } \alpha_i \text{ such that} \\
(8) && Q&(\alpha) = \sum_i \alpha_i - \frac{1}{2} \sum_i \sum_j \alpha_i \alpha_j y_i y_j K(x_i,x_j) \text{ is maximized} \\
    && \text{a}&\text{nd } \alpha_i >= 0 \land \sum_i \alpha_i y_i \equiv 0 \text{ holds}
\end{aligned}
$$

which gives us the following classifying function, $f(x)$:

$$
f(x) = \sum \alpha_i y_i K(x_i,x_j) + b
$$

## multi-class svm

svm can also be used to perform classification w/ a non-binary set of classes. this is done by breaking the main problem into smaller, multiple, binary classification problems, then combining the results.

there are two approaches:

1. one-vs-all

   done by creating $k$ classifier function (one for each class) that separates
   a single class from the rest by treating any classs that is not the targeted
   class as the same class.

   this approach requires $k$ classifier functions. this approach is typically
   cheaper than _one-vs-one_ because it requires less classifiers; however, it
   can be less accurate. in practice, the loss of accuracy is rarely
   significant enough to justify the higher cost of _one-vs-one_.

2. one-vs-one

   done by creating classifiers for each class that separates it from each of
   the other classes, then choosing the class that the data best fits after
   checking each of the classifiers.

   this approach requires $k(k - 1) / 2$ classifier functions, thus it can be
   more computationally expensive. however, it is often more accurate.
