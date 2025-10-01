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
       && P \land Q &\models {(x_1,y_1),(x_2,y_2),...,(x_n,y_n)} && \textit{data are sets of pairs}
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
2. given some example $x$, predict $Positive(y) \iff w_1 x >= 0$
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

| i   | T   | $w^T$            | $x_i$        | $>= 0 \iff +$ | correct? |
| --- | --- | ---------------- | ------------ | ------------- | -------- |
|     | 1   | $[0\space0]^T$   |              |               |          |
| 1   | 1   | $[0\space0]^T$   | $[1\space2]$ | +             | y        |
| 2   | 1   | $[0\space0]^T$   | $[2\space3]$ | +             | y        |
| 3   | 1   | $[0\space0]^T$   | $[2\space1]$ | +             | n        |
| 3   | 2   | $w^{T-1} - x_i$  |              |               |          |
| 4   | 2   | $[-2\space-1]^T$ | $[3\space0]$ | -             | y        |
