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
    \textit{prediction is correct, continue} \\
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
