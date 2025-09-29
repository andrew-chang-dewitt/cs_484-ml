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
  published: "2025-08-27T00:00-06:00"
  updated: "2025-09-23T00:00-06:00"
---

:::aside{.right-aside.after#toc}

## contents

- linear separator

:::

## linear separator

_**def:** linearly separable_&mdash;a characteristic of two datasets that is true when all values in one set can be differentiated from all values in the other by drawing a vector between the two

More formally, for two datasets of $n$ pairs, $P$ & $Q$, they are _linearly separable _ if and only if there is a set of $n + 1$ real numbers that satisfies the following:

$$
\begin{aligned}
    && \because
       && P \land Q &\models {(x_1,y_1),(x_2,y_2),...,(x_n,y_n)} && \textit{data are sets of pairs}
    && && \exists W &= {w_1,w_2,...,w_n,k} \\
(2) && && wx &< k \forall w \in W[1,n], x \in P \\
(3) && && wx &> k \forall w \in W[1,n], x \in Q \\
    && \therefore
       && (2) \land (3) \iff LinSep(P,Q) \\
\end{aligned}
$$

_binary classification_&mdash; the act of classifying values into one of two values&mdash; can be modeled using linear separation. by placing values in feature space, if a linear separator can be found, then it can be used as a hypothesis to divide the entire feature space, thus useful for prediction and other ml tasks.
