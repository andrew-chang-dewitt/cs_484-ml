---
title: "ML: decision trees"
description: "Introducing decision trees & related algorithms (ID3 etc.)."
keywords:
  - "decision trees"
  - "id3"
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

> [!ASIDE]
>
> ## contents
>
> - definitions
> - construction
>   - top-down induction (ID3)
> - error reduction

## definitions

_**def:** decision tree_&mdash;

- from slides:
  > method for **learning discrete-valued target functions** in which the
  > function to be learned is represented by a decision tree
- is tree where:
  - each non-leaf node is associated w/ an attribute/feature
  - each leaf node is associated w/ a classification (fitting or not fitting;
    positive or negative)&mdash; this can be generalized to more than 2 classes; but binary trees are typical
  - each edge is associated w/ one value of the feature at the node from which
    the edge emmenates

used to classify examples as fitting, or not fitting, a concept; relies on
supervised learning.

an example shown as both a tree & a table:

```
           [color]
           /  |  \
      green  red  blue
         /    |    \
    [size]  < + >  [shape]
     /  \           /   \
  big    small square   round
   /      \       /       \
< - >    < + > [size]    < + >
                /  \
             big    small
              /      \
           < - >    < + >
```

| color | size  | shape  | class |
| ----- | ----- | ------ | ----- |
| red   | x     | x      | +     |
| green | big   | x      | -     |
| green | small | x      | +     |
| blue  | x     | round  | +     |
| blue  | big   | square | -     |
| blue  | small | square | +     |

and some examples as function definitions:

$f(x) = x_2 \cap x_5$:

```
      [x_2]
     1/   \0
   [x_5] < F >
  1/   \0
< T > < F >
```

$f(x) = x_2 \cup x_5$:

```
      [x_2]
     1/   \0
   < T > [x_5]
        1/   \0
      < T > < F >
```

## construction

going forward, this lecture uses the following example training data
set:

| day | outlook  | temp. | humidity | wind   | play tennis |
| --- | -------- | ----- | -------- | ------ | ----------- |
| d1  | sunny    | hot   | high     | weak   | no          |
| d2  | sunny    | hot   | high     | strong | no          |
| d3  | overcast | hot   | high     | weak   | yes         |
| d4  | rain     | mild  | high     | weak   | yes         |
| d5  | rain     | cool  | normal   | weak   | yes         |
| d6  | rain     | cool  | normal   | strong | no          |
| d7  | overcast | cool  | normal   | strong | yes         |
| d8  | sunny    | mild  | high     | weak   | no          |
| d9  | sunny    | cool  | normal   | weak   | yes         |
| d10 | rain     | mild  | normal   | weak   | yes         |
| d11 | sunny    | mild  | normal   | strong | yes         |
| d12 | overcast | mild  | high     | strong | yes         |
| d13 | overcast | hot   | normal   | weak   | yes         |
| d14 | rain     | mild  | high     | strong | no          |

using _supervised classification_ we will use an algorithm to construct
a decision tree, doing the following:

**input:** _labeled training data_ $(x_i, y_i)$ of unknown target
function, $f$

- examples described by their values
- unknown target fn., $f: X -> Y$
  - e.g. $Y = {0,1} label space
  - e.g. $1$ if we play tennis, else $0$

**output:** _hypothesis_ $h \in H$ that best _approximates_ target fn.
$f$

- set of function hypotheses $H = {h|f:X->Y}$
- each _hypothesis_ $h$ is a _decision tree_

but what makes a good hypothesis?

- short tree height, while still fitting the data
- a complex tree (large height) may be more likely to fit that data,
  but also more likely to be a statistical coincidence; thus Occam's
  razor leads us to seek the smallest tree that fits since it is \*\*less

how to actually build the tree?

### top-down induction (ID3)

the _Iterative Dichotomiser 3_ algorithm (ID3):

- invented by Ross Quinlan, used to generate decision tree from a
  dataset
- greedy, using top-down approach

algorithm: grow tree from root to leaves by repeatedly _replacing
existing leaf_ w/ an _internal node_

```
S <- training dataset

main loop:
    A <- attr in S   \\ pick _best_ attr to split at root
    for each a in A: \\ recurse on children that are _impure_[1]
        create new descendent of node
    sort training examples to leaf nodes
```

- [1]: a node that contains both Yes & No (or not all one class)

an example:

training dataset

| $x_1$ | $x_2$ | $y$ |
| ----- | ----- | --- |
| 0     | 0     | 0   |
| 0     | 1     | 0   |
| 1     | 0     | 0   |
| 1     | 1     | 1   |

tree is constructed in iterations

given $x_1$ as root, iteration 0 starts as:

create tree of root & 1 edge for each attr

```
      [x_1]
     0/   \1
```

iteration 1:

- check outputs for all values matching each branch from $x_1$
  - if pure, replace w/ that output value
  - otherwise recur

```
      [x_1]
     0/   \1 \\ 0 is pure (all y for x_1 === 0 are 0)
   < F >  ...
```

iteration 2:

- check outputs for all values matching attr $x_2$

```
      [x_1]
     0/   \1
   < F > [x_2]
        0/   \1
      < F > < T > // both are pure, done!
```

#### find best attr: information gain

_**def:**_ statistical measure that tells how much new information is
added by making some choice

we'll use a measure called _entropy_, which measures _impurity_, to
help us quantify the information contained by an attributehelp us quantify the information contained by an attribute

> [!ASIDE]
>
> Entropy, `E`, measures impurity of `S`:
>
> $$
> \begin{aligned}
> ∵ &&    S &\gets \text{sample of training examples}, \\
>   &&  p_+ &\gets \text{proportion of positive examples in $S$}, \\
>   &&  p_- &\gets \text{proportion of negative examples in $S$} \\
> \\
> ∴ && E(S) &= - p_+ log_2(p_+) - p_- log_2(p_-)
> \end{aligned}
> $$
>
> - if _all negative_ **or** _all positive_ then $E = 0$
> - if equally divided, then $E = 1$
> - if 14 examples w/ 9 positive & 5 negative, then $E = 0.94$
>
> when discussing a non-Boolean attribute, entropy is measured a little differently:
>
> $$
> \begin{aligned}
> ∴ && E(S) &= \sum_{i \in Y} - p_i \log_2 p_i, \textit{where} \\
>   &&      & p \gets \text{proportion of $S$ belonging in class $i$}
> \end{aligned}
> $$

the _information gain_ of an attribute, `A`, is the _expected reduction_ in entropy caused by partitioning the data sample `S` on `A`:

$$
\begin{aligned}
(1) && Gain(S,A) = E(S) - \sum_{v \in values(A)} \frac{|S_v|}{|S|} E(S_v)
\end{aligned}
$$

##### example

let's calc gain for our sample dataset's attrs:

$$
\begin{aligned}
    && \because
       && S &\gets \{...\}, \\
(2) && && A &\gets \{O, T, H, W\} \\
(3) && && &\quad\textit{where O is Outlook, T is Temperature,} \\
    && && &\quad\quad\quad\quad\textit{H is Humidity, \& W is Wind}
\end{aligned}
$$

**humidity:**

$$
\begin{aligned}
(4) && \because
       &&             V_H &\gets values(H) \\
    && &&                 &= \{\textit{high}, \textit{norm}\} \\
\\
    && &&    S_{H_h} &= \{\text{No}, \text{No}, \text{Yes}, \text{Yes}, \text{No}, \text{Yes}, \text{No}\} \\
    && &&        p_t &= \frac{3}{7} \\
    && &&        p_f &= \frac{4}{7} \\
\\
    && && E(S_{H_h}) &= -p_t \log_2(p_t) - p_f \log_2(p_f) \\
    && &&                 &= -(3/7) \log_2(3/7) - 4/7 \log_2(4/7) \\
    && &&                 &\approx 0.5238 + 0.4613 \\
(5) && && E(S_{H_h}) &\approx 0.985 \\
\\
    && &&    S_{H_n} &= \{\text{Yes}, \text{No}, \text{Yes}, \text{Yes}, \text{Yes}, \text{Yes}, \text{Yes}\} \\
    && &&        p_t &= \frac{6}{7} \\
    && &&        p_f &= \frac{1}{7} \\
\\
    && && E(S_{H_n}) &= -p_t \log_2(p_t) - p_f \log_2(p_f) \\
    && &&                 &= -(6/7) \log_2(6/7) - 1/7 \log_2(1/7) \\
    && &&                 &\approx 0.1906 + 0.4011 \\
(6) && && E(S_{H_n}) &\approx 0.592 \\
\\
    && &&  Gain(S,H) &= E(S) - \frac{|S_{H_h}|}{|S|} E(S_{H_h}) - \frac{|S_{H_n}|}{|S|} E(S_{H_n}) \\
    && &&            &= 0.940 - \frac{7}{14} 0.985 - \frac{7}{14} 0.592 \\
    && \therefore
       &&  Gain(S,H) &= 0.151 \quad\blacksquare \\
\end{aligned}
$$

**wind:**

$$
\begin{aligned}
    && \because
       && S_{W_w} &= \{ \text{No}, \text{Yes}, \text{Yes}, \text{Yes}, \text{No}, \text{Yes}, \text{Yes}, \text{Yes} \}, \\
    && && S_{W_s} &= \{ \text{No}, \text{Yes}, \text{Yes}, \text{Yes}, \text{No}, \text{Yes}, \text{Yes}, \text{Yes} \} \\
\\
    && && Gain(S,W) &= E(S) - \frac{|S_{W_w}|}{|S|} E(S_{W_w}) - \frac{|S_{W_s}|}{|S|} E(S_{W_s}) \\
    && &&            &= 0.940 - \frac{8}{14} 0.811 - \frac{6}{14} 1.0 \\
    && && Gain(S,W) &= 0.048 \quad\blacksquare
\end{aligned}
$$

**outlook:**

$$
\begin{aligned}
    && \because
       && S_{O_s} &= \{\text{No}, \text{No}, \text{No}, \text{Yes}, \text{Yes}\}, \\
    && && S_{O_o} &= \{\text{Yes}, \text{Yes}, \text{Yes}, \text{Yes}\}, \\
    && && S_{O_r} &= \{\text{Yes}, \text{Yes}, \text{No}, \text{Yes}, \text{No}\} \\
\\
    && && Gain(S,O) &= E(S) - \frac{|S_{O_s}|}{|S|} E(S_{O_s}) - \frac{|S_{O_o}|}{|S|} E(S_{O_o}) - \frac{|S_{O_r}|}{|S|} E(S_{O_r}) \\
    && &&            &= 0.940 - \frac{5}{14} 0.970 - \frac{4}{14} 0.000 - \frac{5}{14} 0.970 \\
    && && Gain(S,O) &= 0.248 \quad\blacksquare
\end{aligned}
$$

finally, we can determine which of these attributes is the best one to partition on from the root by selecting the one with the highest _information gain_:

$$
\begin{aligned}
\because
&& Gain(S,H) &= 0.151, \\
&& Gain(S,W) &= 0.048, \\
&& Gain(S,O) &= 0.248 \\
\\
&& Gain(S,O) &> Gain(S,H) > Gain(S,W) \\
\\
\therefore
&& \text{best is } O & \quad\blacksquare
\end{aligned}
$$

> [!TODO]
>
> finish example? ...

#### props of ID3

- heuristic search through _space_ of decision trees
- stops at smallest acceptable tree
- can overfit, can help to _prune_ the found tree sometimes

##### overfitting

occurs when data is noisy and because ID3 is _not guaranteed_ to output a small hypothesis even if one exists

to avoid:

- stop growing when data split not statistically significant
- acquire more training data
- remove irrelevant attributes
- grow full tree, then post-prune
