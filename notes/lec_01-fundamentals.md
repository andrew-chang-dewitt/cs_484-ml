---
title: "ML: fundamentals"
description: "Lecture notes on ML fundamentals."
keywords:
  - "fundamentals"
  - "machine learning"
  - "lecture notes"
  - "computer science"
  - "cs 484"
  - "illinois tech"
meta:
  byline: Andrew Chang-DeWitt
  published: "2025-08-20T00:00-07:00"
  updated: "2025-08-27T00:00-06:00"
---

what is machine learning? defined as difference between trad & ml programs is:

- trad: defined algo used to compute output from input
  ```
  input + algo ==[ trad prg ]=> output
  ```
- ml: model/logic learned from data, then applied to new inputs to gen new
  outputs
  ```
  input + expected output ==[ ml training ]=> model + input ==[ml inference]=> output
  ```

_\*\*def: machine learning_&mdash;the study of algos that:

- improve their performance (`P`)
- at some task (`T`)
- with experience (`E`)

## why use ml?

- models that can incorporate huge amounts of data
- human expertiese does not exist
- humans can't explain our expertiese (e.g. can't make an algo to replicate it)

common applications:

- recognizing patterns
  - facial expressions/identies
  - written/spoken words
- generating patterns
- recognizing anomalies
- prediction

> [!NOTE]
>
> ML != AI

## examples

```
T := playing checkers
P := percentage of games won against arbitrary opponent
E := playing practice games against itself

T := recognizing hand-written words
P := percentage of words correctly classified
E := db of human-labeled images of handwritten words
```

## The Task, `T`

tasks usually describe in terms of how ml _should_ process ex:

```
x is in set Real^n, where each x_i is a _feature_
```

main two types of tasks:

- **classification**: learn `f: Real^n -> {1,..,k}`
  - `y = f(x)` assigns input to category w/ output `y`
  - ex: object recognition
- **regression**: learn `f: Real^n -> Real`
  - `y = f(x)` assigns predicted value derived from input value `x`
  - ex: weather prediction

there are some others that are less common:

- transcription: unstructured repr transformed to discrete text
- machine translation: sequence to sequence
- synthesis & sampling: gen ex that are similar to those in training data

these three can be thought of as sub examples of the main two, but they are still distinct enough to be worth mentioning

## The Experience, `E`

types:

- supervised learning:
  - experience is a _labeled_ dataset
  - datapoint has label (or target)
  - has _lots_ of information

  ex: given set of ordered pairs, `(x_i,y_i)`, learn a fn, `f` to predict `y`
  given `x`

- unsupervised learning:
  - _un_-labeled
  - clustering, prob. distribution, denoising, etc.
  - has _some_ information

  ex: raw input data for learning -> learning must interpret datapoints itself
  & begin to notice patterns w/out guidance -> gives model which given input
  gives output

- reinforcement learning:
  - exp is interaction w/ an environment
  - has _almost no_ information

  ex: agent has no info, receives information (state, `s`) from env -> takes an
  action, `a` -> env then gives reward, `r` of value according to quality of action

  ```
   /--->[ agent ]--\
   |     ^         |
   |     |         |
  {s}   {r}       {a}
   |     |         |
   |     |         |
   \----[  env  ]<-/
  ```

## The Performance, `P`

many ways of determining performance of a model:

- numeric/quantitative measures:
  - **accuracy**: typ as a proportion of examples for which model produces correct output
  - **error rate**: complement of _accuracy_
  - **loss fn.**: a fn that quantifies diff btwn predicted outputs & actual target values
- qualitative measures:
  - **generalization**: ability to perform well on prev. unobserved data

## No Free Lunch Theorem

> without having substantive information about the modeling problem, there is no single model that will always do better than any other model

originated in statistics, but describes model perf in ML vv well as well. goal is not to seek a universal learning algo or absolute best algo in ML. instead, algos typ. specialize by aligning w/ true nature of problem.

## important concepts used in ML

- datasets & features

### datasets & features

- _**def**: dataset_&mdash;set of data grouped into a collection; rows rep numer of data points & cols rep _features_ of dataset
- _**def**: features_&mdash;information about each datapoint (columns in table)

feature scaling:

- an important part of normalizing/cleaning data to prepare for use in ml application
- scale the data for a feature to a fixed range (e.g. [0,1])
- done by formulas:
  - normalization: rescale data 'x' using mean 'm' and standard dev 'd' of data
    ```
    x_norm = (x - m)/d
    ```
  - min-max scaling:
    ```
    x_minmax = (x - x_min)/(x_max - x_min)
    ```

### types of data

- quantitative (numerical)
  - continuous
  - discrete
- qualitative (categorical)
  - ordinal
  - nominal

## splitting the dataset

diff types of data sets can serve diff purposes. for any algo/model, a data set will _always_ be split into at least two, and often three, sets:

1. training set: used to train model
2. test set: evaluate if model works & how well
3. validation set: (optional) used between training & test to determine how training is going & then fine tune training process to improve training

important assumptions abt IID (independent & identically distributed) datasets:

- examples in each dataset are independent of one another
- training & testing sets are identically distributed, i.e. drawn from the same probability distribution as each other

```
+ - - - - - - - - - - - - - - - - - - - - - - +

|    /-> [    train on training set     ]     |
     |                |
|    |                v                       |
     |   [  evaluate on validation set  ]
|    |                |                       |
     |                v
|    \-- { tweak model per eval results }     |

+ - - - - - - - - - - - - - - - - - - - - - - +
                      |
                      v
  { pick model w/ best validation set perf }
                      |
                      v
        [ confirm results on test set ]
```

## under- & over-fitting

_**def:** underfitting_&mdash;occurs when a model is too simple to capture
the underlying patterns in the training data, leading to poor perf on
training & test sets

- means model is not complex enough to learn relationships w/in data
- id by looking @ error rate: if rate is holding or increasing, then might be
  underfitting

_**def:** overfitting_&mdash;occurs when model learns the data too well,
including noise & random fluctuations, leading to poor perf on new, unseen
data

- means model is too complex; finds "relationships" between unrelated
  datapoints that aren't actually there; works more by having memorized
  training data instead of actually finding true relationships
- only performs well on training set; when presented w/ new data it will be
  unable to generalize relationships from training data & will have high error

```
< train error > --{big}--> [ underfitting ]
      |
   {small}
      |
      v
< valid error > --{big}--> [ overfitting ]
      |
   {small}
      |
      v
< test error  > --{big}--> [ train & test sets are too different ]
      |
   {small}
      |
      +------------------> [ good model! ]
```

### overfitting: a formal exploration

_**def:** hypothesis_&mdash;(in machine learning) the model's presumption regarding the connection between the input features & the output

$$
\begin{aligned}
\because   && &h := \text{some hypothesis}, \\
           && &err_{\text{train}}(h), \textit{and} && \textit{error rate over training data} \\
           && &err_{\text{true}}(h)                && \textit{true error rate over all data} \\
\\
           && &\exists\space h' \textit{s.t.}     && \textit{alternative hypothesis exists where} \\
           && &\quad err_{\text{train}}(h) < err_{\text{train}}(h') \\
           && &\quad err_{\text{true}}(h) \space> err_{\text{true}}(h') \\
\\
\therefore && &\text{Overfit}(h)          && \textit{hypothesis $h$ overfits training data}
\end{aligned}
$$

### resolutions

- for underfitting:
  - increase model complexity (train longer/on larger dataset)
  - choose a diff. ml algo
  - combine models to create better outputs (a.k.a. _ensemble methods_)
  - create new model features from existing ones that may be more relevant to
    the learning task to compensate for errors around those types of features
    in the model (a.k.a. _feature engineering_)

  - _cross-validation_: evaluate ml models by training several ml models on
    subsets of the input data, then evaluate them on another subset
  - _regularization_: add a penalty term to the loss function, discouraging the
    model from assigning too much importance to individual features&mdash;typ.
    the most popular technique (so much so that it's usually embedded in code
  - decrease model complexity (train for less time/on smaller dataset)
    - stop early when a specific metric has been achieved/stopped improving
    - _early stopping_: end training when a specific metric has been
      achieved/stopped improving
  - _bagging_: learn multiple models in parallel & apply a majority votingprocedure to choose the final candidate model

#### cross-validation

most popular technique called $k-fold$ cross validation

- divide data into $k$ folds
- train on $k - 1$ folds, use the $k^{\textit{th}}$ fold to measure error
- repeat $k$ times, using _average error_ to measure generalization accuracy
- statistically valid & gives good accuracy estimates

also popular: _leave-one-out_ cross-validation (LOOCV):

- $k-fold$ cv w/ $k = N$, where $N$ is number of data points
- v accurate, v expensive, requires build $N$ different models

## parametric learning

any algo that assumes there's a direct, functional dependency btwn the model input & the model output. includes:

- logistic regression
- linear regression
- perceptron
- naive bayes
- neural network

benefits are:

- easier to understand/interpret results
- can be much faster to train/learn
- does not require as much training data
- can work well even if it doesn't fit data perfectly

downside: by choosing a functional form before even beginning training, then model is highly constrained to fit the form of the function chosen

## non-parametric learning

complement to parametric learning; makes no assumption about functional dependency or shape of said function.

examples:

- svm
- k-nearest-neighbor
- k-means
- decision tree

benefits:

- capable of fitting a large number of functional shapes
- no assumptions about underlying function
- can result in higher perf for prediction

downsides:

- requires a lot more training data
- takes longer to train
- prone to overfitting

## classification

another modeling technique for predictive modeling where model predicts categorical output from input data.

types:

- binary classification
- multi-class
- multi-label
- imbalanced

uses:

- medical diagnosises:
  - features: age, gender, history, symptoms, test results
  - label: disease/condition/diagnosis
- spam detection:
  - features: sender-domain, length, images, keywords
  - label: spam/not-spam
- credit card fraud detection:
  - features: user, location, item, price
  - label: fraud/not-fraud

## wrap-up

a tree of algorithms by types:

```
                                         [ml]
                                          |
                  +-----------------------+----------+-----------------+
                  |                                  |                 |
                  v                                  v                 v
            [supervised lrn.]                  [unsup. lrn.]  [reinforcement lrn.]
                  |                                  |                 |
         +--------+------------+                     |                 |
         |                     |                     |                 |
         v                     v                     v                 v
  [classification]       [regression]          [clustering]      [decision making]
         |                     |                     |                 |
         v                     v                     v                 v
+------------------+  +------------------+  +-----------------+  +---------------+
| - naive bayes    |  | - linear         |  | - k-means       |  | - q-learning  |
|   clasifier      |  |   regression     |  |   clustering    |  | - r-learning  |
| - decision trees |  | - neural network |  | - mean-shift    |  | - td-learning |
| - support vector |  |   regression     |  |   clustering    |  +---------------+
|   machines       |  | - support vector |  | - dbscan        |
| - random forest  |  |   regression     |  |   clustering    |
| - k-nearest      |  | - decision tree  |  | - agglomerative |
|   neighbors      |  |   regression     |  |   hierarchical  |
+------------------+  | - laso regr.     |  |   clustering    |
                      | - ridge regr.    |  | - gaussian      |
                      +------------------+  |   mixture       |
                                            +-----------------+
```
