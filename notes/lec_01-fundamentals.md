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
  updated: "2025-08-22T00:00-06:00"
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

## examples

```
T := playing checkers
P := percentage of games won against arbitrary opponent
E := playing practice games against itself

T := recognizing hand-written words
P := percentage of words correctly classified
E := db of human-labeled images of handwritten words
```

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

## important concepts used in AI

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
  - clustering, prob. distributino, denoising, etc.
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
