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
  published: "2025-09-23T00:00-07:00"
  updated: "2025-08-27T00:00-06:00"
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
    positive or negative)
  - each edge is associated w/ one value of the feature at the node from which
    the edge emmenates

used to classify examples as fitting, or not fitting, a concept; relies on
supervised learning.
