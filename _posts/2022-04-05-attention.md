---
title: 'Attention'
date: 2022-04-12
permalink: /posts/2022/04/attention/
tags:
  - nlp
---
That term you keep seeing in papers without really knowing what it meant.

## Introduction
Though attention as a concept was introduced in the 1990s, ["Attention is All You Need"](https://arxiv.org/abs/1706.03762) revolutionized its usage for machine learning models, particularly in natural language processing. I will try to legibly explain the mechanism of attention and its role in the Transformer architecture.

### ELI5: What is attention?
Say we have some sequence, as follows:

"Jane is hungry, and she's really craving pasta from Italy." 

_(Figure to come)_

Intuitively, we can see that there are words that are more closely connected to other words than not. For instance, "hungry" and "Jane" are closely linked because "hungry" is an attribute of how Jane feels. "Jane" and "she's" are also closely related, since "she" is referring to "Jane". So is "pasta" and "Italy", since the pasta is from Italy. Thus, there are certain pieces of this sentence that provide information more relevant to "Jane", and there are other pieces of the sentence that provide information more relevant to "pasta".

Learning which pieces of the sequence are more related to a particular word (like "Jane" or "pasta") would be really useful. It would allow us to play down the effects of irrelevant parts of the sequence and boost the impact of the more relevant parts of the sequence. **Attention** is a mechanism that allows ML models to replicate this same effect in its computations. (For now, we will focus on **self-attention** -- attention within a single sequence).

More formally, given a particular piece of the sequence, attention spits out a linear combination across the information provided by _every other_ piece of the sequence. The weights of this linear combination reduce the impact of the "unimportant" bits of the sequence and increase the impact of the "important" bits. Thus, each piece of the sequence gets a custom linear combination which can be thought of as "selective summarization" of the information in the sequence.

### The key components of attention
There are three key players in the attention mechanism: 

1. **Query**: $$q \in \mathbb{R}^{d_q}$$
2. **Keys**: $${k_i} \in \mathbb{R}^{d_k}$$
3. **Values**: $$\{v_i\} \in \mathbb{R}^{d_v}$$

Broadly, the query asks the model, "I'm looking at this particular token right now. What parts of the sequence are relevant?". Using the query, the keys will be used to compute the **attention scores**, which will determine the weights of the linear combination mentioned above. Then, these weights are used along with the values to calculate the linear combination.
 
