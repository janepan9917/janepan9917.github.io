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

[Figures forthcoming. Also, not complete yet.]

## ELI5: What is attention?
Say we have some sequence as follows:

> "Jane is hungry, and she's really craving pasta from Italy." 

Intuitively, we can see that there are words that are more closely connected to other words than not. For instance, "hungry" and "Jane" are closely linked because "hungry" is an attribute of how Jane feels. "Jane" and "she's" are also closely related, since "she" is referring to "Jane". So is "pasta" and "Italy", since the pasta is from Italy. Thus, there are certain pieces of this sentence that provide information more relevant to "Jane", and there are other pieces of the sentence that provide information more relevant to "pasta".

Learning which pieces of the sequence are more related to a particular word (like "Jane" or "pasta") would be really useful. It would allow us to play down the effects of irrelevant parts of the sequence and boost the impact of the more relevant parts of the sequence. **Attention** is a mechanism that allows ML models to replicate this same effect in its computations. (For now, we will focus on **self-attention** -- attention within a single sequence).

More formally, given a particular piece of the sequence, attention spits out a linear combination of the information provided by _every other_ piece of the sequence. The weights of this linear combination reduce the impact of the "unimportant" bits of the sequence and increase the impact of the "important" bits. Thus, each piece of the sequence gets a custom linear combination which can be thought of as "selective summarization" of the information in the sequence.

## The key components of attention
We have $$n$$ input vectors: $$\{x_1, ..., x_n\} \in \mathbb{R_{d_x}}$$. Our self-attention layer will map these vectors to $$n$$ output vectors: $$\{y_1, ..., y_n\} \in \mathbb{R_{d_y}}$$. But let's forget about these for now. 

### Computing attention, simply
There are three key players in the attention mechanism: 
1. **Query**: $$q_j \in \mathbb{R}^{d_q}$$
2. **Keys**: $$\{k_1, ..., k_n\} \in \mathbb{R}^{d_k}$$
3. **Values**: $$\{v_1, ..., v_n\} \in \mathbb{R}^{d_v}$$

All three are **representations** of tokens in the sequence.[^note] The query $q_j$ is computed for the $j$th token, which that we are currently processing. On the other hand, we have $n$ keys and $n$ values -- one for each value in the part of the sequence. 

[^note]: We will see later how these representations are computed. 

Broadly, the query asks the model, "Here's a representation of the token I'm looking at right now. What parts of the sequence are similar to this?"[^corr] (Since the query gets to "pick" which values get higher weights in the weighted sum, we say the query **attends** to the values.) Using the keys and the queries, we then compute the **attention scores**, which will determine the weights of the linear combination mentioned above. Finally, we calculate a linear combination of the values using weights computed in step 2.
 
[^corr]: More precisely, the similarity is calculated between a representation of the token we're looking at (query) and a representation of the other tokens in the sequences (keys). 

Here's the three steps to the attention function:
1. Calculate the attention scores: $$s_j = g(\{k_1, ..., k_n\}, q_j)$$
2. Make the attention scores a probability distribution: $$\alpha_j = \text{softmax}(s_j)$$
3. Take the weighted sum: $$a_j = \sum^{n}_{i=1} \alpha_{j, i} v_i$$

$$g$$ can be thought of as a similarity function; it uses the key and query to compute the raw attention scores $s_j$, which is a similarity score between the keys and the query. Then, we normalize $s_j$ using softmax normalization so that the weights in our linear combination will sum to 1.[^align] Finally, we compute the weighted sum across the values. The weighted sum $a_j$ is called the **context vector** for the $j$th token in the sequence.

[^align]: Sometimes, the term "alignment score" is used synonymously with "attention score", and sometimes the alignment score is assumed to be normalized.


### One step deeper
This is not too bad so far. But how exactly do we pick the queries, keys, and values? It would be very difficult to learn a particular query, key, and value for every possible input vector. Instead, we'll learn weight matrices that we can multiply to any arbitrary input vector. This way, we can compute a query or key or value for any arbitary input vector that could be thrown at our model. 

More formally, given some input vector $$x_i$$:

1. **Query**: $$q_i = W_Qx_i$$, where $$W_Q \in \mathbb{R}^{d_q \times d_x}$$
2. **Key**: $$k_i = W_Kx_i$$, where $$W_K \in \mathbb{R}^{d_k\times d_x}$$
3. **Value**: $$v_i = W_Vx_i$$, where $$W_V \in \mathbb{R}^{d_v\times d_x}$$

We also need to pick a good choice of $$g$$. Recall that its inputs are $k_i$ and $q_i$. Since $$g$$ will determine the weights of the linear combination we compute, we should pick a computationally efficient way of comparing how similar $k_i$ and $q_i$ are. Transformers usually use **scaled dot product**, where we scale down the dot product of $k_i$ and $q_i$ by $\sqrt{d_k}$ [^scp].

[^scp]: Two notes about scaled dot product. First, the dot product will require $d_k = d_q$, or the dot product doesn't work. Second, why do we scale down by $\sqrt{d_k}$? Assume that each component of $q_i, k_i \in \mathbb{R^{d_k}}$ is i.i.d. with mean 0 and variance 1. Then their dot product has mean 1 and variance $d_k$ (since the sum is computed over $d_k$ components). Scaling down by $\sqrt{d_k}$ reduces the variance down to 1, which is much nicer.

Now, our attention computation looks like this:
1. Calculate the attention scores: $$s_{j, i} = \frac{k_i \cdot q_i}{\sqrt{d_k}}$$
2. Normalize the attention scores: $$\alpha_j = \text{softmax}(s_j)$$
3. Compute the context vector: $$a_j = \sum^{n}_{i=1} \alpha_{j, i} v_i$$

### Putting it all together
Okay, so we've discussed what happens with a single input vector $x_i$. Now, let's rewrite everything we've learned so far, but performing this computation across all $n$ input vectors simultaneously.

Let's stack the $n$ input vectors into one input matrix $X \in \mathbb{R^{n \times d_x}}$. Then, we can compute all queries, keys, and values as such:

1. **Query matrix**: $$Q = XW_Q$$, where $$Q \in \mathbb{R}^{n \times d_q}$$
2. **Key matrix**: $$K= XW_K$$, where $$K \in \mathbb{R}^{d_k\times n}$$
3. **Value matrix**: $$V = XW_V$$, where $$V \in \mathbb{R}^{n\times d_v}$$

This allows us to reduce the attention function to a single, elegant line:

$$
\begin{aligned} 
\text{Attention}(K, Q, V) &= \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V
\end{aligned}
$$

### Multi-head attention
An attention function, or **attention head**, is defined by a learned $W_Q$, $W_K$, and $W_V$. But why use one set of $W_Q$, $W_K$, and $W_V$ when we could use multiple? 

**Multi-head attention**, as the name implies, uses multiple attention functions. Each one will compute its own attention; then, we concatenate them to get our final context vector. Assuming we have $h$ heads $\{ z_1, ..., z_h \}$ in our multi-head attention block:

$$
\begin{aligned} 
z_i &= \text{Attention}(K, Q, V) \\
a_j &= \text{concat}((z_1, ..., z_i, ..., z_h))
\end{aligned}
$$

The advantages of using multiple attention heads are very appealing. For starters, by choosing different values of $W_Q$, $W_K$, and $W_V$, we can direct our model to focus on different parts of the input sequence. More importantly, we can reduce the dimensions of the weight matrices allowing us to compute more attention scores with less computational cost. In particular, we let $d_q = d_k = d_v = \frac{d}{h}$, where $d$ is the hidden layer size. Since we've concatenated $h$ vectors of length $\frac{d}{h}$, the length of our final context vector $a_i$ is $h \cdot \frac{d}{h}  = d$. Convenient!

Now, we can see that the computational cost of this multihead attention is (more or less) the same as a single-head attention with $d_q = d_k = d_v = d$. So the magic of using multihead attention is that we get to fold in $h$ different "selective summaries" of the sequence into a single context vector, for the same price as a single attention head. 

### Finishing touches
So now we have a bunch of context vectors $A = {a_1, ..., a_n} \in \mathbb{n \times R_{d}}$. In order to get back to the output space, we simply apply a projection matrix $W_O \in  \mathbb{ R_{d} \times d_y}:

$$
\begin{aligned} 
Y &= AW_0
\end{aligned} 
$$

