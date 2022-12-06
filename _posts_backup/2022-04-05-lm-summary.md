---
title: 'Language Models'
date: 2022-04-05
permalink: /posts/2022/04/lm-summary/
tags:
  - nlp
---

Sparknotes, but for language models.

## Introduction
Anybody can rattle off a list of well-known pretrained language models, but there are several key characteristics that are worth memorizing. Here's a quick guide to the key characteristics of the various LM's. (To be completed 4/9/22)

## Summary 

| Model                                                                                   | Year | Type    | Learning Objective                 | Architecture                           | Model Sizes (L, H, A)                                                                | Positional Embeddings |
|-----------------------------------------------------------------------------------------|------|---------|------------------------------------|----------------------------------------|--------------------------------------------------------------------------------------|-----------------------|
| [BERT](https://arxiv.org/abs/1810.04805v2)                                              | 2017 | MLM     | MLM/Cloze Next Sentence Prediction | Bidirectional encoder-only Transformer | BERT-base (110M):<br/> (12, 768, 16)<br/> BERT-large (340M):<br/>(24, 1024, 16)      | Absolute              |
| [GPT-2](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf) | 2018 | Causal  | Language modeling                  | L2R decoder-only Transformer           | GPT2-medium (345M):<br/> (24, 1024, 20)<br/>  GPT2-large (774M): <br/>(36, 1280, 20) | Absolute              |
| [GPT-3](https://arxiv.org/pdf/2005.14165.pdf)                                           | 2020 | Causal  | Language modeling                  | L2R decoder-only Transformer           | Ada (350M)<br/> Babbage (2.7B)<br/> Curie (6.7B)<br/> Davinci (175B)<br/>            | Absolute              |
| [T5](https://arxiv.org/pdf/1910.10683.pdf)                                              | 2019 | Seq2Seq | Span reconstruction denoising      | Encoder-decoder Transformer            | T5-base (220M): <br/>(12, 768, 12)<br/> T5-large (770M): <br/>(24, 1024, 16)         | Relative              |
| [BART](https://arxiv.org/pdf/1910.13461.pdf)                                            | 2019 | Seq2Seq | Corrupted span reconstruction      | Encoder-decoder Transformer            | bart-base (140M): <br/>(12, 768, 16)<br/> bart-large (400M):<br/>(24, 1024, 16)      | Absolute              |

## BERT

### ROBERTA

### ELECTRA

## GPT Family

### GPT-2

### GPT-3

### InstructGPT GPT-3

### GPT-J

### GPT-Neo
