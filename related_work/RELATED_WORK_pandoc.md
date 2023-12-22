---
bibliography: [columnformers.bib]
nocite: '@*'
---

# Related work

To update this document:

- Add bib entries to [columnformers.bib](columnformers.bib).
- Make edits to [RELATED_WORK_pandoc.md](RELATED_WORK_pandoc.md).
- Generate [RELATED_WORK.md](RELATED_WORK.md) with compiled citations using [pandoc](https://pandoc.org/) by running `make`.

## Topographic neural networks

- [@Lu2023] is a major inspiration for this work. The authors introduce the All-TNN architecture, which is basically a CNN without weight sharing.
- [@Doshi2023; @Margalit2023] are other important works studying the emergence of topography in neural networks.
- [@Pogodin2021] discusses the biological implausibility of weight sharing and proposes some strategies for training locally connected networks without weight sharing.
- [@Finzi2023] shows that imposing topographic constraints on the hidden units of a CNN results in emergent processing "streams" similar to the primate dorsal/ventral stream.

## Alternatives to transformers

- Attention free transformers (AFT) [@Zhai2021]. The idea of the additive bias in place of the multiplicative query is especially relevant.
- RWKV with builds on AFT [@Peng2023].
- Graph attention networks [@Velickovic2018].
- Capsule networks [@Sabour2017], which have a similar inspiration to what we're exploring.

## General inspiration

- The perspective in [@Karpathy2023] viewing the cortex as a uniform sheet of computational modules, and thinking of attention as communication.
- Geoff Hinton's discussion of weight sharing and local constrastive distillation in [@Hinton2022].
- The discussion of geometry constraining brain function in [@Pang2023A].
- Spatially embedded recurrent networks in [@Achterberg2023].

### References
