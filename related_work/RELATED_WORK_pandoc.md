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

## Wiring cost minimization

- [@Chen2006] is a classic paper studying wiring cost minimization as an explanatory principle of brain structure and function.
- [@Blauch2022] studies how penalizing wiring cost can explain the emergence of functional topography.

## Transformer variants and alternatives

- [@Dehghani2019] introduce Universal Transformers which share weights across network depth. Very relevant.
- [@Goel2022] uses similar recurrent unrolling in depth.
- [@Tolstikhin2021] introduce MLP-Mixer, a similar architecture to the transformer but replacing classic self-attention with a much simpler communication mechanism using a static mixing MLP.
- [@Liu2022] introduce ConvNext, a similar architecture to MLP-Mixer but using depthwise convolution communication.
- [@Katharopoulos2020] replace softmax based attention with linear kernel dot product attention, reducing compute complexity.
- [@He2024] introduces several simplifications to transformer blocks, including removing skip connections and value/projection weights. They use signal propagation as a way to evaluate architecture changes.
- Attention free transformers (AFT) [@Zhai2021]. The idea of the additive bias in place of the multiplicative query is especially relevant.
- RWKV [@Peng2023] takes inspiration from AFT.
- Capsule networks [@Sabour2017], which have a similar inspiration to what we're exploring.

## Transformers

- [@Touvron2021] introduces techniques for effectively training large vision transformers, including `LayerScale`.

## General inspiration

- The perspective in [@Karpathy2023] viewing the cortex as a uniform sheet of computational modules, and thinking of attention as communication.
- Geoff Hinton's discussion of weight sharing and local constrastive distillation in [@Hinton2022].
- The discussion of geometry constraining brain function in [@Pang2023A].
- Spatially embedded recurrent networks in [@Achterberg2023].

### References
