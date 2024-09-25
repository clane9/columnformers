# Brain-inspired vision transformers

<p align="center">
  <img src=".github/images/columns.png" width="600">
</p>

**Work in progress, feedback and collaboration welcome!**

Vision Transformers (ViTs) have a structure that is vaguely similar to the human visual cortex. The overall network is organized as a sequence of blocks, similar to the hierarchy of visual areas. The output of each block is a grid of patch embeddings, similar to a retinotopic representation map. At each patch position, there is a transformer module that gathers input from other positions, performs some computation, and communicates its output, similar to the computations in cortical hypercolumns. The goal of this project is to make the ViT even more brain-like. In particular, we are interested in incorporating two key properties:

**Topographic functional specialization**. Within a ViT block, weights are shared across all patch positions. This means that at each patch position, the same computations are performed. In visual cortex however, there is within-area functional specialization. For example, in IT cortex, there are specialized sub-areas for processing different object categories. How can we add topographic functional specialization to the ViT?

**Recurrence**. The ViT operates in a purely feedforward way. In visual cortex however, there are significant lateral and feedback connections between areas. Moreover, these connections are crucial for explaining behavior. How can we add recurrence to the ViT?

## Contributing

This project is under active development in collaboration with [MedARC](https://www.medarc.ai/) and we welcome contributions or feedback! If you're interested in the project, please get in touch on [discord](https://discord.com/invite/CqsMthnauZ).
