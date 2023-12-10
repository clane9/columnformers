# Columnformer: A Transformer-inspired model of the brain

<p align="center">
  <img src=".github/images/columns.png" width="600">
</p>

**Work in progress, feedback and collaboration welcome!**

## Motivation

Transformers have been very successful. Could they be useful models of the brain? At a fine scale, the transformer module somewhat reminds us of cortical columns. They receive vector input, do some computation, and produce vector output. Following this analogy, we adapt the transformer to make the overall architecture more brain-like:

- We untie weights across the modules in each block. (The brain doesn't share weights.)
- We flatten the blocks into a single sheet. (Like the cortex.)
- During a forward pass, we unroll the sheet for several time steps. (The brain is recurrent.)
- We spatially embed the sheet and promote local communication. (Connectivity in the brain is mostly local.)

We call our architecture the **columnformer**.

<p align="center">
  <img src=".github/images/columnformer.svg" width="600">
</p>

We think the architecture is interesting because of two key properties:

- **Topography**. By untying weights and embedding the columns spatially, the model has the potential to learn topographic functional specialization, as seen in the primate ventral visual stream [1-3].

- **Recurrence**. Most popular neural network architectures, including the transformer, are purely feedforward. By recursively unrolling the sheet, the columns of our model can communicate in any direction: feedforward, feedback, lateral, etc.

<!-- You could ask, is there even a well-defined feedforward direction? Indeed, any feedforward direction is determined only by the geometry and where input is injected. -->

## Architecture details

See [`model_v1.py`](columnformers/model_v1.py) for the implementation of our initial model. In short, the model "sheet" is just a transformer block but with untied weights across the sequence. Each "column" in the sheet consists of an Attention and MLP module preceded by LayerNorm. The Attention module handles communication between columns, while the MLP does within-column computation [4].

To save parameters, we make a few changes to the Attention and MLP:
  - We use a single attention head
  - We eliminate the value weights and instead set the value to be the input
  - We use a small MLP hidden dimension

<!-- The largest best performing transformers use many attention heads and a large MLP hidden dimension. In our case, we shouldn't need to do this since the columns are untied. Our effective number of heads and MLP hidden dimension scales with the number of columns. -->

We also add a learned bias to our Attention (`attn = softmax(q @ k.T + bias)`), which encodes the learned connectivity between columns.

The key final part of the architecture is a fixed distance matrix encoding the geometry of the sheet. For example, this could be the Euclidean distance matrix computed from a fixed embedding. E.g. a flat 2D grid, points on a sphere, or a stack of 2D layers.

<!-- We could even generalize the distance matrix to arbitrary directed graph edge weights. This would let us encode feedforward architectures as a special case. -->

<p align="center">
  <img src=".github/images/geometries.svg" width="430">
</p>

We use the distance matrix to promote local communication in two ways:

1. By initializing the attention bias (e.g. `bias = -dist**2`).
2. By penalizing the total "wiring cost" of the attention matrix (e.g. `cost = (dist * attn).mean()`).

Effectively, the geometry of the sheet constrains how information can flow through the network [6].

## Questions

- Can we train the model?
- Can we get decent task performance?
- What kinds of connectivity will the model learn?
- What kinds of topographic structure will emerge?
- What kinds of geometries, initialization, and regularization promote more brain-like connectivity and topography?
- How do dataset and task impact connectivity and topography?
- How well do the learned representations match brain activity data?
- Does the architecture have any advantages over the transformer, e.g. in task performance, robustness, scalability, or interpretability?

## Roadmap

### Short-term

- [x] Initial model implementation ([`model_v1.py`](columnformers/model_v1.py))
- [ ] Research related work
- [ ] Implement benchmark train/eval pipelines
  - [ ] Image classification
  - [ ] Masked image modeling
- [ ] Get baseline performance of v1 model
- [ ] Iterate to understand and improve performance
  - [ ] Iterate training recipe
  - [ ] Iterate architecture

### Longer-term

- [ ] Analyze learned topography and connectivity
- [ ] Evaluate brain activity encoding performance
- [ ] Evaluate robustness
- [ ] Efficient implementation, e.g. [sparse attention](https://github.com/facebookresearch/xformers/blob/40d39673285217d9c6b9a0e01e8809a10b771209/xformers/components/attention/random.py#L40)
- [ ] Explore strategies for model scaling, e.g. leveraging sparse connectivity

## Contributing

This project is under active development in collaboration with [MedARC](https://www.medarc.ai/) and we welcome contributions or feedback! If you're interested in the project, please get in touch on [discord](https://discord.com/invite/CqsMthnauZ).

## References and related work

<div id="refs" class="references csl-bib-body" entry-spacing="0">

<div id="ref-Lu2023" class="csl-entry">

<span class="csl-left-margin">\[1\]
</span><span class="csl-right-inline">Z. Lu *et al.*, “End-to-end
topographic networks as models of cortical map formation and human
visual behaviour: Moving beyond convolutions,” *arXiv preprint
arXiv:2308.09431*, 2023.</span>

</div>

<div id="ref-Doshi2023" class="csl-entry">

<span class="csl-left-margin">\[2\]
</span><span class="csl-right-inline">F. R. Doshi and T. Konkle,
“Cortical topographic motifs emerge in a self-organized map of object
space,” *Science Advances*, 2023.</span>

</div>

<div id="ref-Margalit2023" class="csl-entry">

<span class="csl-left-margin">\[3\]
</span><span class="csl-right-inline">E. Margalit *et al.*, “A unifying
principle for the functional organization of visual cortex,” *bioRxiv*,
2023.</span>

</div>

<div id="ref-Karpathy2023" class="csl-entry">

<span class="csl-left-margin">\[4\]
</span><span class="csl-right-inline">A. Karpathy, “Introduction to
transformers.” <https://youtu.be/XfpMkf4rD6E?si=AM9AWDegUaFB7KCe>,
2023.</span>

</div>

<div id="ref-Pang2023A" class="csl-entry">

<span class="csl-left-margin">\[5\]
</span><span class="csl-right-inline">J. C. Pang *et al.*, “Geometric
constraints on human brain function,” *Nature*, 2023.</span>

</div>

<div id="ref-Achterberg2023" class="csl-entry">

<span class="csl-left-margin">\[6\]
</span><span class="csl-right-inline">J. Achterberg *et al.*, “Spatially
embedded recurrent neural networks reveal widespread links between
structural and functional neuroscience findings,” *Nature Machine
Intelligence*, 2023.</span>

</div>

<div id="ref-Hinton2022" class="csl-entry">

<span class="csl-left-margin">\[7\]
</span><span class="csl-right-inline">G. Hinton, “The robot brains
season 2 episode 22.”
<https://www.therobotbrains.ai/who-is-geoff-hinton-part-two>,
2022.</span>

</div>

<div id="ref-Pogodin2021" class="csl-entry">

<span class="csl-left-margin">\[8\]
</span><span class="csl-right-inline">R. Pogodin, Y. Mehta, T.
Lillicrap, and P. E. Latham, “Towards biologically plausible
convolutional networks,” *Advances in Neural Information Processing
Systems*, 2021.</span>

</div>

<div id="ref-Sabour2017" class="csl-entry">

<span class="csl-left-margin">\[9\]
</span><span class="csl-right-inline">S. Sabour, N. Frosst, and G. E.
Hinton, “Dynamic routing between capsules,” *Advances in neural
information processing systems*, 2017.</span>

</div>

<div id="ref-Velickovic2018" class="csl-entry">

<span class="csl-left-margin">\[10\]
</span><span class="csl-right-inline">P. Veličković *et al.*, “Graph
attention networks,” in *International conference on learning
representations*, 2018.</span>

</div>

</div>
