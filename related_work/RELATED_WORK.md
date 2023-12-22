# Related work

To update this document:

- Add bib entries to [columnformers.bib](columnformers.bib).
- Make edits to [RELATED_WORK_pandoc.md](RELATED_WORK_pandoc.md).
- Generate [RELATED_WORK.md](RELATED_WORK.md) with compiled citations
  using [pandoc](https://pandoc.org/) by running `make`.

## Topographic neural networks

- \[1\] is a major inspiration for this work. The authors introduce the
  All-TNN architecture, which is basically a CNN without weight sharing.
- \[2\], \[3\] are other important works studying the emergence of
  topography in neural networks.
- \[4\] discusses the biological implausibility of weight sharing and
  proposes some strategies for training locally connected networks
  without weight sharing.
- \[5\] shows that imposing topographic constraints on the hidden units
  of a CNN results in emergent processing “streams” similar to the
  primate dorsal/ventral stream.

## Alternatives to transformers

- Attention free transformers (AFT) \[6\]. The idea of the additive bias
  in place of the multiplicative query is especially relevant.
- RWKV with builds on AFT \[7\].
- Graph attention networks \[8\].
- Capsule networks \[9\], which have a similar inspiration to what we’re
  exploring.

## General inspiration

- The perspective in \[10\] viewing the cortex as a uniform sheet of
  computational modules, and thinking of attention as communication.
- Geoff Hinton’s discussion of weight sharing and local constrastive
  distillation in \[11\].
- The discussion of geometry constraining brain function in \[12\].
- Spatially embedded recurrent networks in \[13\].

### References

<div id="refs" class="references csl-bib-body" entry-spacing="0">

<div id="ref-Lu2023" class="csl-entry">

<span class="csl-left-margin">\[1\]
</span><span class="csl-right-inline">Z. Lu *et al.*, “End-to-end
topographic networks as models of cortical map formation and human
visual behaviour: Moving beyond convolutions,” *arXiv preprint
arXiv:2308.09431*, 2023, doi:
[10.48550/arXiv.2308.09431](https://doi.org/10.48550/arXiv.2308.09431).</span>

</div>

<div id="ref-Doshi2023" class="csl-entry">

<span class="csl-left-margin">\[2\]
</span><span class="csl-right-inline">F. R. Doshi and T. Konkle,
“Cortical topographic motifs emerge in a self-organized map of object
space,” *Science Advances*, 2023, doi:
[10.1126/sciadv.ade8187](https://doi.org/10.1126/sciadv.ade8187).</span>

</div>

<div id="ref-Margalit2023" class="csl-entry">

<span class="csl-left-margin">\[3\]
</span><span class="csl-right-inline">E. Margalit *et al.*, “A unifying
principle for the functional organization of visual cortex,” *bioRxiv*,
2023, doi:
[10.1101/2023.05.18.541361](https://doi.org/10.1101/2023.05.18.541361).</span>

</div>

<div id="ref-Pogodin2021" class="csl-entry">

<span class="csl-left-margin">\[4\]
</span><span class="csl-right-inline">R. Pogodin, Y. Mehta, T.
Lillicrap, and P. E. Latham, “Towards biologically plausible
convolutional networks,” *Advances in Neural Information Processing
Systems*, 2021.</span>

</div>

<div id="ref-Finzi2023" class="csl-entry">

<span class="csl-left-margin">\[5\]
</span><span class="csl-right-inline">D. Finzi *et al.*, “A single
computational objective drives specialization of streams in visual
cortex,” *bioRxiv*, 2023, doi:
[10.1101/2023.12.19.572460](https://doi.org/10.1101/2023.12.19.572460).</span>

</div>

<div id="ref-Zhai2021" class="csl-entry">

<span class="csl-left-margin">\[6\]
</span><span class="csl-right-inline">S. Zhai *et al.*, “An attention
free transformer,” *arXiv preprint arXiv:2105.14103*, 2021.</span>

</div>

<div id="ref-Peng2023" class="csl-entry">

<span class="csl-left-margin">\[7\]
</span><span class="csl-right-inline">B. Peng *et al.*, “RWKV:
Reinventing RNNs for the transformer era,” *arXiv preprint
arXiv:2305.13048*, 2023.</span>

</div>

<div id="ref-Velickovic2018" class="csl-entry">

<span class="csl-left-margin">\[8\]
</span><span class="csl-right-inline">P. Veličković *et al.*, “Graph
attention networks,” in *International conference on learning
representations*, 2018.</span>

</div>

<div id="ref-Sabour2017" class="csl-entry">

<span class="csl-left-margin">\[9\]
</span><span class="csl-right-inline">S. Sabour, N. Frosst, and G. E.
Hinton, “Dynamic routing between capsules,” *Advances in neural
information processing systems*, 2017.</span>

</div>

<div id="ref-Karpathy2023" class="csl-entry">

<span class="csl-left-margin">\[10\]
</span><span class="csl-right-inline">A. Karpathy, “Introduction to
transformers.” <https://youtu.be/XfpMkf4rD6E?si=AM9AWDegUaFB7KCe>,
2023.</span>

</div>

<div id="ref-Hinton2022" class="csl-entry">

<span class="csl-left-margin">\[11\]
</span><span class="csl-right-inline">G. Hinton, “The robot brains
season 2 episode 22.”
<https://www.therobotbrains.ai/who-is-geoff-hinton-part-two>,
2022.</span>

</div>

<div id="ref-Pang2023A" class="csl-entry">

<span class="csl-left-margin">\[12\]
</span><span class="csl-right-inline">J. C. Pang *et al.*, “Geometric
constraints on human brain function,” *Nature*, 2023.</span>

</div>

<div id="ref-Achterberg2023" class="csl-entry">

<span class="csl-left-margin">\[13\]
</span><span class="csl-right-inline">J. Achterberg *et al.*, “Spatially
embedded recurrent neural networks reveal widespread links between
structural and functional neuroscience findings,” *Nature Machine
Intelligence*, 2023.</span>

</div>

</div>
