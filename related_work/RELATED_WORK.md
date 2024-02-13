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

## Wiring cost minimization

- \[6\] is a classic paper studying wiring cost minimization as an
  explanatory principle of brain structure and function.
- \[7\] studies how penalizing wiring cost can explain the emergence of
  functional topography.

## Recurrence

- \[8\] introduce Universal Transformers which share weights across
  network depth. Very relevant.
- \[9\] uses similar recurrent unrolling in depth.

## Weight untying

- \[10\] reviews sparse mixture of expert (SMoE) models. The key idea of
  SMoE models is that for each layer, each token is processed by only a
  subset of parameters. This increases “capacity” without increasing
  compute. Similarly, we can view our sheet of columns as a giant
  mixture of experts that uses self-attention for routing.
- Mixtral 8x7B \[11\] famously uses SMoE feedforward modules.

## Communication mechanisms

- \[12\] introduce MLP-Mixer, a similar architecture to the transformer
  but replacing classic self-attention with a much simpler communication
  mechanism using a static mixing MLP.
- \[13\] introduce ConvNext, a similar architecture to MLP-Mixer but
  using depthwise convolution communication.
- Attention free transformers (AFT) \[14\]. The idea of the additive
  bias in place of the multiplicative query is especially relevant.
- RWKV \[15\] takes inspiration from AFT, reformulating attention as
  recurrent communication, similar to \[16\].
- Capsule networks \[17\], which have a similar inspiration to what
  we’re exploring, leverage dynamic routing.

## Transformer variants

- \[18\] introduces several simplifications to transformer blocks,
  including removing skip connections and value/projection weights. They
  use signal propagation as a way to evaluate architecture changes.
- \[19\] introduces the GLU variants to the feedforward module.

## Training transformers

- \[20\] introduces techniques for effectively training large vision
  transformers, including `LayerScale`.

## General inspiration

- The perspective in \[21\] viewing the cortex as a uniform sheet of
  computational modules, and thinking of attention as communication.
- Geoff Hinton’s discussion of weight sharing and local constrastive
  distillation in \[22\].
- The discussion of geometry constraining brain function in \[23\].
- Spatially embedded recurrent networks in \[24\].

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

<div id="ref-Chen2006" class="csl-entry">

<span class="csl-left-margin">\[6\]
</span><span class="csl-right-inline">B. L. Chen, D. H. Hall, and D. B.
Chklovskii, “Wiring optimization can relate neuronal structure and
function,” *Proceedings of the National Academy of Sciences*,
2006.</span>

</div>

<div id="ref-Blauch2022" class="csl-entry">

<span class="csl-left-margin">\[7\]
</span><span class="csl-right-inline">N. M. Blauch, M. Behrmann, and D.
C. Plaut, “A connectivity-constrained computational account of
topographic organization in primate high-level visual cortex,”
*Proceedings of the National Academy of Sciences*, 2022.</span>

</div>

<div id="ref-Dehghani2019" class="csl-entry">

<span class="csl-left-margin">\[8\]
</span><span class="csl-right-inline">M. Dehghani *et al.*, “Universal
transformers,” in *International conference on learning
representations*, 2019. Available:
<https://openreview.net/forum?id=HyzdRiR9Y7></span>

</div>

<div id="ref-Goel2022" class="csl-entry">

<span class="csl-left-margin">\[9\]
</span><span class="csl-right-inline">S. Goel, S. Kakade, A. Kalai, and
C. Zhang, “Recurrent convolutional neural networks learn succinct
learning algorithms,” in *Advances in neural information processing
systems*, S. Koyejo, S. Mohamed, A. Agarwal, D. Belgrave, K. Cho, and A.
Oh, Eds., Curran Associates, Inc., 2022, pp. 7328–7341. Available:
<https://proceedings.neurips.cc/paper_files/paper/2022/file/300900a706e788163b88ed3c08cbe23c-Paper-Conference.pdf></span>

</div>

<div id="ref-Fedus2022" class="csl-entry">

<span class="csl-left-margin">\[10\]
</span><span class="csl-right-inline">W. Fedus, J. Dean, and B. Zoph, “A
review of sparse expert models in deep learning,” *arXiv preprint
arXiv:2209.01667*, 2022, Available:
<https://arxiv.org/abs/2209.01667></span>

</div>

<div id="ref-Jiang2024" class="csl-entry">

<span class="csl-left-margin">\[11\]
</span><span class="csl-right-inline">A. Q. Jiang *et al.*, “Mixtral of
experts,” *arXiv preprint arXiv:2401.04088*, 2024, Available:
<https://arxiv.org/abs/2401.04088></span>

</div>

<div id="ref-Tolstikhin2021" class="csl-entry">

<span class="csl-left-margin">\[12\]
</span><span class="csl-right-inline">I. O. Tolstikhin *et al.*,
“MLP-mixer: An all-MLP architecture for vision,” in *Advances in neural
information processing systems*, M. Ranzato, A. Beygelzimer, Y. Dauphin,
P. S. Liang, and J. W. Vaughan, Eds., Curran Associates, Inc., 2021, pp.
24261–24272. Available:
<https://proceedings.neurips.cc/paper_files/paper/2021/file/cba0a4ee5ccd02fda0fe3f9a3e7b89fe-Paper.pdf></span>

</div>

<div id="ref-Liu2022" class="csl-entry">

<span class="csl-left-margin">\[13\]
</span><span class="csl-right-inline">Z. Liu *et al.*, “A ConvNet for
the 2020s,” in *Proceedings of the IEEE/CVF conference on computer
vision and pattern recognition (CVPR)*, 2022, pp. 11976–11986.</span>

</div>

<div id="ref-Zhai2021" class="csl-entry">

<span class="csl-left-margin">\[14\]
</span><span class="csl-right-inline">S. Zhai *et al.*, “An attention
free transformer,” *arXiv preprint arXiv:2105.14103*, 2021.</span>

</div>

<div id="ref-Peng2023" class="csl-entry">

<span class="csl-left-margin">\[15\]
</span><span class="csl-right-inline">B. Peng *et al.*, “RWKV:
Reinventing RNNs for the transformer era,” *arXiv preprint
arXiv:2305.13048*, 2023.</span>

</div>

<div id="ref-Katharopoulos2020" class="csl-entry">

<span class="csl-left-margin">\[16\]
</span><span class="csl-right-inline">A. Katharopoulos, A. Vyas, N.
Pappas, and F. Fleuret, “Transformers are rnns: Fast autoregressive
transformers with linear attention,” in *International conference on
machine learning*, PMLR, 2020, pp. 5156–5165.</span>

</div>

<div id="ref-Sabour2017" class="csl-entry">

<span class="csl-left-margin">\[17\]
</span><span class="csl-right-inline">S. Sabour, N. Frosst, and G. E.
Hinton, “Dynamic routing between capsules,” *Advances in neural
information processing systems*, 2017.</span>

</div>

<div id="ref-He2024" class="csl-entry">

<span class="csl-left-margin">\[18\]
</span><span class="csl-right-inline">B. He and T. Hofmann, “Simplifying
transformer blocks,” in *The twelfth international conference on
learning representations*, 2024. Available:
<https://openreview.net/forum?id=RtDok9eS3s></span>

</div>

<div id="ref-Shazeer2020" class="csl-entry">

<span class="csl-left-margin">\[19\]
</span><span class="csl-right-inline">N. Shazeer, “Glu variants improve
transformer,” *arXiv preprint arXiv:2002.05202*, 2020, Available:
<https://arxiv.org/abs/2002.05202></span>

</div>

<div id="ref-Touvron2021" class="csl-entry">

<span class="csl-left-margin">\[20\]
</span><span class="csl-right-inline">H. Touvron *et al.*, “Going deeper
with image transformers,” in *Proceedings of the IEEE/CVF international
conference on computer vision*, 2021, pp. 32–42.</span>

</div>

<div id="ref-Karpathy2023" class="csl-entry">

<span class="csl-left-margin">\[21\]
</span><span class="csl-right-inline">A. Karpathy, “Introduction to
transformers.” <https://youtu.be/XfpMkf4rD6E?si=AM9AWDegUaFB7KCe>,
2023.</span>

</div>

<div id="ref-Hinton2022" class="csl-entry">

<span class="csl-left-margin">\[22\]
</span><span class="csl-right-inline">G. Hinton, “The robot brains
season 2 episode 22.”
<https://www.therobotbrains.ai/who-is-geoff-hinton-part-two>,
2022.</span>

</div>

<div id="ref-Pang2023A" class="csl-entry">

<span class="csl-left-margin">\[23\]
</span><span class="csl-right-inline">J. C. Pang *et al.*, “Geometric
constraints on human brain function,” *Nature*, 2023.</span>

</div>

<div id="ref-Achterberg2023" class="csl-entry">

<span class="csl-left-margin">\[24\]
</span><span class="csl-right-inline">J. Achterberg *et al.*, “Spatially
embedded recurrent neural networks reveal widespread links between
structural and functional neuroscience findings,” *Nature Machine
Intelligence*, 2023.</span>

</div>

<div id="ref-Puxeddu2024" class="csl-entry">

<span class="csl-left-margin">\[25\]
</span><span class="csl-right-inline">J. A. S. Puxeddu Maria Grazia AND
Faskowitz, “Relation of connectome topology to brain volume across 103
mammalian species,” *PLOS Biology*, Feb. 2024, doi:
[10.1371/journal.pbio.3002489](https://doi.org/10.1371/journal.pbio.3002489).</span>

</div>

<div id="ref-Velickovic2018" class="csl-entry">

<span class="csl-left-margin">\[26\]
</span><span class="csl-right-inline">P. Veličković *et al.*, “Graph
attention networks,” in *International conference on learning
representations*, 2018.</span>

</div>

</div>
