# Related work

The full comparison, with paper links and an honest account of what EigenTune does and does not add, is in
[research/RELATED_WORK.md](research/RELATED_WORK.md).

In one sentence: training only singular-value scales (SVFit), an `r × r` core between frozen SVD bases (LoRA-XS) and
sparse patterns over singular-vector outer products (SVFT) are published ideas; EigenTune is a careful implementation of
them behind one interface, with the systems work in the kernels and the storage accounting, not a new adaptation.
