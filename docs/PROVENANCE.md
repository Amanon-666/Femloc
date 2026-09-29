# Source code and papers

- GGA: Ballas & Diou, CVPR 2025, corrected arXiv v7 (2025-07-21), https://arxiv.org/html/2502.20162v7 . Official MIT source https://github.com/aristotelisballas/GGA commit `5450c2b8a9072e014be5c42a5bb33cfe1a03bfb9`. Adapted `ERM_GGA` from `domainbed/algorithms/algorithms.py` (early search, perturb feature extractor, pairwise min gradient cosine, loss gate, then ERM step). Original code uses classification cross-entropy and ImageNet ResNet; this branch uses 2D regression and RSSI encoders.
- DomainBed: https://github.com/facebookresearch/DomainBed , MIT. Its leave-one-test-domain and source-only selection policy informed the protocol. We do not copy its image dataset or network stack.
- Set Transformer: Lee et al., ICML 2019, https://proceedings.mlr.press/v97/lee19d.html . Official MIT source https://github.com/juho-lee/set_transformer commit `73432c640ac78140496d6738416c54d32c686d65`. `src/set_blocks.py` adapts its MAB/SAB/PMA blocks by adding masks for padded UJI AP sets.
- UJIIndoorLoc: Torres-Sospedra et al., IPIN 2014; official UCI data https://archive.ics.uci.edu/dataset/310/ujiindoorloc . Training file has 19,937 rows; validation/test has 1,111 later rows. Dataset metadata gives WAP sentinel 100, coordinates and user/device/time fields.
- Copies/adaptations retain attribution and the corresponding MIT license texts in `third_party/`.
