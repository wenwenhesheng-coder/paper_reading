# Bayesian Learning and RPCA: Broad Reading Map

## Scope

This list treats Bayesian learning and RPCA as two connected but independent research
tracks. The project contains Masked-RPCA and deep-unfolding material, but these are not
used as a restriction: general variational inference, Bayesian deep learning, matrix and
tensor RPCA, and online/non-convex RPCA are all in scope.

## Bayesian learning foundation

1. David M. Blei, Alp Kucukelbir, and Jon D. McAuliffe. "Variational Inference: A
   Review for Statisticians." *Journal of the American Statistical Association*,
   112(518):859-877, 2017. DOI: 10.1080/01621459.2017.1285773.

   The best starting point for ELBO, mean-field variational inference, coordinate-ascent
   updates, and model-selection issues. It provides the inference vocabulary needed for
   most Bayesian matrix/tensor factorization papers.

2. Matthew D. Hoffman, David M. Blei, Chong Wang, and John Paisley. "Stochastic
   Variational Inference." *Journal of Machine Learning Research*, 14:1303-1347,
   2013.

   Essential when inference must scale to video streams, large matrices, or minibatches.
   It provides a principled route from batch variational Bayes to online learning.

3. Yarin Gal and Zoubin Ghahramani. "Dropout as a Bayesian Approximation:
   Representing Model Uncertainty in Deep Learning." *Proceedings of ICML*, 2016.

   A useful Bayesian-deep-learning baseline if uncertainty needs to be attached to a
   neural or unfolded network without a full posterior over every parameter.

4. Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. "Weight
   Uncertainty in Neural Networks." *Proceedings of ICML*, 2015.

   The Bayes-by-Backprop reference for variational parameter posteriors in neural
   networks. It is relevant to a learned solver but is independent of RPCA.

## RPCA foundations beyond Bayesian methods

1. Emmanuel J. Candes, Xiaodong Li, Yi Ma, and John Wright. "Robust Principal
   Component Analysis?" *Journal of the ACM*, 58(3), 2011.
   DOI: 10.1145/1970392.1970395.

   The canonical exact-recovery theory for Principal Component Pursuit (low-rank plus
   sparse decomposition). This is the required theoretical citation for almost any RPCA
   paper.

2. Huan Xu, Constantine Caramanis, and Sujay Sanghavi. "Robust PCA via Outlier
   Pursuit." *IEEE Transactions on Information Theory*, 58(5):3047-3064, 2012.
   DOI: 10.1109/TIT.2011.2173722.

   Addresses column-sparse corruptions rather than entrywise noise. This is helpful for
   sample-level anomalies, corrupted frames, and grouped outlier settings.

3. Praneeth Netrapalli, U. N. Niranjan, Sujay Sanghavi, Anima Anandkumar, and
   Prateek Jain. "Non-convex Robust PCA." *Proceedings of NeurIPS*, 2014.

   A key non-convex alternative to nuclear-norm RPCA. It is especially useful for
   understanding factorized low-rank parameterizations and their convergence behavior.

4. Jiashi Feng, Huan Xu, and Shuicheng Yan. "Online Robust PCA via Stochastic
   Optimization." *Proceedings of NeurIPS*, 2013.

   Establishes an online RPCA route for streaming data and should be considered
   separately from batch video decomposition methods.

## Priority 1: Read and cite first

1. Xinghao Ding, Lihan He, and Lawrence Carin. "Bayesian Robust Principal
   Component Analysis." *IEEE Transactions on Image Processing*, 20(12):3419-3430,
   2011. DOI: 10.1109/TIP.2011.2156801.

   Direct Bayesian RPCA baseline. It jointly infers low-rank, sparse-outlier, and noise
   components with a hierarchical model. It additionally discusses Markov dependence
   between successive video frames, making it especially relevant to the project’s
   temporal foreground prior.

2. S. Derin Babacan, Martin Luessi, Rafael Molina, and Aggelos K. Katsaggelos.
   "Sparse Bayesian Methods for Low-Rank Matrix Estimation." *IEEE Transactions on
   Signal Processing*, 60(8):3964-3977, 2012. DOI: 10.1109/TSP.2012.2197748.

   Uses sparse Bayesian learning (SBL) and automatic relevance determination to learn
   the effective rank in matrix completion and RPCA. This is the cleanest reference for
   replacing manually selected rank/regularization parameters with learned hierarchical
   precisions.

3. Jing Liu and Bhaskar D. Rao. "Sparse Bayesian Learning for Robust PCA:
   Algorithms and Analyses." *IEEE Transactions on Signal Processing*,
   67(22):5837-5849, 2019. DOI: 10.1109/TSP.2019.2946026.

   A modern, direct RPCA formulation using empirical-Bayes MAP-EM. It derives a
   rank-plus-sparsity-equivalent objective and includes a variant that exploits the
   sparse outlier support pattern. This is the closest Bayesian paper to a foreground
   mask/support prior.

4. Naiyan Wang and Dit-Yan Yeung. "Bayesian Robust Matrix Factorization for Image
   and Video Processing." *Proceedings of ICCV*, 1785-1792, 2013.
   DOI: 10.1109/ICCV.2013.224.

   Top computer-vision conference paper. Its full Bayesian robust matrix factorization
   uses a Laplace-mixture observation model and includes a spatial/temporal-proximity
   extension for image and video outliers. It is a strong comparison point for a Bayesian
   version of Masked-RPCA.

## Priority 2: Use for spatial-temporal or tensor extensions

5. Ningning Han, Yumeng Song, and Zhanjie Song. "Bayesian Robust Principal
   Component Analysis with Structured Sparse Component." *Computational Statistics &
   Data Analysis*, 109:144-158, 2017. DOI: 10.1016/j.csda.2016.12.005.

   Variational Bayesian RPCA with correlated support and amplitudes for the sparse
   component. The structured support is directly useful when foreground masks should be
   spatially contiguous instead of pixelwise independent. The venue is solid but less
   central than the IEEE papers above.

6. Qibin Zhao, Guoxu Zhou, Liqing Zhang, Andrzej Cichocki, and Shun-ichi Amari.
   "Bayesian Robust Tensor Factorization for Incomplete Multiway Data." *IEEE
   Transactions on Neural Networks and Learning Systems*, 27(4):736-748, 2016.
   DOI: 10.1109/TNNLS.2015.2423694.

   A full Bayesian low-rank-plus-sparse tensor model with variational inference,
   automatic CP-rank selection, missing-data handling, and predictive distributions.
   This is relevant if video is modeled natively as height x width x time rather than
   stacking vectorized frames into a matrix.

7. Yang Zhou and Yiu-Ming Cheung. "Bayesian Low-Tubal-Rank Robust Tensor
   Factorization with Multi-Rank Determination." *IEEE Transactions on Pattern
   Analysis and Machine Intelligence*, 43(1):62-76, 2021.
   DOI: 10.1109/TPAMI.2019.2923240.

   The strongest journal choice for a tensor-video direction. It employs a fully
   Bayesian low-tubal-rank model, automatically determines multi-rank and the
   low-rank/sparse trade-off, and demonstrates image denoising and background modeling.

## Priority 3: Useful methodological supplement

8. Kewei Tang, Zhixun Su, Jie Zhang, Lihong Cui, Wei Jiang, Xiaonan Luo, and Xiyan
   Sun. "Bayesian Rank Penalization." *Neural Networks*, 116:246-256, 2019.
   DOI: 10.1016/j.neunet.2019.04.018.

   Develops a general Bayesian rank penalty and Bayesian versions of RPCA/LRR. Read it
   for posterior uncertainty and rank regularization rather than for foreground
   segmentation details.

## Recommended citation strategy

- Cite Ding et al. (2011) and Babacan et al. (2012) in the Bayesian-RPCA background
  section.
- Cite Liu and Rao (2019) when motivating automatic rank/sparsity learning and sparse
  support modeling.
- Cite Wang and Yeung (2013) and Han et al. (2017) when motivating spatial-temporal
  foreground-mask structure.
- Cite Zhao et al. (2016) and Zhou and Cheung (2021) only if the manuscript proposes or
  compares a tensor/video-volume formulation.

## A credible next research angle

A good extension of the current ROMAN formulation is a variational-Bayesian
Masked-RPCA/deep-unfolding network: place hierarchical priors on low-rank factors,
mask sparsity, and noise; use spatial-temporal structured support for masks; and unfold
the variational or MAP-EM updates. This would retain ROMAN’s interpretable layers while
adding automatic regularization selection and calibrated foreground uncertainty.
