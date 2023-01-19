# Description of Methods

## Latent Representations Visualization
Latent representations visualization is suggested for better understanding of intra-layer manifold transformations by the neural network.

Visualization is performed on the data, provided by the user. The method takes representations on some layers (specified by the user). Those representations are projected onto 2d space with a dimensionality reduction technique (either UMAP [1] or PCA [2]).

[1] McInnes, L, Healy, J, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, ArXiv e-prints 1802.03426, 2018.

[2] Tipping, M. E., and Bishop, C. M. (1999). “Probabilistic principal component analysis”. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 61(3), 611-622.

## Persistent Homologies
Persistent homologies analysis is suggested for better understanding of intra-layer manifold transformations by the neural network.

The calculation of persistent homologies is carried out using the Vietoris-Rips complex [1]. The results are displayed using barcode representation [2].

[1] https://en.wikipedia.org/wiki/Vietoris%E2%80%93Rips_complex

[2] https://en.wikipedia.org/wiki/Persistent_homology

## Uncertainty Estimation via Bayesianization
One of the advantages of Bayesian neural networks (BNNs) [1] over classic ones (NNs) is their ability to evaluate uncertainty of their prediction. Our library takes advantage of this property of Bayesian neural networks. We provide a collection of methods that allows to build a Bayesian version of the trained neural network (to perform _bayesianisation_) and use the Bayesian model to evaluate uncertainty of the model.

How a neurla network can be bayesianized? Let's say, original model has weights $W_i$. Basic BNN can be builed by replacing those weights with distributions $\hat{W}_i \sim W_i \cdot diag(z_1,...,z_n)$, where $z_j \sim Bernoulli(p)$. We also provide more advanced, beta-distribution bayesianization, where we take $p \sim B(\alpha, \beta)$, where $B(\alpha, \beta)$ is beta-distribution with parameters $\alpha, \beta$.

For computational tractability we use Monte-Carlo sampling [2].

[1] Jospin, Laurent & Buntine, Wray & Boussaid, Farid & Laga, Hamid & Bennamoun, Mohammed. (2020). Hands-on Bayesian Neural Networks -- a Tutorial for Deep Learning Users. 

[2] https://en.wikipedia.org/wiki/Monte_Carlo_method
