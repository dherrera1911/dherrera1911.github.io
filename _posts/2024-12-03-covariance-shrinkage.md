---
title: "Covariance shrinkage for begginers with Python implementation"
date: 2024-12-03
permalink: /posts/2024/09/covariance-shrinkage-for-begginers/
tags:
  - Statistics
  - Estimation
  - Python
---

Covariance matrices are one of the most important objects in
statistics and machine learning.
However, estimating covariance matrices can be difficult,
especially in high-dimensional settings. In this post we
introduce covariance shrinkage, a technique
to improve the estimation of covariance matrices. We also
provide a PyTorch implementation of a popular shrinkage
estimator, the Oracle Approximating Shrinkage (OAS) estimator.

## Introduction

One issue with estimating covariance matrices is that
for a random variable with $$p$$ dimensions, the covariance
matrix has $$p \times p$$ entries. Of those entries, $$(p+1)p/2$$
are independent parameters need to be estimated, because
of the symmetry of the covariance matrix.[^1] This means
that the number of parameters needed to estimate the
covariance matrix grows fast with the number of dimensions.
An interesting and challenging statistical problem is how to
estimate a covariance matrix when the number of observations $$n$$
is not large compared to the number of dimensions $$p$$.

The simplest approach to estimating the covariance
matrix is to use the sample covariance. Given a dataset of
$$n$$ observations of $$p$$-dimensional
random variable $$X \in \mathbb{R}^p$$, with the observations labeled
as $$X_1, X_2, \ldots, X_n$$, the sample covariance matrix is
defined as

$$S = \frac{1}{n-1} \sum_{i=1}^n (X_i - \bar{X})(X_i - \bar{X})^T$$

where $$\bar{X}$$ is the sample mean. $$S$$ is an unbiased
estimator of the true population covariance matrix. This means
that the expected value of $$S$$ is the true covariance matrix.

However, the sample covariance has some drawbacks as an
estimator. Mainly, $$S$$ is unstable when $$n$$ is small
compared to $$p$$, meaning that it can have large errors.
Let's visualize the variability in $$S$$ with some simulations:

```python
import torch
from torch.distributions import MultivariateNormal
import matplotlib.pyplot as plt

torch.manual_seed(2)

n_dim = 10
n_samples = 20
n_reps = 5

cov_pop = torch.linspace(start=0.1, end=1, steps=n_dim)
cov_pop = torch.diag(cov_pop)

# Compute sample covariance matrices
covs_sample = []
for _ in range(n_reps):
    X = MultivariateNormal(torch.zeros(n_dim), cov_pop).sample((n_samples,))
    covs_sample.append(torch.cov(X.T))

# Plot sample covariance matrices
max_val = 1.2
fig, axs = plt.subplots(1, n_reps+1, figsize=(15, 3))

# Plot and store the image objects
im = axs[0].imshow(cov_pop, cmap='seismic', vmin=-max_val, vmax=max_val)
axs[0].set_title("Population cov", fontsize=18)
for i, ax in enumerate(axs[1:]):
    im = ax.imshow(covs_sample[i], cmap='seismic', vmin=-max_val, vmax=max_val)
    ax.set_title(f"Sample cov {i+1}", fontsize=18)

plt.tight_layout()

# Adjust the subplots to make room for the color bar
fig.subplots_adjust(right=0.85)

# Add an axes for the color bar on the right
cbar_ax = fig.add_axes([0.87, 0.25, 0.02, 0.5])

# Add the color bar to the figure
fig.colorbar(im, cax=cbar_ax)

plt.show()
```
![](/files/blog/shrinkage/cov_var1.png)

Also, many applications require inverting the covariance matrix.
For one, $$S$$ is not even invertible if the number of observations
is less than the number of dimensions. But even if the sample
covariance is invertible, the estimation errors can be highly
amplified when inverting the matrix.

Thus, the sample covariance matrix might not be the best
estimator for some applications. How can we obtain better
estimates of the covariance matrix?
One alternative is to use shrinkage.

## Shrinkage estimators

Shrinkage estimators are a class of estimators that shrink the
estimates towards a target value. These estimators reduce the
variance in the estimates, at the cost of introducing some bias,
a classic trade-off in statistics.

Shrinkage is an essential idea in statistics. Intuitively,
if we observe an extreme value in a random sample
(like the entries of the sample covariance), it is likely that
noise contributed to the extremeness of the value. In other words,
the extreme value is not representative of the true underlying
parameter. Shrinkage accounts for this phenomenon by
pulling the more extreme values towards the middle.

One way to apply the shrinkage rationale above to a covariance
matrix is to compute a "middle", or "target", that we can shrink
the entries of the covariance towards. This will reduce the variance
of the estimates, at the cost of introducing some bias.
A popular target matrix is the diagonal matrix that
shares the same total variance as the sample covariance matrix:

$$\hat{F} = \frac{1}{p} \text{tr}(S) I$$

where $$I$$ is the identity matrix. $$\hat{F}$$ is a highly
structured estimator of $$\Sigma$$, that has low variance but
possibly high bias.

We can bring the values of $$S$$ towards $$\hat{F}$$ (i.e. shrink
$$S$$) by taking a linear combination of the two matrices.
This achieves a compromise between the low bias
of $$S$$ and the low variance of $$\hat{F}$$. This leads to
the class of linear shrinkage estimators:

$$\hat{\Sigma} = (1-\rho) S + \rho \hat{F}$$

In the formula above, $$\rho$$ is a parameter that controls the
amount of shrinkage. When $$\rho = 0$$, the estimate is the sample
covariance matrix.

The challenge in covariance shrinkage estimation is to find
the optimal value of $$\rho$$ that optimizes some criterion.
We present a popular method and its implementation in PyTorch
the next section.


## Oracle Approximating Shrinkage estimator (with implementation)

The Oracle Approximating Shrinkage (OAS) estimator is a method
for obtaining the value of $$\rho$$.[^2] The estimator tries to
approximate the optimal value of $$\rho$$ that minimizes the
mean squared error (MSE) with respect to the true covariance
matrix:

$$\min_{\rho} \mathbb{E} \left[ \left\| \hat{\Sigma} - \Sigma \right\|_F^2 \right]$$

where $$\| \cdot \|_F$$ is the Frobenius norm.[^3] The authors of
the original [OAS paper](https://ieeexplore.ieee.org/document/5484583)
derive the following closed-form formula approximating the
the $$\rho$$ that minimizes the expression above (under a Gaussian
assumption). This approximation is particularly useful when the
number of observations is small. The formula for the OAS
approximation is given by:

$$\hat{\rho}_{OAS} = \frac{(1-2p)\mathrm{Tr}(S^2) + \mathrm{Tr}^2(S)}
{(n+1-2/p) (\mathrm{Tr}(S^2) - \mathrm{Tr}^2(S)/p} $$

where we cap the result at 1 (if the value of the formula
above is larger than one, we set $\hat{\rho}_{OAS}=1$).
 
Let's implement the OAS estimator in PyTorch:

```python
def isotropic_estimator(sample_covariance):
    """Isotropic covariance estimate with same trace as sample.
    
    Arguments:
    ---------- 
    sample_covariance : torch.Tensor
        Sample covariance matrix.
    """
    n_dim = sample_covariance.shape[0]
    return torch.eye(n_dim) * torch.trace(sample_covariance) / n_dim

def oas_shrinkage(sample_covariance, n_samples):
    """Get OAS shrinkage parameter.
    
    Arguments:
    ----------
    sample_covariance : torch.Tensor
        Sample covariance matrix.
    """
    n_dim = sample_covariance.shape[0]
    tr_cov = torch.trace(sample_covariance)
    tr_prod = torch.sum(sample_covariance ** 2)
    shrinkage = (
      (1 - 2 / n_dim) * tr_prod + tr_cov ** 2
    ) / (
      (n_samples + 1 - 2 / n_dim) * (tr_prod - tr_cov ** 2 / n_dim)
    )
    shrinkage = min(1, shrinkage)
    return shrinkage

def oas_estimator(X, assume_centered=False):
    """Oracle Approximating Shrinkage (OAS) covariance estimate.

    Arguments:
    ----------
    X : torch.Tensor
        Data matrix with shape (n_samples, n_features).
    """
    n_samples = X.shape[0]

    # Compute sample covariance
    if not assume_centered:
        sample_covariance = torch.cov(X.T)
    else:
        sample_covariance = X.T @ X / n_samples

    # Compute isotropic estimator F
    isotropic = isotropic_estimator(sample_covariance)

    # Compute OAS shrinkage parameter
    shrinkage = oas_shrinkage(sample_covariance, n_samples)

    # Compute OAS shrinkage covariance estimate
    oas_estimate = (1 - shrinkage) * sample_covariance + shrinkage * isotropic
    return oas_estimate
```

Let's now test the OAS estimator with the same simulations as before,
and compare the MSE of the OAS estimator $$\hat{Sigma}$$ to the MSE of
the sample covariance matrix $$S$$:

```python
n_dim = 10
n_samples = 20
n_reps = 5

# Compute sample covariance matrices
covs_sample = []
covs_oas = []
for _ in range(n_reps):
    X = MultivariateNormal(torch.zeros(n_dim), cov_pop).sample((n_samples,))
    covs_sample.append(torch.cov(X.T))
    covs_oas.append(oas_estimator(X))

# Plot sample covariance matrices
max_val = 1.2
fig, axs = plt.subplots(2, n_reps+1, figsize=(15, 7))
axs[0,0].imshow(cov_pop, cmap='seismic', vmin=-max_val, vmax=max_val)
axs[0,0].set_title("Population cov", fontsize=18)
axs[1,0].imshow(cov_pop, cmap='seismic', vmin=-max_val, vmax=max_val)
axs[1,0].set_title("Population cov", fontsize=18)

for i in range(n_reps):
    axs[0,1+i].imshow(covs_sample[i], cmap='seismic', vmin=-max_val, vmax=max_val)
    axs[0,1+i].set_title(f"Sample cov {i+1}", fontsize=18)
    axs[1,1+i].imshow(covs_oas[i], cmap='seismic', vmin=-max_val, vmax=max_val)
    axs[1,1+i].set_title(f"OAS cov {i+1}", fontsize=18)

plt.tight_layout()
plt.show()

# Compute mean squared error
mse_sample = torch.stack([torch.linalg.norm(cov_pop - cov) ** 2 for cov in covs_sample]).mean()
mse_oas = torch.stack([torch.linalg.norm(cov_pop - cov) ** 2 for cov in covs_oas]).mean()
print(f"MSE sample covariance: {mse_sample}")
print(f"MSE OAS covariance: {mse_oas}")
```

![](/files/blog/shrinkage/cov_oas1.png)

```
MSE sample covariance: 1.9461839199066162
MSE OAS covariance: 0.659830629825592
```

We see that:
- OAS estimator provides a more stable estimate than $$S$$
- OAS estimator diagonal elements are more biased than $$S$$ 
- OAS estimator has lower MSE than $$S$$


## Linear Discriminant Analysis with shrinkage

Let's next see the OAS estimator used in practice, by applying
Linear Discriminant Analysis (LDA) to the MNIST dataset.

LDA is a standard technique for
dimensionality reduction and classification. In a labeled dataset,
LDA learns the filters that maximize the separation between classes
while minimizing the within-class variability. This goal can be
mathematically formulated as follows:

- In a dataset with $$C$$ classes, the between-class scatter matrix
$$S_B = \frac{1}{C}\sum_{i=1}^C (\mu_i - \mu)(\mu_i - \mu)^T$$
(i.e. covariance of the class means centered around the global mean)
indicates how the class means are spread out in the data space
- The within-class scatter matrix
$$S_W = \frac{1}{N} \sum_{i=1}^C \sum_{x \in X_i} (x - \mu_i)(x - \mu_i)^T$$,
where $$N$$ is the total number of samples, and $$X_i$$ is the
set of samples in class $$i$$ indicates how the data points are spread out
around their class means (this is the residual covariance around the
class means)
- Along direction $$w$$, the variance between classes is given by
$$w^T S_B w$$, and the variance within classes is given by
$$w^T S_W w$$
- The directions that maximize between-class separation while
minimizing within-class variance are the directions that maximize
the following ratio:

$$\frac{w^T S_B w}{w^T S_W w}$$

It turns out that the directions $$w$$ that maximize the ratio above
are the eigenvectors of the matrix $$S_W^{-1} S_B$$
(see [here](https://en.wikipedia.org/wiki/Linear_discriminant_analysis#Multiclass_LDA)).
Thus, we can find the LDA filters by computing the two scatter matrices,
inverting $$S_W$$, and computing the eigenvectors of the product $$S_W^{-1} S_B$$. 

The quality of the LDA filters depends on the quality of the
estimates of $$S_W$$ and $$S_B$$. Thus, we can compare the filters
learned when using the sample covariance estimator and the OAS
estimator for $$S_W$$. This gives us a natural way to evaluate
both methods: we can use the classification accuracy of the LDA
linear decoder with the projections learned with both methods
as a measure of the quality of the covariance matrix.

We apply LDA to the MNIST dataset because it has a large number of
dimensions.

Let's first load the MNIST dataset using the `torchvision` package.
Note that we modify the dataset in two ways to better suit our example. 
First, we subsample the number of images, keeping only 2000 pictures so that
the number of observations $$n$$ is close to the number of dimensions $$p$$.
Second, we remove from the learning procedure some pixels that have zero
variance (i.e. they are constant across all images) to avoid
singular covariance matrices.

```python
import torchvision

# Download and load training and test datasets
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

# Reshape images into vectors
n_samples, n_row, n_col = trainset.data.shape
n_dim = trainset.data[0].numel()
x_train = trainset.data.reshape(-1, n_dim).float()
y_train = trainset.targets
x_test = testset.data.reshape(-1, n_dim).float()
y_test = testset.targets

# Subsample data
N_SUBSAMPLE = 2000 # Number of images to keep
rand_idx = torch.randperm(x_train.shape[0])
rand_idx = torch.sort(rand_idx[:N_SUBSAMPLE]).values
x_train = x_train[rand_idx]
y_train = y_train[rand_idx]
x_train = x_train[:N_SUBSAMPLE]
y_train = y_train[:N_SUBSAMPLE]

# Mask pixels with zero variance
mask = x_train.std(dim=0) > 0
x_train = x_train[:, mask]
x_test = x_test[:, mask]

def unmask_image(image):
    """Function to return data vector to original shape."""
    unmasked = torch.zeros(n_dim)
    unmasked[mask] = image
    return unmasked

# Scale data and subtract global mean
def scale_and_center(x_train, x_test):
    std = x_train.std()
    x_train = x_train / std
    x_test = x_test / std
    global_mean = x_train.mean(axis=0, keepdims=True)
    x_train = x_train - global_mean
    x_test = x_test - global_mean
    return x_train, x_test

# Scale data and subtract global mean
x_train, x_test = scale_and_center(x_train, x_test)

# Plot some images
names = y_train.unique().tolist()
n_classes = len(y_train.unique())
fig, ax = plt.subplots(1, n_classes, figsize=(10, 2))
for i in range(n_classes):
    ax[i].imshow(
      unmask_image(x_train[y_train == i][0]).reshape(n_row, n_col), cmap='gray')
    ax[i].axis('off')
    ax[i].set_title(names[i], fontsize=10)
plt.tight_layout()
plt.show()
```

![](/files/blog/shrinkage/mnist.png)

Next, let's compute the scatter matrices $$S_W$$ and $$S_B$$

```python
# Compute the class means
class_means = torch.stack([x_train[y_train == i].mean(dim=0) for i in range(n_classes)])
mu = class_means.mean(dim=0, keepdim=True)

# Compute between-class scatter matrix
between_class = (class_means - mu).T @ (class_means - mu) / n_classes

# Compute the within-class scatter matrix
x_train_centered = x_train - class_means[y_train]
within_class_sample = x_train_centered.T @ x_train_centered / N_SUBSAMPLE
within_class_oas = oas_estimator(x_train_centered, assume_centered=True)
```

Then, we compute the LDA filters obtained from each covariance matrix. We
add a small value to the diagonal of $$S_W$$ for numerical stability:

```python
# Get LDA filters
n_filters = n_classes - 1
def get_lda_filters(between_class, within_class):
    within_class_inv = torch.linalg.inv(within_class)
    lda_mat = within_class_inv @ between_class
    eigvals, eigvecs = torch.linalg.eigh(lda_mat)
    filters = eigvecs[:, -n_filters:]
    return filters.T

# Get the LDA projections
small_reg = torch.eye(within_class_sample.shape[0]) * 1e-7
lda_filters_sample = get_lda_filters(between_class, within_class_sample + small_reg)
lda_filters_oas = get_lda_filters(between_class, within_class_oas + small_reg)
```

Let's now plot the LDA filters obtained with both covariance estimators:

```python
# Plot both LDA filters
fig, axs = plt.subplots(2, 9, figsize=(12, 4))
for i in range(9):
    sample_filter_im = unmask_image(lda_filters_sample[i]).reshape(n_row, n_col)
    oas_filter_im = unmask_image(lda_filters_oas[i]).reshape(n_row, n_col)
    axs[0, i].imshow(sample_filter_im, cmap='gray')
    axs[0, i].axis('off')
    axs[1, i].imshow(oas_filter_im, cmap='gray')
    axs[1, i].axis('off')
plt.tight_layout()
plt.show()
```

![](/files/blog/shrinkage/lda_filters.png)

We see that the filters obtained with the sample covariance estimator are
unstable, in that they put most of their weight into a few pixels. The
OAS filters are noisy, as we could expect from the small number of samples
used, but they are smoother and better distributed across the image.

Finally, we can compute the classification accuracy of the LDA filters

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score

def get_filters_accuracy(filters, x_train, y_train, x_test, y_test):
    x_train_lda = x_train @ filters.T
    x_test_lda = x_test @ filters.T
    lda_classifier = LinearDiscriminantAnalysis()
    lda_classifier.fit(x_train_lda, y_train)
    acc = lda_classifier.score(x_test_lda, y_test)
    return acc

# Compute accuracy of LDA filters
sample_acc = get_filters_accuracy(lda_filters_sample, x_train, y_train, x_test, y_test)
oas_acc = get_filters_accuracy(lda_filters_oas, x_train, y_train, x_test, y_test)
print(f"Sample LDA accuracy: {sample_acc:.2f}")
print(f"OAS LDA accuracy: {oas_acc:.2f}")
```

```
Sample LDA accuracy: 0.79
OAS LDA accuracy: 0.83
```

We see that the OAS estimator provides a better classification accuracy
than the sample covariance estimator. This is a common result in
practice, as the OAS estimator provides a more stable estimate of the
covariance matrix. In fact, there is
[an sklearn tutorial](https://scikit-learn.org/dev/auto_examples/classification/plot_lda.html) comparing
the performance of LDA on simulated data with different shrinkage
estimators. This problem has also been studied in the literature, for
example in the influential paper
[Regularized Discriminant Analysis](https://www.tandfonline.com/doi/abs/10.1080/01621459.1989.10478752).


[^1]: The covariance matrix is symmetric, so the number of unique entries is given by those at and below the diagonal. The first row has $1$ element at/below the diagonal, the second row has $2$, and so on up to the $n$th row, which has $p$ elements. So the number of unique elements equals the sum $1 + 2 + \ldots + p$. Note that adding the first and last element equals $p+1$, adding the second and second-to-last element also equals $p+1$, and so on. So, we have $p/2$ pairs of elements that sum to $p+1$, resulting in the known formula $1 + 2 + \ldots + p = \frac{(p+1)p}{2}$

[^2]: The name Oracle Approximating Shrinkage comes from the fact that there is a formula for the optimal shrinkage coefficient when the true covariance matrix is known, which we can refer to as the oracle value. The OAS estimator approximates this oracle value when the true covariance matrix is unknown.

[^3]: The Frobenius norm of a matrix $A$ is defined as $\| A \|_F = \sqrt{\sum_{i,j} A_{ij}^2}$, which also equal to $\text{tr}(A^T A)$, where $\text{tr}$ is the trace operator.
