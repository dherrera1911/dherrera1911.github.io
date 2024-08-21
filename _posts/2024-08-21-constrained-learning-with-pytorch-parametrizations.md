---
title: 'Easy constrained optimization in Pytorch with Parametrizations'
date: 2024-08-21
permalink: /posts/2024/08/constrained-learning-with-pytorch-parametrizations/
tags:
  - Pytorch
  - Machine Learning
  - Optimization
  - Constrained Optimization
  - Manifolds
---

Often times we want to optimize some model parameter while
keeping it constrained. For example, we might want to optimize
a vector while constraining it to have unit norm,
a set of vectors that are orthogonal, parameters
that we want to be positive, or a covariance
matrix that should be symmetric positive
definite. Parametrizations are a technique to turn a
difficult constrained optimization problem into a simpler
unconstrained optimization problem that can be solved with standard
optimization algorithms.

In this post we will introduce the tool of parametrizations for
the non-expert user, and show how we can implement them in Pytorch
efficiently with a few lines of code, using the Pytorch
Parametrizations tool `torch.nn.utils.parametrize`. We will
see how to constrain a vector to be on the unit circle, and
how to constrain a matrix to be symmetric positive definite.
More in depth information on Pytorch parametrizations can be found in the
[Parametrizations tutorial](https://pytorch.org/tutorials/intermediate/parametrizations.html),
but in contrast to that tutorial, here we focus on
simpler examples and a more general audience.

Constrained optimization
---------------------

When doing optimization in Pytorch (or any
other deep learning library), we have a parameter
$$\theta \in \mathbb{R}^n$$ that we optimize to minimize
some loss function $$L(\theta)$$ using gradient descent.
At each iteration, we compute the negative gradient of the loss
function with respect to the parameter $$\theta$$, and then update
the parameter that direction.
Thus, a typical optimization step looks like this:

$$\theta \leftarrow \theta - \alpha \nabla_{\theta} L(\theta)$$

where $$\alpha$$ is the learning rate and $$\nabla_{\theta} L(\theta)$$
is the gradient of the loss function with respect to $$\theta$$.

In constrained optimization, we additionally constrain
$$\theta$$ to fulfill some condition, so that $$\theta \in C$$
where $$C \subseteq \mathbb{R}^n$$. One simple example
is the unit norm constraint, where we want
to optimize a vector $$\theta$$ such that $$\|\theta\| = 1$$.
In this case, $$\theta \in C$$, where $$C = \{x: \|x \| = 1\}$$,
which is the unit sphere $$S^{n-1}$$. Now we are doing optimization
over the sphere, which is a non-Euclidean space and thus
a more complicated optimization problem.

Example: Average on a circle, unconstrained
---------------------

Let's show a concrete example of such a problem in Pytorch.
We have some data distributed on the unit circle and
we want to find the point $$\theta$$ that has the minimum average
distance to the data. We also want to constrain $$\theta$$
to be on the circle (e.g. we might want the average
'direction' of the data). We will show how to solve this
constrained optimization problem in a while, but first
lets solve the unconstrained version of the problem
to use as a reference later.

First we generate the data by sampling points from a
Gaussian distribution in $$\mathbb{R}^2$$, and projecting
these points onto to the unit sphere by dividing them by
their norm [^1].

```python
import torch

# Simulation parameters
n_dimensions = torch.tensor(2)
mu = torch.ones(n_dimensions) / torch.sqrt(n_dimensions)
sigma = 1
n_points = 200

# Generate Gaussian data
data_gauss = mu + torch.randn(n_points, n_dimensions) * sigma
# Project to the unit sphere
data = data_gauss / data_gauss.norm(dim=1, keepdim=True)
```

We can visualize the data to see that they are in the
unit circle (we add a little bit of jitter for visualization).

```python
import matplotlib.pyplot as plt

def plot_circle_data(ax, data, title):
    ax.scatter(data[:, 0], data[:, 1], alpha=0.4, s=12)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title)

data_jitter = data + 0.01 * torch.randn(n_points, 2)

fig, ax = plt.subplots(figsize=(4, 3.8))
plot_circle_data(ax, data_jitter, "Data on unit circle")
plt.tight_layout()
plt.show()
```

![](/files/blog/parametrizations/data.png)


Next, we generate a class that has as
a parameter the vector ``theta``, and a function
``forward`` that computes the squared distance of each
point to ``theta`` (functions named ``forward`` are called
when we call the call the model object as a
function, i.e. ``model(data)``). We
use the Pytorch `nn.Module` class to define the model [^2].

```python
import torch.nn as nn

class AverageUnconstrained(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize theta randomly
        theta = torch.randn(dim)
        theta = theta / torch.norm(theta)
        # Make theta a parameter so it is optimized by Pytorch
        self.theta = nn.Parameter(theta)

    def forward(self, x):
        # Compute the distance of each point to theta
        difference = x - self.theta
        distance_squared = torch.sum(difference**2, dim=1)
        return distance_squared

```

Next, we define the loss function that just averages the
individual losses of each data point, and a function
`train_model` to perform gradient descent on the model
parameters:

```python
# Loss function
def loss_function(loss_vector):
    loss = torch.mean(loss_vector)
    return loss

# Optimization function
def train_model(model, data, n_iterations=100, lr=0.1):
    # We initialize an optimizer for the model parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # We take n_iterations steps of gradient descent
    for i in range(n_iterations):
        optimizer.zero_grad()
        loss_vector = model(data)
        loss = loss_function(loss_vector)
        loss.backward()
        optimizer.step()
```

We are now done with the setup. Next we optimize the model and
visualize the result. In the case of unconstrained $$\theta$$
we know that the solution is the Euclidean average of the data, which
will not be on the circle.

```python
# Initialize the model
model_unconstrained = AverageUnconstrained(n_dimensions)

# Fit the model
train_model(model_unconstrained, data, n_iterations=100, lr=0.1)

# Visualize the theta learned by the model
theta = model_unconstrained.theta.detach().numpy()

fig, ax = plt.subplots(figsize=(4, 3.8))
plot_circle_data(ax, data_jitter, "Unconstrained theta")
ax.scatter(theta[0], theta[1], color="red", s=20, label="theta")
ax.legend(loc="upper left")
plt.show()
```


![](/files/blog/parametrizations/unconstrained.png)

We see that the learned point is not on the circle, as expected.

Formalism of parametrizations
---------------------

In the previous example, we just updated $$\theta$$ in the direction
of the negative gradient at each iteration. The problem
is that updating the parameter in this way
will lead to a point that is not on the circle. So, how can
we update the parameter in such a way
that it remains on the circle?

An intuitive alternative is to project $$\theta$$ back
onto the unit circle after each update. However, doing this naively
might introduce some problems, such as affecting the
inner workings of the optimization algorithm (e.g.
if there's a momentum term). Also, while
the unit circle is a simple set, other constraints might be
more complicated to project $$\theta$$ onto.

Parametrizations also involve projecting something
onto the set $$C$$, but in a more principled way.
The idea is to introduce a new parameter $$\eta \in \mathbb{R}^m$$
that is unconstrained, and a differentiable function
$$f$$ mapping from values of $$\eta$$ to values of $$\theta$$
in the set $$C$$, $$f: \mathbb{R}^m \rightarrow C$$. Then, we
just apply the optimization algorithm to $$\eta$$ instead
of $$\theta$$, using the following update rule:

$$\eta \leftarrow \eta - \alpha \nabla_{\eta} L\left(f(\eta)\right)$$

Note that we use the same loss function as before, now
composed with the function $$f$$, and we take the gradient
with respect to $$\eta$$. We need $$f$$ to be differentiable
so that we can obtain the gradient with respect to $$\eta$$.

Lets come back to our example of constraining $$\theta$$
to be on the unit circle. We can make $$\eta$$ an unconstrained
2D vector, and define $$f(\eta) = \eta / \| \eta \|$$, which maps
$$\eta$$ onto the unit circle[^3]. Lets see how we can
implement this idea in Pytorch.

Implementation of unit vector parametrization in Pytorch
---------------------

If we wanted to manually include the parametrization
above in our class ``AverageUnconstrained``, we would
have to take into account several considerations.
The new class would have a parameter ``eta``, and
``theta`` would no longer be an ``nn.Parameter``. We
would have to figure out how we want
to update and access ``theta``. If we wanted to assign
a value to ``theta``, we need to go through ``eta``,
which would also require some thinking, particularly
for more complicated parametrizations.

The Pytorch tool `torch.nn.utils.parametrize` does all of
this work for us. We just need to implement the
function $$f$$, and then `parametrize` takes care of the
rest. The only thing we need to do is to implement
$$f$$ in a specific way, as described next.

**How to implement $$f$$ for Pytorch parametrizations**

To use ``parametrize``, we need to define the function inside
an `nn.Module` class, and our function $$f$$ should
be called ``forward`` inside this class. Let's
see how this looks like in our example:

```python
# Define the parametrization function f
class NormalizeVector(nn.Module):
    def forward(self, eta):
        theta = eta / eta.norm()
        return theta
```

The function `forward` implements the function $$f$$
by taking a vector `eta` and returning the normalized
vector `theta` with a length of 1 (the names of the variables
don't have to be `eta` and `theta`).

Next, let's use `parametrize` to create a new class
`AverageInCircle` where parameter `theta` is constrained
to be on the unit circle. This is done by only one line
to our original class ``AverageUnconstrained``:

```python
from torch.nn.utils import parametrize

class AverageInCircle(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # Initialize theta randomly
        theta = torch.randn(dim)
        theta = theta / torch.norm(theta)
        # Make theta a parameter so it is optimized by Pytorch
        self.theta = nn.Parameter(theta)

        ### ONLY CHANGE: Add the parametrization in terms of f to theta
        parametrize.register_parametrization(self, "theta", NormalizeVector())


    def forward(self, x):
        # Compute the distance of each point to theta
        difference = x - self.theta
        distance_squared = torch.sum(difference**2, dim=1)
        return distance_squared
```

Now, the `eta` parameter that is actually being optimized
is taken care of in the background. The code for optimizing
this constrained model is the same as
for the unconstrained model. Let's optimize the model and
see the result.

```python
# Initialize the model
model_circle = AverageInCircle(n_dimensions)

# Fit the model
train_model(model_circle, data, n_iterations=100, lr=0.1)

# Visualize the theta learned by the model
theta_circle = model_circle.theta.detach()

fig, ax = plt.subplots(figsize=(4, 3.8))
plot_circle_data(ax, data_jitter, "Parametrized theta")
ax.scatter(theta_circle[0], theta_circle[1], color="red", s=20, label="theta")
ax.legend(loc="upper left")
plt.show()
```

![](/files/blog/parametrizations/constrained.png)

Now in this parametrized model the optimized ``theta`` is on the unit
circle.

Implementation of symmetric positive definite parametrization in Pytorch
---------------------

Let's look at a more complicated example that appears often
in statistics and machine learning: optimization of a matrix
constrained to be symmetric positive definite
(SPD). This is a common problem because
covariance matrices and many other
important mathematical objects are SPD matrices.

SPD matrices are symmetric matrices whose eigenvalues are
all positive. There is no simple condition on the matrix entries
to ensure that it is SPD. We could think of parametrizing
the matrix in terms of its eigenvalues and eigenvectors and
constrain the eigenvalues to be positive, but this would lead
to more parameters than the original matrix, and other potential
problems, such as having to implement orthogonal constraints
on the eigenvectors.

There are some well-known parametrizations for SPD matrices,
and we will show how to implement two of them in Pytorch:
the Log-Cholesky parametrization, and the matrix logarithm
parametrization (see the article
[Unconstrained parametrizations for variance-covariance matrices](https://link.springer.com/article/10.1007/BF00140873) for an overview).
Remember that to define the parametrization we need to define
the function $$f$$ that maps from the unconstrained parameter
to the SPD matrix.

**Log-Cholesky parametrization**

The Log-Cholesky parametrization uses the property of
SPD matrices that they can be decomposed as $$\Sigma = LL^T$$,
where $$L$$ is a lower triangular matrix with
positive diagonal elements (in this section we refer
to our model parameter as $$\Sigma$$ instead of $$\theta$$).
This is called the Cholesky decomposition of $\Sigma$,
and it is unique.

We could think of parametrizing SPD matrices in terms of
the lower triangular matrix $$L$$, but $$L$$ is still
constrained to have positive diagonal elements.
We can get rid of this positive constraint by
taking the logarithm of the diagonal elements of $$L$$,
resulting in an unconstrained lower-triangular
matrix $$M$$. Let's express this idea in terms of the
function $$f$$.

Let $$\lfloor M \rfloor$$ denote the matrix with only
the strictly lower-triangular part of lower-triangular
matrix $$M$$, and $$\mathbb{D}(M)$$ the diagonal matrix
with the diagonal elements of $$M$$. Thus, we have that
$$M = \mathbb{D}(M) + \lfloor M \rfloor$$ and
$$L = \exp(\mathbb{D}(M)) + (\lfloor M \rfloor)$$ (note
that to take the exponential of a diagonal matrix we just 
take the exponential of the diagonal elements).
Then, the function $$f$$ that maps from the unconstrained
space of lower triangular matrices
$$\mathbb{R}^{\frac{n(n+1)}{2}}$$[^4] to the
SPD matrices is defined as follows:

$$f(M) = \left( \exp(\mathbb{D}(M)) + \lfloor M \rfloor \right) \left( \exp(\mathbb{D}(M)) + \lfloor M \rfloor \right)^T = L L^T = \Sigma$$

($$M$$ is equivalent to the $$\eta$$ from the previous section,
it is the unconstrained parameter).
Thus, we have a function that we can use to define our
parametrization.

Before we implement this in Pytorch,
we should note that there is one thing that
``parametrize`` needs some more help with: assigning
a value to the parameter $$M$$.

**Right-inverse function for assigning the constrained parameter**

Imagine that we implemented our model using the Log-Cholesky
parametrization, and that we want to assign a given value
to $$\Sigma$$ (for example, we might want to initialize it
a certain way). However, behind the curtain
there is the parameter $$M$$ that gives us $$\Sigma$$
by $$f(M) = \Sigma$$. To assign a certain value to
$$\Sigma$$, we actually need to assign a value to $$M$$
such that $$f(M) = \Sigma$$. The `parametrize` tool
does not know how to do this automatically, so we
need to define a function that
maps from $$\Sigma$$ to $$M$$ (it's a sort of inverse
of $$f$$, although it doesn't need to mathematically
be an inverse, since $$f$$ might not be injective, like
in the circle constrain example). We do
this by defining a function called ``right_inverse`` inside
the class that implements the parametrization.

We already described this right-inverse function when we
explained how to go from $$\Sigma$$ to $$M$$:
1) Take the Cholesky decomposition of $$\Sigma$$ to get $$L$$
2) Take the logarithm of the diagonal elements of $$L$$ to get $$M$$

Let's now implement both the parametrization function $$f$$
and the right-inverse function for the Log-Cholesky
parametrization in Pytorch:

```python
# Log-Cholesky parametrization
class SPDLogCholesky(nn.Module):
    def forward(self, M):
        # Take strictly lower triangular matrix
        M_strict = M.tril(diagonal=-1)
        # Make matrix with exponentiated diagonal
        D = M.diag()
        # Make the Cholesky decomposition matrix
        L = M_strict + torch.diag(torch.exp(D))
        # Invert the Cholesky decomposition
        Sigma = torch.matmul(L, L.t())
        return Sigma

    def right_inverse(self, Sigma):
        # Compute the Cholesky decomposition
        L = torch.linalg.cholesky(Sigma)
        # Take strictly lower triangular matrix
        M_strict = L.tril(diagonal=-1)
        # Take the logarithm of the diagonal
        D = torch.diag(torch.log(L.diag()))
        # Return the log-Cholesky parametrization
        M = M_strict + D
        return M
```

We are now ready to implement a model that optimizes a matrix
constrained to be SPD using the Log-Cholesky parametrization.
Let's now set up a problem where we want to do such optimization
and implement the model.

**Estimating a covariance matrix with missing data**

We will use a problem suggested by a user at
[CrossValidated](https://stats.stackexchange.com/a/653094/134438).
The problem is estimating the covariance matrix of a dataset
where some observations are missing.

We have a dataset $$X$$ that is a matrix with $$n$$ rows
and $$p$$ columns, where each row is an observation and
each column is a variable. Our dataset is missing
some variables for some rows, that is, some entries
$$X_{ij}$$ are missing. To estimate any given element
$$\Sigma_{kl}$$ of the covariance of $$X$$, we could
use the rows where both columns $$k$$ and $$l$$ are
observed. However, this procedure does not guarantee
that the resulting matrix is SPD. We will estimate
the covariance matrix for such a missing-data problem
using unconstrained optimization and then using
the Log-Cholesky parametrization.

Next, we generate a dataset for this problem. First we
generate a mean ``mu_true`` and a covariance matrix
``Sigma_true`` (we generate
the covariance matrix starting from a random
lower-triangular matrix and then applying the function
$$f$$ to obtain an SPD matrix).

```python
# Set random seed
torch.manual_seed(1911)

# Generate the distribution parameters
n_dimensions = torch.tensor(5)
mu_true = torch.ones(n_dimensions)
M = torch.randn(n_dimensions, n_dimensions)
log_chol_par = SPDLogCholesky()
Sigma_true = log_chol_par.forward(M)
```

Next, we generate the data by sampling from a multivariate
Gaussian with mean `mu_true` and covariance `Sigma_true`,
and then we set some entries to be missing.

```python
from torch.distributions import MultivariateNormal

# Generate data
n_points = 200
data = MultivariateNormal(mu_true, Sigma_true).sample((n_points,))

# Remove random datapoints
mask = torch.rand(n_points, n_dimensions) > 0.2
# Make data where mask is False NaN
data[~mask] = float("nan")

# Print the first 8 rows of the data
print(data[:8])
```

```
tensor([[ 1.1146,  0.4793,  1.4138,  0.7827,     nan],
        [-0.9918,     nan,  0.4070,  1.7971,  1.4161],
        [ 1.3888,  1.8095,  0.9842,  0.4521, -1.9231],
        [ 0.9323, -0.0999,     nan,     nan, -0.9474],
        [ 2.3576,  1.7566,     nan,  0.3555,  0.7441],
        [ 1.8579,  1.3301,  1.1172, -0.0374,  2.5136],
        [ 1.0750,  1.2708,     nan, -0.4027,     nan],
        [ 0.7755,  2.7918,  1.0426,  1.2220,     nan]])
```

We will now implement models that compute the log-likelihood
of the dataset under a Gaussian distribution with parameters
``mu`` and ``Sigma``. How we will compute the log-likelihood
of an observation with missing entries is by using only
the values of ``mu`` and ``Sigma`` corresponding to the observed
entries. That is, if a row has missing values for the columns $$1$$
and $$3$$, we will remove the elements $$1$$ and $$3$$ from the ``mu``
and the rows and columns $$1$$ and $$3$$ from the ``Sigma``,
and we will use the remaining elements to compute the
log-likelihood with a Gaussian distribution of lower dimension.

First we implement three useful functions: one that computes the
log-likelihood of a data point under a Gaussian distribution,
one that takes as input the statistics ``mu`` and ``Sigma``,
and returns the statistics with only the elements corresponding
to the observed data, and one that computes the negative
log-likelihood of the data as described in the previous
paragraph.

```python
def gaussian_log_likelihood(data, mu, Sigma):
    # Compute the log likelihood of a single
    # data point under a Gaussian distribution
    n_dimensions = data.shape[0]
    # Subtract the mean from the data
    diff = data - mu
    # Compute the quadratic term
    quadratic = torch.einsum('i,ij,j->', diff, Sigma.inverse(), diff)
    # Compute the gaussian log-likelihood
    log_likelihood = -0.5 * (n_dimensions * torch.log(torch.tensor(2.0 * torch.pi)) \
                             + torch.slogdet(Sigma)[1] \
                             + quadratic)
    return log_likelihood


def remove_nan_statistics(mu, Sigma, nan_indices):
    # Remove the missing value elements from the mean and covariance
    mu_no_nan = mu[~nan_indices]
    Sigma_no_nan = Sigma[~nan_indices][:, ~nan_indices]
    return mu_no_nan, Sigma_no_nan


def nll_observed_data(mu, Sigma, data):
    # Compute the negative log-likelihood of the data under the
    # Gaussian distribution, ignoring NaN values
    ll = torch.zeros(data.shape[0])
    for i in range(data.shape[0]):
        # Get NaN indices for this row
        nan_indices = torch.isnan(data[i])
        # Remove the NaN columns from the statistics
        mu_no_nan, Sigma_no_nan = remove_nan_statistics(mu,
                                                        Sigma,
                                                        nan_indices)
        # Remove NaN columns from the data
        data_no_nan = data[i][~nan_indices]
        # Compute the likelihood of the observed data
        ll[i] = gaussian_log_likelihood(data_no_nan,
                                        mu_no_nan,
                                        Sigma_no_nan)
    return -ll
```

Now, we implement the model that computes the negative log-likelihood
as described above, and we don't constrain ``Sigma`` to be
SPD:

```python
# Create class that computes the negative log-likelihood of data
# with missing values under a Gaussian distribution.

class NLLObserved(nn.Module):
    def __init__(self, mu, Sigma):
        super().__init__()
        self.mu = nn.Parameter(mu.clone())
        self.Sigma = nn.Parameter(Sigma.clone())

    def forward(self, data):
        nll = nll_observed_data(self.mu, self.Sigma, data)
        return nll
```

We then implement the model that uses the Log-Cholesky parametrization,
which only requires adding one line to the previous model:

```python
class NLLObservedCholesky(nn.Module):
    def __init__(self, mu, Sigma):
        super().__init__()
        self.mu = nn.Parameter(mu.clone())
        self.Sigma = nn.Parameter(Sigma.clone())
        parametrize.register_parametrization(self, "Sigma", SPDLogCholesky())

    def forward(self, data):
        nll = nll_observed_data(self.mu, self.Sigma, data)
        return nll

```

We are now ready to optimize the models and compare the results.
Because of how we set up the functions, we can use the same
`train_model` function as in the previous example.

```python
# Generate initial parameters
mu_init = torch.zeros(n_dimensions)
Sigma_init = torch.eye(n_dimensions)

# Initialize the models
model_unconstrained = NLLObserved(mu_init, Sigma_init)
model_cholesky = NLLObservedCholesky(mu_init, Sigma_init)

# Fit the models
train_model(model_unconstrained, data, n_iterations=100, lr=0.1)
train_model(model_cholesky, data, n_iterations=100, lr=0.1)
```

Let's first check whether the covariance matrices are
SPD for each model:

```python
eigenvalues_unconstrained = torch.linalg.eigh(model_unconstrained.Sigma.detach())
eigenvalues_cholesky = torch.linalg.eigh(model_cholesky.Sigma.detach())

print("Minimum eigenvalue unconstrained model:", eigenvalues_unconstrained[0].min())
print("Minimum eigenvalue Cholesky model:", eigenvalues_cholesky[0].min())
```

```
Minimum eigenvalue unconstrained model: tensor(-3.0754)
Minimum eigenvalue Cholesky model: tensor(0.4976)
```

The covariance matrix of the unconstrained model is not
positive definite, while the covariance matrix of the
Log-Cholesky parametrized model is positive definite.
Let's see how the estimated covariances compare to the true
covariance matrix:

```python
# Visualize the learned Sigmas
fig, ax = plt.subplots(1, 3, figsize=(9, 3))

vmax = torch.max(torch.abs(Sigma_true))
vmin = -vmax

# True Sigma
im0=ax[0].imshow(Sigma_true, cmap="coolwarm", vmin=vmin, vmax=vmax)
ax[0].set_title(r"True $\Sigma$")
# Learned Sigma
ax[1].imshow(model_unconstrained.Sigma.detach(), cmap="coolwarm", vmin=vmin, vmax=vmax)
ax[1].set_title(r"Unconstrained $\Sigma$")
# Learned Sigma with Cholesky parametrization
ax[2].imshow(model_cholesky.Sigma.detach(), cmap="coolwarm", vmin=vmin, vmax=vmax)
ax[2].set_title(r"Log-Cholesky parametrized $\Sigma$")
# Add a colorbar to the right of the subplots
cbar = fig.colorbar(im0, ax=ax.ravel().tolist(), shrink=0.95,
                    cax=plt.axes([0.93, 0.15, 0.02, 0.7]))

plt.show()
```
![](/files/blog/parametrizations/covariances.png)

We see that not only is the covariance matrix of the constrained
model positive definite, but it is also much closer to the
true covariance matrix than the unconstrained model!

**Matrix logarithm parametrization**

To have some Linear Algebra fun, let's also implement the
matrix logarithm parametrization. We will not go into detail on
this parametrization, or use it to solve our problem, but
just show how to implement it in Pytorch.

The logarithm and exponential of a matrix are defined in terms of
[series of matrix powers](https://en.wikipedia.org/wiki/Logarithm_of_a_matrix).
For invertible matrices however (like SPD matrices), the matrix logarithm
and exponential can be obtained from the eigenvalue decomposition.
If $$A = U \Lambda U^{-1}$$ is the eigenvalue decomposition of $$A$$,
then $$\log(A) = U \log(\Lambda) U^{-1}$$, where $$\log(\Lambda)$$
is the diagonal matrix with the logarithm of the eigenvalues of $$A$$.

While an SPD matrix will have positive eigenvalues, the matrix logarithm
of an SPD matrix will not necessarily have positive eigenvalues.
In fact, the matrix logarithm of an SPD matrix will be a symmetric matrix.
Thus, the matrix logarithm function maps from the non-linear space of
SPD matrices to the vector space of symmetric matrices. Since the
matrix exponential is the inverse of the matrix logarithm, the matrix
exponential function maps from the vector space of symmetric matrices
to the non-linear space of SPD matrices.

From the above, we see that we can parametrize SPD matrices in terms of
symmetric matrices (more specifically, the non-redundant elements
of symmetric matrices), and we can use as a function $$f$$ the
matrix exponential and as the right-inverse the matrix logarithm.
Let's implement this in Pytorch:

```python

import scipy # Scipy has the matrix logarithm function

# Define positive scalar constraint
def symmetric(X):
    # Use upper triangular part to construct symmetric matrix
    return X.triu() + X.triu(1).transpose(0, 1)

class SPDMatrixExp(nn.Module):
    def forward(self, X):
        # Make symmetric matrix and exponentiate
        SPD = torch.linalg.matrix_exp(symmetric(X))
        return SPD

    def right_inverse(self, SPD):
        # Take logarithm of matrix
        symmetric = scipy.linalg.logm(SPD.numpy())
        X = torch.triu(torch.tensor(symmetric))
        X = torch.as_tensor(X, dtype=dtype)
        return X
```


Optimization on manifolds
---------------------

Many constrained optimization problems can be seen as
optimization on manifolds, particularly
[matrix manifolds](https://sites.uclouvain.be/absil/amsbook/).
The sphere and the SPD matrices examples we showed are
examples of manifolds. The reader interested in other examples
of how to use parametrizations for constrained optimization
can check the package [``geotorch``](https://github.com/lezcano/geotorch)
for optimization in manifolds, which implements several parametrizations
for manifolds in Pytorch. Curiously, the developer of ``geotorch``,
Mario Lezcano, is the same person who developed the Pytorch
parametrizations tool we used in this post, and who wrote the
Parametrizations tutorial in the Pytorch documentation. Thank you
Mario for making time to chat about manifolds with me!


[^1]: Of interest, the distribution on the sphere obtained by projecting a Gaussian distribution is known as the Angular Gaussian or Projected Normal distribution.

[^2]: A Pytorch `nn.Module` is a Pytorch class that has several methods that are useful for defining and optimizing models. By creating our class with the call `class MyClass(nn.Module):`, the class inherits these methods from `nn.Module`. These then come in handy for example to define parameters with `nn.Parameter`, or to pass the model to an optimizer.

[^3]: Note that actually, the function $$f(\eta) = \eta/ \| \eta \|$$ does not map from $$\mathbb{R}^m$$ to the unit circle, because it is not defined at 0. However, in practice this is not generally a problem, as $\eta$ will not generally be driven towards 0 in this setup.

[^4]: Note that although $$M$$ is a matrix, it is equivalent to an unconstrained vector in $$\mathbb{R}^{\frac{n(n+1)}{2}}$$ having its lower-triangular elements, and it is unconstrained


