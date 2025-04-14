Okay, here is the content from the provided PDF pages converted into Markdown format.

---

*Section 10.1*

and so we can transform the model into an equivalent standard HMM having a single chain of latent variables each of which has $K^M$ latent states. We can then run the standard forward-backward recursions in the E step. This has computational complexity $O(NK^{2M})$ that is exponential in the number $M$ of latent chains and so will be intractable for anything other than small values of $M$. One solution would be to use sampling methods (discussed in Chapter 11). As an elegant deterministic alternative, Ghahramani and Jordan (1997) exploited variational inference techniques to obtain a tractable algorithm for approximate inference. This can be done using a simple variational posterior distribution that is fully factorized with respect to the latent variables, or alternatively by using a more powerful approach in which the variational distribution is described by independent Markov chains corresponding to the chains of latent variables in the original model. In the latter case, the variational inference algorithms involves running independent forward and backward recursions along each chain, which is computationally efficient and yet is also able to capture correlations between variables within the same chain.

Clearly, there are many possible probabilistic structures that can be constructed according to the needs of particular applications. Graphical models provide a general technique for motivating, describing, and analysing such structures, and variational methods provide a powerful framework for performing inference in those models for which exact solution is intractable.

## 13.3. Linear Dynamical Systems

In order to motivate the concept of linear dynamical systems, let us consider the following simple problem, which often arises in practical settings. Suppose we wish to measure the value of an unknown quantity $z$ using a noisy sensor that returns a observation $x$ representing the value of $z$ plus zero-mean Gaussian noise. Given a single measurement, our best guess for $z$ is to assume that $z = x$. However, we can improve our estimate for $z$ by taking lots of measurements and averaging them, because the random noise terms will tend to cancel each other. Now let's make the situation more complicated by assuming that we wish to measure a quantity $z$ that is changing over time. We can take regular measurements of $x$ so that at some point in time we have obtained $x_1,...,x_N$ and we wish to find the corresponding values $z_1,..., z_N$. If we simply average the measurements, the error due to random noise will be reduced, but unfortunately we will just obtain a single averaged estimate, in which we have averaged over the changing value of $z$, thereby introducing a new source of error.

Intuitively, we could imagine doing a bit better as follows. To estimate the value of $z_N$, we take only the most recent few measurements, say $x_{N-L}, ..., x_N$ and just average these. If $z$ is changing slowly, and the random noise level in the sensor is high, it would make sense to choose a relatively long window of observations to average. Conversely, if the signal is changing quickly, and the noise levels are small, we might be better just to use $x_N$ directly as our estimate of $z_N$. Perhaps we could do even better if we take a weighted average, in which more recent measurements make a greater contribution than less recent ones.

Although this sort of intuitive argument seems plausible, it does not tell us how to form a weighted average, and any sort of hand-crafted weighing is hardly likely to be optimal. Fortunately, we can address problems such as this much more systematically by defining a probabilistic model that captures the time evolution and measurement processes and then applying the inference and learning methods developed in earlier chapters. Here we shall focus on a widely used model known as a *linear dynamical system*.

As we have seen, the HMM corresponds to the state space model shown in Figure 13.5 in which the latent variables are discrete but with arbitrary emission probability distributions. This graph of course describes a much broader class of probability distributions, all of which factorize according to (13.6). We now consider extensions to other distributions for the latent variables. In particular, we consider continuous latent variables in which the summations of the sum-product algorithm become integrals. The general form of the inference algorithms will, however, be the same as for the hidden Markov model. It is interesting to note that, historically, hidden Markov models and linear dynamical systems were developed independently. Once they are both expressed as graphical models, however, the deep relationship between them immediately becomes apparent.

One key requirement is that we retain an efficient algorithm for inference which is linear in the length of the chain. This requires that, for instance, when we take a quantity $\hat{\alpha}(z_{n-1})$, representing the posterior probability of $z_{n-1}$ given observations $x_1,..., x_{n-1}$, and multiply by the transition probability $p(z_n|z_{n-1})$ and the emission probability $p(x_n|z_n)$ and then marginalize over $z_{n-1}$, we obtain a distribution over $z_n$ that is of the same functional form as that over $\hat{\alpha}(z_{n-1})$. That is to say, the distribution must not become more complex at each stage, but must only change in its parameter values. Not surprisingly, the only distributions that have this property of being closed under multiplication are those belonging to the exponential family.

Here we consider the most important example from a practical perspective, which is the Gaussian. In particular, we consider a linear-Gaussian state space model so that the latent variables $\{z_n\}$, as well as the observed variables $\{x_n\}$, are multivariate Gaussian distributions whose means are linear functions of the states of their parents in the graph. We have seen that a directed graph of linear-Gaussian units is equivalent to a joint Gaussian distribution over all of the variables. Furthermore, marginals such as $\hat{\alpha}(z_n)$ are also Gaussian, so that the functional form of the messages is preserved and we will obtain an efficient inference algorithm. By contrast, suppose that the emission densities $p(x_n|z_n)$ comprise a mixture of $K$ Gaussians each of which has a mean that is linear in $z_n$. Then even if $\hat{\alpha}(z_1)$ is Gaussian, the quantity $\hat{\alpha}(z_2)$ will be a mixture of $K$ Gaussians, $\hat{\alpha}(z_3)$ will be a mixture of $K^2$ Gaussians, and so on, and exact inference will not be of practical value.

We have seen that the hidden Markov model can be viewed as an extension of the mixture models of Chapter 9 to allow for sequential correlations in the data. In a similar way, we can view the linear dynamical system as a generalization of the continuous latent variable models of Chapter 12 such as probabilistic PCA and factor analysis. Each pair of nodes $\{z_n, x_n\}$ represents a linear-Gaussian latent variable model for that particular observation. However, the latent variables $\{z_n\}$ are no longer treated as independent but now form a Markov chain.

Because the model is represented by a tree-structured directed graph, inference problems can be solved efficiently using the sum-product algorithm. The forward recursions, analogous to the $\alpha$ messages of the hidden Markov model, are known as the *Kalman filter* equations (Kalman, 1960; Zarchan and Musoff, 2005), and the backward recursions, analogous to the $\beta$ messages, are known as the *Kalman smoother* equations, or the *Rauch-Tung-Striebel (RTS) equations* (Rauch et al., 1965). The Kalman filter is widely used in many real-time tracking applications.

Because the linear dynamical system is a linear-Gaussian model, the joint distribution over all variables, as well as all marginals and conditionals, will be Gaussian. It follows that the sequence of individually most probable latent variable values is the same as the most probable latent sequence. There is thus no need to consider the analogue of the Viterbi algorithm for the linear dynamical system.

**Exercise 13.19**

Because the model has linear-Gaussian conditional distributions, we can write the transition and emission distributions in the general form
$$
p(z_n|z_{n-1}) = \mathcal{N}(z_n | A z_{n-1}, \Gamma) \quad (13.75)
$$
$$
p(x_n|z_n) = \mathcal{N}(x_n | C z_n, \Sigma). \quad (13.76)
$$
The initial latent variable also has a Gaussian distribution which we write as
$$
p(z_1) = \mathcal{N}(z_1 | \mu_0, V_0). \quad (13.77)
$$
Note that in order to simplify the notation, we have omitted additive constant terms from the means of the Gaussians. In fact, it is straightforward to include them if desired. Traditionally, these distributions are more commonly expressed in an equivalent form in terms of noisy linear equations given by
$$
z_n = A z_{n-1} + w_n \quad (13.78)
$$
$$
x_n = C z_n + v_n \quad (13.79)
$$
$$
z_1 = \mu_0 + u \quad (13.80)
$$
where the noise terms have the distributions
$$
w \sim \mathcal{N}(w|0, \Gamma) \quad (13.81)
$$
$$
v \sim \mathcal{N}(v|0, \Sigma) \quad (13.82)
$$
$$
u \sim \mathcal{N}(u|0, V_0). \quad (13.83)
$$

**Exercise 13.24**

The parameters of the model, denoted by $\theta = \{A, \Gamma, C, \Sigma, \mu_0, V_0\}$, can be determined using maximum likelihood through the EM algorithm. In the E step, we need to solve the inference problem of determining the local posterior marginals for the latent variables, which can be solved efficiently using the sum-product algorithm, as we discuss in the next section.

### 13.3.1 Inference in LDS

We now turn to the problem of finding the marginal distributions for the latent variables conditional on the observation sequence. For given parameter settings, we also wish to make predictions of the next latent state $z_n$ and of the next observation $x_n$ conditioned on the observed data $x_1,..., x_{n-1}$ for use in real-time applications. These inference problems can be solved efficiently using the sum-product algorithm, which in the context of the linear dynamical system gives rise to the Kalman filter and Kalman smoother equations.

It is worth emphasizing that because the linear dynamical system is a linear-Gaussian model, the joint distribution over all latent and observed variables is simply a Gaussian, and so in principle we could solve inference problems by using the standard results derived in previous chapters for the marginals and conditionals of a multivariate Gaussian. The role of the sum-product algorithm is to provide a more efficient way to perform such computations.

Linear dynamical systems have the identical factorization, given by (13.6), to hidden Markov models, and are again described by the factor graphs in Figures 13.14 and 13.15. Inference algorithms therefore take precisely the same form except that summations over latent variables are replaced by integrations. We begin by considering the forward equations in which we treat $z_N$ as the root node, and propagate messages from the leaf node $h(z_1)$ to the root. From (13.77), the initial message will be Gaussian, and because each of the factors is Gaussian, all subsequent messages will also be Gaussian. By convention, we shall propagate messages that are normalized marginal distributions corresponding to $p(z_n |x_1,..., x_n)$, which we denote by
$$
\hat{\alpha}(z_n) = \mathcal{N}(z_n | \mu_n, V_n). \quad (13.84)
$$
This is precisely analogous to the propagation of scaled variables $\hat{\alpha}(z_n)$ given by (13.59) in the discrete case of the hidden Markov model, and so the recursion equation now takes the form
$$
c_n \hat{\alpha}(z_n) = p(x_n|z_n) \int \hat{\alpha}(z_{n-1}) p(z_n|z_{n-1}) dz_{n-1}. \quad (13.85)
$$
Substituting for the conditionals $p(z_n|z_{n-1})$ and $p(x_n|z_n)$, using (13.75) and (13.76), respectively, and making use of (13.84), we see that (13.85) becomes
$$
c_n \mathcal{N}(z_n|\mu_n, V_n) = \mathcal{N}(x_n|Cz_n, \Sigma) \int \mathcal{N}(z_n|Az_{n-1}, \Gamma) \mathcal{N}(z_{n-1}|\mu_{n-1}, V_{n-1}) dz_{n-1}. \quad (13.86)
$$
Here we are supposing that $\mu_{n-1}$ and $V_{n-1}$ are known, and by evaluating the integral in (13.86), we wish to determine values for $\mu_n$ and $V_n$. The integral is easily evaluated by making use of the result (2.115), from which it follows that
$$
\int \mathcal{N}(z_n|Az_{n-1}, \Gamma) \mathcal{N}(z_{n-1}|\mu_{n-1}, V_{n-1}) dz_{n-1} = \mathcal{N}(z_n | A\mu_{n-1}, P_{n-1}) \quad (13.87)
$$
where we have defined
$$
P_{n-1} = AV_{n-1}A^T + \Gamma. \quad (13.88)
$$
We can now combine this result with the first factor on the right-hand side of (13.86) by making use of (2.115) and (2.116) to give
$$
\mu_n = A\mu_{n-1} + K_n (x_n - CA\mu_{n-1}) \quad (13.89)
$$
$$
V_n = (I - K_nC)P_{n-1} \quad (13.90)
$$
$$
c_n = \mathcal{N}(x_n|CA\mu_{n-1}, CP_{n-1}C^T + \Sigma). \quad (13.91)
$$
Here we have made use of the matrix inverse identities (C.5) and (C.7) and also defined the *Kalman gain matrix*
$$
K_n = P_{n-1}C^T (CP_{n-1}C^T + \Sigma)^{-1}. \quad (13.92)
$$
Thus, given the values of $\mu_{n-1}$ and $V_{n-1}$, together with the new observation $x_n$, we can evaluate the Gaussian marginal for $z_n$ having mean $\mu_n$ and covariance $V_n$, as well as the normalization coefficient $c_n$.
The initial conditions for these recursion equations are obtained from
$$
c_1 \hat{\alpha}(z_1) = p(z_1) p(x_1|z_1). \quad (13.93)
$$
Because $p(z_1)$ is given by (13.77), and $p(x_1|z_1)$ is given by (13.76), we can again make use of (2.115) to calculate $c_1$ and (2.116) to calculate $\mu_1$ and $V_1$ giving
$$
\mu_1 = \mu_0 + K_1 (x_1 - C\mu_0) \quad (13.94)
$$
$$
V_1 = (I - K_1C)V_0 \quad (13.95)
$$
$$
c_1 = \mathcal{N}(x_1|C\mu_0, CV_0C^T + \Sigma) \quad (13.96)
$$
where
$$
K_1 = V_0 C^T (CV_0C^T + \Sigma)^{-1}. \quad (13.97)
$$
Similarly, the likelihood function for the linear dynamical system is given by (13.63) in which the factors $c_n$ are found using the Kalman filtering equations.
We can interpret the steps involved in going from the posterior marginal over $z_{n-1}$ to the posterior marginal over $z_n$ as follows. In (13.89), we can view the quantity $A\mu_{n-1}$ as the prediction of the mean over $z_n$ obtained by simply taking the mean over $z_{n-1}$ and projecting it forward one step using the transition probability matrix $A$. This predicted mean would give a predicted observation for $x_n$ given by $CAz_{n-1}$ obtained by applying the emission probability matrix $C$ to the predicted hidden state mean. We can view the update equation (13.89) for the mean of the hidden variable distribution as taking the predicted mean $A\mu_{n-1}$ and then adding a correction that is proportional to the error $x_n - CA\mu_{n-1}$ between the predicted observation and the actual observation. The coefficient of this correction is given by the Kalman gain matrix. Thus we can view the Kalman filter as a process of making successive predictions and then correcting these predictions in the light of the new observations. This is illustrated graphically in Figure 13.21.

```
[Image placeholder: Figure 13.21 showing three Gaussian curves illustrating prediction and update steps in LDS]
```
**Figure 13.21** The linear dynamical system can be viewed as a sequence of steps in which increasing uncertainty in the state variable due to diffusion is compensated by the arrival of new data. In the left-hand plot, the blue curve shows the distribution $p(z_{n-1}|x_1,..., x_{n-1})$, which incorporates all the data up to step $n-1$. The diffusion arising from the nonzero variance of the transition probability $p(z_n|z_{n-1})$ gives the distribution $p(z_n|x_1,..., x_{n-1})$, shown in red in the centre plot. Note that this is broader and shifted relative to the blue curve (which is shown dashed in the centre plot for comparison). The next data observation $x_n$ contributes through the emission density $p(x_n|z_n)$, which is shown as a function of $z_n$ in green on the right-hand plot. Note that this is not a density with respect to $z_n$ and so is not normalized to one. Inclusion of this new data point leads to a revised distribution $p(z_n|x_1,...,x_n)$ for the state density shown in blue. We see that observation of the data has shifted and narrowed the distribution compared to $p(z_n|x_1,..., x_{n-1})$ (which is shown in dashed in the right-hand plot for comparison).

**Exercise 13.27**

If we consider a situation in which the measurement noise is small compared to the rate at which the latent variable is evolving, then we find that the posterior distribution for $z_n$ depends only on the current measurement $x_n$, in accordance with the intuition from our simple example at the start of the section. Similarly, if the latent variable is evolving slowly relative to the observation noise level, we find that the posterior mean for $z_n$ is obtained by averaging all of the measurements obtained up to that time.

**Exercise 13.28**

One of the most important applications of the Kalman filter is to tracking, and this is illustrated using a simple example of an object moving in two dimensions in Figure 13.22.

So far, we have solved the inference problem of finding the posterior marginal for a node $z_n$ given observations from $x_1$ up to $x_n$. Next we turn to the problem of finding the marginal for a node $z_n$ given all observations $x_1$ to $x_N$. For temporal data, this corresponds to the inclusion of future as well as past observations. Although this cannot be used for real-time prediction, it plays a key role in learning the parameters of the model. By analogy with the hidden Markov model, this problem can be solved by propagating messages from node $x_N$ back to node $x_1$ and combining this information with that obtained during the forward message passing stage used to compute the $\hat{\alpha}(z_n)$.

In the LDS literature, it is usual to formulate this backward recursion in terms of $\gamma(z_n) = \hat{\alpha}(z_n)\beta(z_n)$ rather than in terms of $\beta(z_n)$. Because $\gamma(z_n)$ must also be Gaussian, we write it in the form
$$
\gamma(z_n) = \hat{\alpha}(z_n)\beta(z_n) = \mathcal{N}(z_n|\hat{\mu}_n, \hat{V}_n). \quad (13.98)
$$
To derive the required recursion, we start from the backward recursion (13.62) for $\beta(z_n)$, which, for continuous latent variables, can be written in the form
$$
c_{n+1}\beta(z_n) = \int \beta(z_{n+1}) p(x_{n+1}|z_{n+1}) p(z_{n+1}|z_n) d z_{n+1}. \quad (13.99)
$$

```
[Image placeholder: Figure 13.22 showing tracking of a moving object with true positions, noisy measurements, and Kalman filter estimates with covariance ellipses]
```
**Figure 13.22** An illustration of a linear dynamical system being used to track a moving object. The blue points indicate the true positions of the object in a two-dimensional space at successive time steps, the green points denote noisy measurements of the positions, and the red crosses indicate the means of the inferred posterior distributions of the positions obtained by running the Kalman filtering equations. The covariances of the inferred positions are indicated by the red ellipses, which correspond to contours having one standard deviation.

**Exercise 13.29**

We now multiply both sides of (13.99) by $\hat{\alpha}(z_n)$ and substitute for $p(x_{n+1}|z_{n+1})$ and $p(z_{n+1}|z_n)$ using (13.75) and (13.76). Then we make use of (13.89), (13.90) and (13.91), together with (13.98), and after some manipulation we obtain
$$
\hat{\mu}_n = \mu_n + J_n (\hat{\mu}_{n+1} - A\mu_n) \quad (13.100)
$$
$$
\hat{V}_n = V_n + J_n (\hat{V}_{n+1} - P_n) J_n^T \quad (13.101)
$$
where we have defined
$$
J_n = V_n A^T (P_n)^{-1} \quad (13.102)
$$
and we have made use of $AV_n = P_n J_n^T$. Note that these recursions require that the forward pass be completed first so that the quantities $\mu_n$ and $V_n$ will be available for the backward pass.

For the EM algorithm, we also require the pairwise posterior marginals, which can be obtained from (13.65) in the form
$$
\xi(z_{n-1}, z_n) = (c_n)^{-1} \hat{\alpha}(z_{n-1}) p(x_n|z_n) p(z_n|z_{n-1}) \beta(z_n)
$$
$$
= \frac{\mathcal{N}(z_{n-1}|\mu_{n-1}, V_{n-1}) \mathcal{N}(z_n|Az_{n-1}, \Gamma) \mathcal{N}(x_n|Cz_n, \Sigma) \mathcal{N}(z_n|\hat{\mu}_n, \hat{V}_n)}{c_n \hat{\alpha}(z_n)} \quad (13.103)
$$
Substituting for $\hat{\alpha}(z_n)$ using (13.84) and rearranging, we see that $\xi(z_{n-1}, z_n)$ is a Gaussian with mean given with components $\langle z_{n-1} \rangle$ and $\langle z_n \rangle$, and a covariance between $z_n$ and $z_{n-1}$ given by
$$
\text{cov}[z_n, z_{n-1}] = J_{n-1} \hat{V}_n \quad (13.104)
$$

**Exercise 13.31**

### 13.3.2 Learning in LDS

So far, we have considered the inference problem for linear dynamical systems, assuming that the model parameters $\theta = \{A, \Gamma, C, \Sigma, \mu_0, V_0\}$ are known. Next, we consider the determination of these parameters using maximum likelihood (Ghahramani and Hinton, 1996b). Because the model has latent variables, this can be addressed using the EM algorithm, which was discussed in general terms in Chapter 9.

We can derive the EM algorithm for the linear dynamical system as follows. Let us denote the estimated parameter values at some particular cycle of the algorithm by $\theta^{\text{old}}$. For these parameter values, we can run the inference algorithm to determine the posterior distribution of the latent variables $p(Z|X, \theta^{\text{old}})$, or more precisely those local posterior marginals that are required in the M step. In particular, we shall require the following expectations
$$
E[z_n] = \hat{\mu}_n \quad (13.105)
$$
$$
E[z_n z_{n-1}^T] = J_{n-1}\hat{V}_n + \hat{\mu}_n \hat{\mu}_{n-1}^T \quad (13.106)
$$
$$
E[z_n z_n^T] = \hat{V}_n + \hat{\mu}_n \hat{\mu}_n^T \quad (13.107)
$$
where we have used (13.104).

Now we consider the complete-data log likelihood function, which is obtained by taking the logarithm of (13.6) and is therefore given by
$$
\ln p(X, Z|\theta) = \ln p(z_1|\mu_0, V_0) + \sum_{n=2}^N \ln p(z_n|z_{n-1}, A, \Gamma) + \sum_{n=1}^N \ln p(x_n|z_n, C, \Sigma) \quad (13.108)
$$
in which we have made the dependence on the parameters explicit. We now take the expectation of the complete-data log likelihood with respect to the posterior distribution $p(Z|X, \theta^{\text{old}})$ which defines the function
$$
Q(\theta, \theta^{\text{old}}) = E_{Z|\theta^{\text{old}}}[\ln p(X, Z|\theta)]. \quad (13.109)
$$
In the M step, this function is maximized with respect to the components of $\theta$.

**Exercise 13.32**

Consider first the parameters $\mu_0$ and $V_0$. If we substitute for $p(z_1|\mu_0, V_0)$ in (13.108) using (13.77), and then take the expectation with respect to $Z$, we obtain
$$
Q(\theta, \theta^{\text{old}}) = -\frac{1}{2} \ln |V_0| - \frac{1}{2} E_{Z|\theta^{\text{old}}} [(z_1 - \mu_0)^T V_0^{-1} (z_1 - \mu_0)] + \text{const}
$$
where all terms not dependent on $\mu_0$ or $V_0$ have been absorbed into the additive constant. Maximization with respect to $\mu_0$ and $V_0$ is easily performed by making use of the maximum likelihood solution for a Gaussian distribution discussed in Section 2.3.4, giving
$$
\mu_0^{\text{new}} = E[z_1] \quad (13.110)
$$
$$
V_0^{\text{new}} = E[z_1 z_1^T] - E[z_1]E[z_1]^T. \quad (13.111)
$$
Similarly, to optimize $A$ and $\Gamma$, we substitute for $p(z_n|z_{n-1}, A, \Gamma)$ in (13.108) using (13.75) giving
$$
Q(\theta, \theta^{\text{old}}) = -\frac{N-1}{2} \ln |\Gamma| - \frac{1}{2} \sum_{n=2}^N E_{Z|\theta^{\text{old}}} [(z_n - A z_{n-1})^T \Gamma^{-1} (z_n - A z_{n-1})] + \text{const} \quad (13.112)
$$

**Exercise 13.33**

in which the constant comprises terms that are independent of $A$ and $\Gamma$. Maximizing with respect to these parameters then gives
$$
A^{\text{new}} = \left( \sum_{n=2}^N E[z_n z_{n-1}^T] \right) \left( \sum_{n=2}^N E[z_{n-1} z_{n-1}^T] \right)^{-1} \quad (13.113)
$$
$$
\Gamma^{\text{new}} = \frac{1}{N-1} \sum_{n=2}^N \{ E[z_n z_n^T] - A^{\text{new}} E[z_{n-1} z_n^T] - E[z_n z_{n-1}^T] (A^{\text{new}})^T + A^{\text{new}} E[z_{n-1} z_{n-1}^T] (A^{\text{new}})^T \}. \quad (13.114)
$$
Note that $A^{\text{new}}$ must be evaluated first, and the result can then be used to determine $\Gamma^{\text{new}}$.

Finally, in order to determine the new values of $C$ and $\Sigma$, we substitute for $p(x_n|z_n, C, \Sigma)$ in (13.108) using (13.76) giving
$$
Q(\theta, \theta^{\text{old}}) = -\frac{N}{2} \ln |\Sigma| - \frac{1}{2} \sum_{n=1}^N E_{Z|\theta^{\text{old}}} [(x_n - C z_n)^T \Sigma^{-1} (x_n - C z_n)] + \text{const}.
$$

**Exercise 13.34**

Maximizing with respect to $C$ and $\Sigma$ then gives
$$
C^{\text{new}} = \left( \sum_{n=1}^N x_n E[z_n]^T \right) \left( \sum_{n=1}^N E[z_n z_n^T] \right)^{-1} \quad (13.115)
$$
$$
\Sigma^{\text{new}} = \frac{1}{N} \sum_{n=1}^N \{ x_n x_n^T - C^{\text{new}} E[z_n] x_n^T - x_n E[z_n]^T (C^{\text{new}})^T + C^{\text{new}} E[z_n z_n^T] (C^{\text{new}})^T \}. \quad (13.116)
$$

---