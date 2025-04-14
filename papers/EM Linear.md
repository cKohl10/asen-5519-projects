# Parameter Estimation for Linear Dynamical Systems

Zoubin Ghahramani
Geoffrey E. Hinton
Department of Computer Science
University of Toronto
6 King's College Road
Toronto, Canada M5S 1A4
Email: zoubin@cs.toronto.edu
Technical Report CRG-TR-96-2
February 22, 1996

## Abstract

Linear systems have been used extensively in engineering to model and control the behavior of dynamical systems. In this note, we present the Expectation Maximization (EM) algorithm for estimating the parameters of linear systems (Shumway and Stoffer, 1982). We also point out the relationship between linear dynamical systems, factor analysis, and hidden Markov models.

## Introduction

The goal of this note is to introduce the EM algorithm for estimating the parameters of linear dynamical systems (LDS). Such linear systems can be used both for supervised and unsupervised modeling of time series. We first describe the model and then briefly point out its relation to factor analysis and other data modeling techniques.

## The Model

Linear time-invariant dynamical systems, also known as linear Gaussian state-space models, can be described by the following two equations:

$$
\begin{gathered}
\mathbf{x}_{t+1}=A \mathbf{x}_{t}+\mathbf{w}_{t} \\
\mathbf{y}_{t}=C \mathbf{x}_{t}+\mathbf{v}_{t}
\end{gathered}
$$

Time is indexed by the discrete index $t$. The output $\mathbf{y}_{t}$ is a linear function of the state, $\mathbf{x}_{t}$, and the state at one time step depends linearly on the previous state. Both state and output noise, $\mathbf{w}_{t}$ and $\mathbf{v}_{t}$, are zero-mean normally distributed random variables with covariance matrices $Q$ and $R$, respectively. Only the output of the system is observed, the state and all the noise variables are hidden.

Rather than regarding the state as a deterministic value corrupted by random noise, we combine the state variable and the state noise variable into a single Gaussian random variable; we form a similar combination for the output. Based on (1) and (2) we can write the conditional densities for the state and output,

$$
\begin{aligned}
P\left(\mathbf{y}_{t} \mid \mathbf{x}_{t}\right) & =\exp \left\{-\frac{1}{2}\left[\mathbf{y}_{t}-C \mathbf{x}_{t}\right]^{\prime} R^{-1}\left[\mathbf{y}_{t}-C \mathbf{x}_{t}\right]\right\}(2 \pi)^{-p / 2}|R|^{-1 / 2} \\
P\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right) & =\exp \left\{-\frac{1}{2}\left[\mathbf{x}_{t}-A \mathbf{x}_{t-1}\right]^{\prime} Q^{-1}\left[\mathbf{x}_{t}-A \mathbf{x}_{t-1}\right]\right\}(2 \pi)^{-k / 2}|Q|^{-1 / 2}
\end{aligned}
$$

A sequence of $T$ output vectors $\left(\mathbf{y}_{1}, \mathbf{y}_{2}, \ldots, \mathbf{y}_{T}\right)$ is denoted by $\{\mathbf{y}\}$; a subsequence $\left(\mathbf{y}_{t_{0}}, \mathbf{y}_{t_{0}+1}, \ldots, \mathbf{y}_{t_{1}}\right)$ by $\{\mathbf{y}\}_{t_{0}}^{t_{1}}$; similarly for the states.

By the Markov property implicit in this model,

$$
P(\{\mathbf{x}\},\{\mathbf{y}\})=P\left(\mathbf{x}_{1}\right) \prod_{t=2}^{T} P\left(\mathbf{x}_{t} \mid \mathbf{x}_{t-1}\right) \prod_{t=1}^{T} P\left(\mathbf{y}_{t} \mid \mathbf{x}_{t}\right)
$$

Assuming a Gaussian initial state density

$$
P\left(\mathbf{x}_{1}\right)=\exp \left\{-\frac{1}{2}\left[\mathbf{x}_{1}-\boldsymbol{\pi}_{1}\right]^{\prime} V_{1}^{-1}\left[\mathbf{x}_{1}-\boldsymbol{\pi}_{1}\right]\right\}(2 \pi)^{-k / 2}\left|V_{1}\right|^{-1 / 2}
$$

Therefore, the joint log probability is a sum of quadratic terms,

$$
\begin{aligned}
\log P(\{\mathbf{x}\},\{\mathbf{y}\})= & -\sum_{t=1}^{T}\left(\frac{1}{2}\left[\mathbf{y}_{t}-C \mathbf{x}_{t}\right]^{\prime} R^{-1}\left[\mathbf{y}_{t}-C \mathbf{x}_{t}\right]\right)-\frac{T}{2} \log |R| \\
& -\sum_{t=2}^{T}\left(\frac{1}{2}\left[\mathbf{x}_{t}-A \mathbf{x}_{t-1}\right]^{\prime} Q^{-1}\left[\mathbf{x}_{t}-A \mathbf{x}_{t-1}\right]\right)-\frac{T-1}{2} \log |Q| \\
& -\frac{1}{2}\left[\mathbf{x}_{1}-\boldsymbol{\pi}_{1}\right]^{\prime} V_{1}^{-1}\left[\mathbf{x}_{1}-\boldsymbol{\pi}_{1}\right]-\frac{1}{2} \log \left|V_{1}\right|-\frac{T(p+k)}{2} \log 2 \pi
\end{aligned}
$$

Often the inputs to the system can also be observed. In this case, the goal is to model the input-output response of a system. Denoting the inputs by $\mathbf{u}_{t}$, the state equation is

$$
\mathbf{x}_{t+1}=A \mathbf{x}_{t}+B \mathbf{u}_{t}+\mathbf{w}_{t}
$$

where $B$ is the input matrix relating inputs linearly to states. We will present the learning algorithm for the output-only case, although the extensions to the input-output case are straightforward.

If only the outputs of the system can be observed the problem can be seen as an unsupervised problem. That is, the goal is to model the unconditional density of the observations. If both inputs and outputs are observed, the problem becomes supervised, modeling the conditional density of the output given the input.

## Related Methods

In its unsupervised incarnation, this model is an extension of maximum likelihood factor analysis (Everitt, 1984). The factor, $\mathbf{x}_{t}$, evolves over time according to linear dynamics. In factor analysis, a further assumption is made that the output noise along each dimension is uncorrelated, i.e. that $R$ is diagonal. The goal of factor analysis is therefore to compress the correlational structure of the data into the values of the lower dimensional factors, while allowing independent noise terms to model the uncorrelated noise. The assumption of a diagonal $R$ matrix can also be easily incorporated into the estimation procedure for the parameters of a linear dynamical system.

The linear dynamical system can also be seen as a continuous-state analogue of the hidden Markov model (HMM; see Rabiner and Juang, 1986, for a review). The forward part of the forward-backward algorithm from HMMs is computed by the well-known Kalman filter in LDSs; similarly, the backward part is computed by using Rauch's recursion (Rauch, 1963). Together, these two recursions can be used to solve the problem of inferring the probabilities of the states given the observation sequence (known in engineering as the smoothing problem). These posterior probabilities form the basis of the E step of the EM algorithm.

Finally, linear dynamical systems can also be represented as graphical probabilistic models (sometimes referred to as belief networks). The Kalman-Rauch recursions are special cases of the probability propagation algorithms that have been developed for graphical models (Lauritzen and Spiegelhalter, 1988; Pearl, 1988).

## The EM Algorithm

Shumway and Stoffer (1982) presented an EM algorithm for linear dynamical systems where the observation matrix, $C$, is known. Since then, many authors have presented closely related models and extensions, also fit with the EM algorithm (Shumway and Stoffer, 1991; Kim, 1994; Athaide, 1995). Here we present a basic form of the EM algorithm with $C$ unknown, an obvious modification of Shumway and Stoffer's original work. This note is meant as a succinct review of this literature for those wishing to implement learning in linear dynamical systems.

The E step of EM requires computing the expected log likelihood,

$$
\mathcal{Q}=E\left[\log P(\{\mathbf{x}\},\{\mathbf{y}\}) \mid\{\mathbf{y}\}\right]
$$

This quantity depends on three expectations- $E\left[\mathbf{x}_{t} \mid\{\mathbf{y}\}\right], E\left[\mathbf{x}_{t} \mathbf{x}_{t}^{\prime} \mid\{\mathbf{y}\}\right], E\left[\mathbf{x}_{t} \mathbf{x}_{t-1}^{\prime} \mid\{\mathbf{y}\}\right]$-which we will denote by the symbols:

$$
\begin{aligned}
\hat{\mathbf{x}}_{t} & \equiv E\left[\mathbf{x}_{t} \mid\{\mathbf{y}\}\right] \\
P_{t} & \equiv E\left[\mathbf{x}_{t} \mathbf{x}_{t}^{\prime} \mid\{\mathbf{y}\}\right] \\
P_{t, t-1} & \equiv E\left[\mathbf{x}_{t} \mathbf{x}_{t-1}^{\prime} \mid\{\mathbf{y}\}\right]
\end{aligned}
$$

Note that the state estimate, $\hat{\mathbf{x}}_{t}$, differs from the one computed in a Kalman filter in that it depends on past and future observations; the Kalman filter estimates $E\left[\mathbf{x}_{t} \mid\{\mathbf{y}\}_{1}^{t}\right]$ (Anderson and Moore, 1979). We first describe the M step of the parameter estimation algorithm before showing how the above expectations are computed in the E step.

### The M step

The parameters of this system are $A, C, R, Q, \boldsymbol{\pi}_{1}, V_{1}$. Each of these is re-estimated by taking the corresponding partial derivative of the expected log likelihood, setting to zero, and solving. This results in the following:

- Output matrix:

$$
\begin{gathered}
\frac{\partial \mathcal{Q}}{\partial C}=-\sum_{t=1}^{T} R^{-1} \mathbf{y}_{t} \hat{\mathbf{x}}_{t}^{\prime}+\sum_{t=1}^{T} R^{-1} C P_{t}=0 \\
C^{\text {new }}=\left(\sum_{t=1}^{T} \mathbf{y}_{t} \hat{\mathbf{x}}_{t}^{\prime}\right)\left(\sum_{t=1}^{T} P_{t}\right)^{-1}
\end{gathered}
$$

- Output noise covariance:

$$
\begin{gathered}
\frac{\partial \mathcal{Q}}{\partial R^{-1}}=\frac{T}{2} R-\sum_{t=1}^{T}\left(\frac{1}{2} \mathbf{y}_{t} \mathbf{y}_{t}^{\prime}-C \hat{\mathbf{x}}_{t} \mathbf{y}_{t}^{\prime}+\frac{1}{2} C P_{t} C^{\prime}\right)=0 \\
R^{\text {new }}=\frac{1}{T} \sum_{t=1}^{T}\left(\mathbf{y}_{t} \mathbf{y}_{t}^{\prime}-C^{\text {new }} \hat{\mathbf{x}}_{t} \mathbf{y}_{t}^{\prime}\right)
\end{gathered}
$$

- State dynamics matrix:

$$
\begin{gathered}
\frac{\partial \mathcal{Q}}{\partial A}=-\sum_{t=2}^{T} Q^{-1} P_{t, t-1}+\sum_{t=2}^{T} Q^{-1} A P_{t-1}=0 \\
A^{\text {new }}=\left(\sum_{t=2}^{T} P_{t, t-1}\right)\left(\sum_{t=2}^{T} P_{t-1}\right)^{-1}
\end{gathered}
$$

- State noise covariance:

$$
\begin{aligned}
\frac{\partial \mathcal{Q}}{\partial Q^{-1}}= & \frac{T-1}{2} Q-\frac{1}{2} \sum_{t=2}^{T}\left(P_{t}-A P_{t-1, t}-P_{t, t-1} A^{\prime}+A P_{t-1} A^{\prime}\right)=0 \\
= & \frac{T-1}{2} Q-\frac{1}{2}\left(\sum_{t=2}^{T} P_{t}-A^{\text {new }} \sum_{t=2}^{T} P_{t-1, t}\right) \\
& Q^{\text {new }}=\frac{1}{T-1}\left(\sum_{t=2}^{T} P_{t}-A^{\text {new }} \sum_{t=2}^{T} P_{t-1, t}\right)
\end{aligned}
$$

- Initial state mean:

$$
\begin{gathered}
\frac{\partial \mathcal{Q}}{\partial \boldsymbol{\pi}_{1}}=\left(\hat{\mathbf{x}}_{1}-\boldsymbol{\pi}_{1}\right) V_{1}^{-1}=0 \\
\boldsymbol{\pi}_{1}^{\text {new }}=\hat{\mathbf{x}}_{1}
\end{gathered}
$$

- Initial state covariance:

$$
\begin{gathered}
\frac{\partial \mathcal{Q}}{\partial V_{1}^{-1}}=\frac{1}{2} V_{1}-\frac{1}{2}\left(P_{1}-\hat{\mathbf{x}}_{1} \boldsymbol{\pi}_{1}^{\prime}-\boldsymbol{\pi}_{1} \hat{\mathbf{x}}_{1}^{\prime}+\boldsymbol{\pi}_{1} \boldsymbol{\pi}_{1}^{\prime}\right) \\
V_{1}^{\text {new }}=P_{1}-\hat{\mathbf{x}}_{1} \hat{\mathbf{x}}_{1}^{\prime}
\end{gathered}
$$

The above equations can be readily generalized to multiple observation sequences, with one subtlety regarding the estimate of the initial state covariance. Assume $N$ observation sequences of length $T$, let $\hat{\mathbf{x}}_{t}^{(i)}$ be the estimate of state at time $t$ given the $i^{\text {th }}$ sequence, and

$$
\tilde{\hat{\mathbf{x}}}_{t}=\frac{1}{N} \sum_{i=1}^{N} \hat{\mathbf{x}}_{t}^{(i)}
$$

Then the initial state covariance is

$$
V_{1}^{\text {new }}=P_{1}-\tilde{\hat{\mathbf{x}}}_{1} \tilde{\hat{\mathbf{x}}}_{1}^{\prime}+\frac{1}{N} \sum_{i=1}^{N}\left[\hat{\mathbf{x}}_{1}^{(i)}-\tilde{\hat{\mathbf{x}}}_{1}\right]\left[\hat{\mathbf{x}}_{1}^{(i)}-\tilde{\hat{\mathbf{x}}}_{1}\right]^{\prime}
$$

### The E step

Using $\mathbf{x}_{t}^{\tau}$ to denote $E\left(\mathbf{x}_{t} \mid\{\mathbf{y}\}_{1}^{\tau}\right)$, and $V_{t}^{\tau}$ to denote $\operatorname{Var}\left(\mathbf{x}_{t} \mid\{\mathbf{y}\}_{1}^{\tau}\right)$, we obtain the following Kalman filter forward recursions:

$$
\begin{aligned}
\mathbf{x}_{t}^{t-1} & =A \mathbf{x}_{t-1}^{t-1} \\
V_{t}^{t-1} & =A V_{t-1}^{t-1} A^{\prime}+Q \\
K_{t} & =V_{t}^{t-1} C^{\prime}\left(C V_{t}^{t-1} C^{\prime}+R\right)^{-1} \\
\mathbf{x}_{t}^{t} & =\mathbf{x}_{t}^{t-1}+K_{t}\left(\mathbf{y}_{t}-C \mathbf{x}_{t}^{t-1}\right) \\
V_{t}^{t} & =V_{t}^{t-1}-K_{t} C V_{t}^{t-1}
\end{aligned}
$$

where $\mathbf{x}_{1}^{0}=\boldsymbol{\pi}_{1}$ and $V_{1}^{0}=V_{1}$. Following Shumway and Stoffer (1982), to compute $\hat{\mathbf{x}}_{t} \equiv \mathbf{x}_{t}^{T}$ and $P_{t} \equiv V_{t}^{T}+\mathbf{x}_{t}^{T} \mathbf{x}_{t}^{T^{\prime}}$ one performs a set of backward recursions using

$$
\begin{aligned}
J_{t-1} & =V_{t-1}^{t-1} A^{\prime}\left(V_{t}^{t-1}\right)^{-1} \\
\mathbf{x}_{t-1}^{T} & =\mathbf{x}_{t-1}^{t-1}+J_{t-1}\left(\mathbf{x}_{t}^{T}-A \mathbf{x}_{t-1}^{t-1}\right) \\
V_{t-1}^{T} & =V_{t-1}^{t-1}+J_{t-1}\left(V_{t}^{T}-V_{t}^{t-1}\right) J_{t-1}^{\prime}
\end{aligned}
$$

We also require $P_{t, t-1} \equiv V_{t, t-1}^{T}+\mathbf{x}_{t}^{T} \mathbf{x}_{t-1}^{T^{\prime}}$, which can be obtained through the backward recursions

$$
V_{t-1, t-2}^{T}=V_{t-1}^{t-1} J_{t-2}^{\prime}+J_{t-1}\left(V_{t, t-1}^{T}-A V_{t-1}^{t-1}\right) J_{t-2}^{\prime}
$$

which is initialized $V_{T, T-1}^{T}=\left(I-K_{T} C\right) A V_{T-1}^{T-1}$.

## References

Anderson, B. D. O. and Moore, J. B. (1979). *Optimal Filtering*. Prentice-Hall, Englewood Cliffs, NJ.

Athaide, C. R. (1995). *Likelihood Evaluation and State Estimation for Nonlinear State Space Models*. Ph.D. Thesis, Graduate Group in Managerial Science and Applied Economics, University of Pennsylvania, Philadelphia, PA.

Everitt, B. S. (1984). *An Introduction to Latent Variable Models*. Chapman and Hall, London.

Kim, C.-J. (1994). Dynamic linear models with Markov-switching. *J. Econometrics*, 60:1-22.

Lauritzen, S. L. and Spiegelhalter, D. J. (1988). Local computations with probabilities on graphical structures and their application to expert systems. *J. Royal Statistical Society B*, pages 157-224.

Pearl, J. (1988). *Probabilistic Reasoning in Intelligent Systems: Networks of Plausible Inference*. Morgan Kaufmann, San Mateo, CA.

Rabiner, L. R. and Juang, B. H. (1986). An Introduction to hidden Markov models. *IEEE Acoustics, Speech & Signal Processing Magazine*, 3:4-16.

Rauch, H. E. (1963). Solutions to the linear smoothing problem. *IEEE Transactions on Automatic Control*, 8:371-372.

Shumway, R. H. and Stoffer, D. S. (1982). An approach to time series smoothing and forecasting using the EM algorithm. *J. Time Series Analysis*, 3(4):253-264.

Shumway, R. H. and Stoffer, D. S. (1991). Dynamic linear models with switching. *J. Amer. Stat. Assoc.*, 86:763-769.
