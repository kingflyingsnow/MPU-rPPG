## 3 Method

### 3.1 Problem Formulation

Given an input facial video sequence

$$
X=\{x_t\}_{t=1}^{T}, \quad x_t \in \mathbb{R}^{H\times W \times C},
$$

our goal is to jointly estimate heart rate (HR), respiratory rate (RR), and blood oxygen saturation (SpO$_2$):

$$
\hat{y}^{hr}, \quad \hat{y}^{rr}, \quad \hat{y}^{spo}.
$$

Different from conventional fully shared multi-task learning, we assume that the three tasks contain both shared physiological dynamics and task-specific latent factors. Therefore, we decompose the latent representation into one shared physiological state and three task-private states.

### 3.2 Shared Physiological State Modeling

Following the core idea of FacePhys, we model the common physiological dynamics from facial videos as a latent state evolution process. For ease of implementation, we adopt a discrete state-space formulation:

$$
h_t^{s} = f_s(x_t, h_{t-1}^{s}),
$$

where $h_t^{s}\in\mathbb{R}^{d_s}$ denotes the shared physiological state at time step $t$, and $f_s(\cdot)$ is implemented by a lightweight FacePhys-style state-space block.

The shared state summarizes the dominant physiological information that is beneficial across tasks, such as subtle periodic color changes and temporal rhythm patterns.

After temporal aggregation, the shared sequence representation is written as

$$
H^{s} = \{h_t^{s}\}_{t=1}^{T}, \qquad
z^{s} = \mathrm{Pool}(H^{s}),
$$

where $\mathrm{Pool}(\cdot)$ denotes temporal average pooling or attention pooling.

### 3.3 Task-Private State Branches

To alleviate negative transfer caused by pure parameter sharing, we further introduce task-private state branches for HR, RR, and SpO$_2$. Each branch receives the shared state sequence and learns task-specific residual dynamics:

$$
h_t^{hr} = f_{hr}(h_t^{s}, h_{t-1}^{hr}),
$$

$$
h_t^{rr} = f_{rr}(h_t^{s}, h_{t-1}^{rr}),
$$

$$
h_t^{spo} = f_{spo}(h_t^{s}, h_{t-1}^{spo}),
$$

where

$$
h_t^{hr}\in\mathbb{R}^{d_p}, \quad
h_t^{rr}\in\mathbb{R}^{d_p}, \quad
h_t^{spo}\in\mathbb{R}^{d_p}.
$$

The corresponding aggregated task-private features are

$$
z^{hr} = \mathrm{Pool}(\{h_t^{hr}\}_{t=1}^{T}),
$$

$$
z^{rr} = \mathrm{Pool}(\{h_t^{rr}\}_{t=1}^{T}),
$$

$$
z^{spo} = \mathrm{Pool}(\{h_t^{spo}\}_{t=1}^{T}).
$$

Each task prediction is then obtained by combining the shared feature and the corresponding private feature:

$$
\hat{y}^{hr} = g_{hr}([z^{s}; z^{hr}]),
$$

$$
\hat{y}^{rr} = g_{rr}([z^{s}; z^{rr}]),
$$

$$
\hat{y}^{spo} = g_{spo}([z^{s}; z^{spo}]),
$$

where $[\,\cdot\,;\,\cdot\,]$ denotes feature concatenation, and $g_{hr}, g_{rr}, g_{spo}$ are lightweight prediction heads.

This design allows the shared branch to learn common physiological dynamics, while the private branches focus on task-specific factors that are not fully explained by a single shared cardiac state.

### 3.4 Orthogonal Disentanglement

Since HR, RR, and SpO$_2$ are correlated but not identical, directly sharing all features may lead to representation entanglement and negative transfer. To encourage task-specific branches to capture complementary information, we impose a soft orthogonality constraint on private features:

$$
\mathcal{L}_{orth}
=
\| {z^{hr}}^{\top} z^{rr} \|_F^2
+
\| {z^{hr}}^{\top} z^{spo} \|_F^2
+
\| {z^{rr}}^{\top} z^{spo} \|_F^2.
$$

Optionally, a weaker separation term can also be added between the shared and private features:

$$
\mathcal{L}_{sep}
=
\| {z^{s}}^{\top} z^{hr} \|_F^2
+
\| {z^{s}}^{\top} z^{rr} \|_F^2
+
\| {z^{s}}^{\top} z^{spo} \|_F^2.
$$

Here, $\mathcal{L}_{orth}$ is the main term, while $\mathcal{L}_{sep}$ is used with a smaller weight to avoid over-separating naturally correlated physiological representations.

### 3.5 Dynamic Task Weighting

The difficulty and reliability of HR, RR, and SpO$_2$ estimation vary during training. Using fixed task weights may cause the optimization to be dominated by easier tasks. To address this issue, we introduce a simple dynamic weighting module conditioned on the shared physiological feature:

$$
\boldsymbol{\alpha} = [\alpha_{hr}, \alpha_{rr}, \alpha_{spo}]
= \mathrm{Softmax}(W_w z^s + b_w),
$$

where $\alpha_{hr}+\alpha_{rr}+\alpha_{spo}=1$.

The task-specific losses are adaptively combined as

$$
\mathcal{L}_{mtl}
=
\alpha_{hr}\mathcal{L}_{hr}
+
\alpha_{rr}\mathcal{L}_{rr}
+
\alpha_{spo}\mathcal{L}_{spo}.
$$

This formulation is much easier to optimize than a continuous-time weight ODE, while still allowing the model to assign different importance to different tasks according to the current latent physiological state.

### 3.6 Task Losses

We use standard regression losses for the three tasks:

$$
\mathcal{L}_{hr} = \ell(\hat{y}^{hr}, y^{hr}),
$$

$$
\mathcal{L}_{rr} = \ell(\hat{y}^{rr}, y^{rr}),
$$

$$
\mathcal{L}_{spo} = \ell(\hat{y}^{spo}, y^{spo}),
$$

where $\ell(\cdot,\cdot)$ can be implemented as $L_1$, Smooth-$L_1$, or MSE loss depending on the target format.

If waveform supervision is available for HR-related estimation, $\mathcal{L}_{hr}$ can be replaced or augmented by a waveform consistency loss.

### 3.7 Overall Objective

The final training objective is

$$
\mathcal{L}
=
\mathcal{L}_{mtl}
+
\lambda_{orth}\mathcal{L}_{orth}
+
\lambda_{sep}\mathcal{L}_{sep},
$$

where $\lambda_{orth}$ and $\lambda_{sep}$ are balancing coefficients.

In practice, we set $\lambda_{orth} > \lambda_{sep}$, so that the model mainly enforces disentanglement among task-private branches while preserving sufficient shared physiological information.

## 4 Discussion of Simplicity and Implementation

The proposed design is intentionally minimal:

- only one shared FacePhys-style state-space encoder is introduced;
- each task branch is implemented by a lightweight private state block;
- dynamic task weighting is realized by a single linear projection followed by softmax;
- disentanglement is enforced by a simple orthogonality regularizer.

Therefore, the framework preserves the efficiency advantage of FacePhys while extending it to multi-task physiological estimation with reduced risk of negative transfer.
