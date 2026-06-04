## 3 Method
### 3.1 Problem Formulation
Given an input facial video sequence
 [
X={x_t}_{t=1}^{T}, \quad x_t \in \mathbb{R}^{H\times W \times C},
]
our goal is to jointly estimate heart rate (HR), respiratory rate (RR), and blood oxygen saturation (SpO [ o bj ec tO bj ec t ] 2 ​ ):

[
\hat{y}^{hr}, \quad \hat{y}^{rr}, \quad \hat{y}^{spo}.
]

Different from conventional fully shared multi-task learning, we assume that the three tasks contain both shared physiological dynamics and task-specific latent factors . Therefore, we decompose the latent representation into one shared physiological state and three task-private states.

### 3.2 Shared Physiological State Modeling
Following the core idea of FacePhys, we model the common physiological dynamics from facial videos as a latent state evolution process. For ease of implementation, we adopt a discrete state-space formulation:

[
h_t^{s} = f_s(x_t, h_{t-1}^{s}),
]
where [ o bj ec tO bj ec t ] h t s ​ ∈ R d s ​ denotes the shared physiological state at time step [ o bj ec tO bj ec t ] t , and [ o bj ec tO bj ec t ] f s ​ ( ⋅ ) is implemented by a lightweight FacePhys-style state-space block.

The shared state summarizes the dominant physiological information that is beneficial across tasks, such as subtle periodic color changes and temporal rhythm patterns.

After temporal aggregation, the shared sequence representation is written as

[
H^{s} = {h_t^{s}}_{t=1}^{T}, \qquad
z^{s} = \mathrm{Pool}(H^{s}),
]
where [ o bj ec tO bj ec t ] Pool ( ⋅ ) denotes temporal average pooling or attention pooling.

### 3.3 Task-Private State Branches
To alleviate negative transfer caused by pure parameter sharing, we further introduce task-private state branches for HR, RR, and SpO [ o bj ec tO bj ec t ] 2 ​ . Each branch receives the shared state sequence and learns task-specific residual dynamics:

[
h_t^{hr} = f_{hr}(h_t^{s}, h_{t-1}^{hr}),
]
[
h_t^{rr} = f_{rr}(h_t^{s}, h_{t-1}^{rr}),
]
[
h_t^{spo} = f_{spo}(h_t^{s}, h_{t-1}^{spo}),
]
where
[
h_t^{hr}\in\mathbb{R}^{d_p}, \quad
h_t^{rr}\in\mathbb{R}^{d_p}, \quad
h_t^{spo}\in\mathbb{R}^{d_p}.
]

The corresponding aggregated task-private features are

[
z^{hr} = \mathrm{Pool}({h_t^{hr}} {t=1}^{T}),
]
[
z^{rr} = \mathrm{Pool}({h_t^{rr}} {t=1}^{T}),
]
[
z^{spo} = \mathrm{Pool}({h_t^{spo}}_{t=1}^{T}).
]

Each task prediction is then obtained by combining the shared feature and the corresponding private feature:

[
\hat{y}^{hr} = g_{hr}([z^{s}; z^{hr}]),
]
[
\hat{y}^{rr} = g_{rr}([z^{s}; z^{rr}]),
]
[
\hat{y}^{spo} = g_{spo}([z^{s}; z^{spo}]),
]
where [ o bj ec tO bj ec t ] [ ⋅ ; ⋅ ] denotes feature concatenation, and [ o bj ec tO bj ec t ] g h r ​ , g rr ​ , g s p o ​ are lightweight prediction heads.

This design allows the shared branch to learn common physiological dynamics, while the private branches focus on task-specific factors that are not fully explained by a single shared cardiac state.

### 3.4 Orthogonal Disentanglement
Since HR, RR, and SpO [ o bj ec tO bj ec t ] 2 ​ are correlated but not identical, directly sharing all features may lead to representation entanglement and negative transfer. To encourage task-specific branches to capture complementary information, we impose a soft orthogonality constraint on private features:

# [
\mathcal{L}_{orth}
| {z^{hr}}^{\top} z^{rr} |_F^2
+
| {z^{hr}}^{\top} z^{spo} |_F^2
+
| {z^{rr}}^{\top} z^{spo} |_F^2.
]

Optionally, a weaker separation term can also be added between the shared and private features:

# [
\mathcal{L}_{sep}
| {z^{s}}^{\top} z^{hr} |_F^2
+
| {z^{s}}^{\top} z^{rr} |_F^2
+
| {z^{s}}^{\top} z^{spo} |_F^2.
]

Here, [ o bj ec tO bj ec t ] L or t h ​ is the main term, while [ o bj ec tO bj ec t ] L se p ​ is used with a smaller weight to avoid over-separating naturally correlated physiological representations.

### 3.5 Dynamic Task Weighting
The difficulty and reliability of HR, RR, and SpO [ o bj ec tO bj ec t ] 2 ​ estimation vary during training. Using fixed task weights may cause the optimization to be dominated by easier tasks. To address this issue, we introduce a simple dynamic weighting module conditioned on the shared physiological feature:

[
\boldsymbol{\alpha} = [\alpha_{hr}, \alpha_{rr}, \alpha_{spo}]
= \mathrm{Softmax}(W_w z^s + b_w),
]
where [ o bj ec tO bj ec t ] α h r ​ + α rr ​ + α s p o ​ = 1 .

The task-specific losses are adaptively combined as

# [
\mathcal{L}_{mtl}
\alpha_{hr}\mathcal{L} {hr}
+
\alpha {rr}\mathcal{L} {rr}
+
\alpha {spo}\mathcal{L}_{spo}.
]

This formulation is much easier to optimize than a continuous-time weight ODE, while still allowing the model to assign different importance to different tasks according to the current latent physiological state.

### 3.6 Task Losses
We use standard regression losses for the three tasks:

[
\mathcal{L} {hr} = \ell(\hat{y}^{hr}, y^{hr}),
]
[
\mathcal{L} {rr} = \ell(\hat{y}^{rr}, y^{rr}),
]
[
\mathcal{L}_{spo} = \ell(\hat{y}^{spo}, y^{spo}),
]
where [ o bj ec tO bj ec t ] ℓ ( ⋅ , ⋅ ) can be implemented as [ o bj ec tO bj ec t ] L 1 ​ , Smooth- [ o bj ec tO bj ec t ] L 1 ​ , or MSE loss depending on the target format.

If waveform supervision is available for HR-related estimation, [ o bj ec tO bj ec t ] L h r ​ can be replaced or augmented by a waveform consistency loss.

### 3.7 Overall Objective
The final training objective is

# [
\mathcal{L}
\mathcal{L} {mtl}
+
\lambda {orth}\mathcal{L} {orth}
+
\lambda {sep}\mathcal{L}_{sep},
]
where [ o bj ec tO bj ec t ] λ or t h ​ and [ o bj ec tO bj ec t ] λ se p ​ are balancing coefficients.

In practice, we set [ o bj ec tO bj ec t ] λ or t h ​ > λ se p ​ , so that the model mainly enforces disentanglement among task-private branches while preserving sufficient shared physiological information.
