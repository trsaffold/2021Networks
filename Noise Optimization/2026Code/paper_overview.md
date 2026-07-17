# Optimizing Synchronization by Shaping Correlated Noise

## 1. The Problem

Consider a network of coupled oscillators driven by Gaussian white noise. The oscillator dynamics and network couplings are assumed to be fixed. The control variable is instead the **spatial covariance structure of the noise** applied across the oscillators.

The central question is:

> Given fixed oscillator dynamics, fixed network couplings, and fixed marginal noise amplitudes, how should the noise acting on different oscillators be correlated in order to maximize synchronization?

Write the noise vector as

$$
\eta(t)=
\begin{pmatrix}
\eta_1(t)\\
\vdots\\
\eta_N(t)
\end{pmatrix},
$$

with

$$
\langle \eta(t)\rangle=0
$$

and

$$
\langle \eta(t)\eta(t')^T\rangle
=
C\,\delta(t-t').
$$

The matrix $C$ is the noise covariance matrix. Its diagonal entries specify the variance applied to each oscillator, while its off-diagonal entries specify the correlations between the noise inputs.

The optimization problem is therefore not to remove noise, but to choose an admissible $C$ whose correlations make the oscillator network as synchronized as possible.

---

## 2. Oscillator Dynamics

### 2.1 Kuramoto Dynamics

The paper uses the noisy network Kuramoto model

$$
\dot{\theta}_i
=
\omega_i
+
\sum_{j=1}^{N}
K_{ij}\sin(\theta_j-\theta_i)
+
\eta_i(t).
$$

Here:

- $\theta_i(t)$ is the phase of oscillator $i$.
- $\omega_i$ is its natural frequency.
- $K_{ij}$ is the coupling strength from oscillator $j$ to oscillator $i$.
- $\eta_i(t)$ is the Gaussian white-noise input.

The coupling depends only on phase differences, so adding the same phase to every oscillator does not change the dynamics.

### 2.2 Kuramoto as the First Fourier Harmonic

The sine interaction is not the only possible coupling law. A general phase-reduced oscillator network can be written as

$$
\dot{\theta}_i
=
\omega_i
+
\sum_j \Gamma_{ij}(\theta_j-\theta_i)
+
\eta_i(t),
$$

where each $2\pi$-periodic coupling function can be expanded in a Fourier series:

$$
\Gamma_{ij}(\phi)
=
a_{ij,0}
+
\sum_{m=1}^{\infty}
\left[
a_{ij,m}\cos(m\phi)
+
b_{ij,m}\sin(m\phi)
\right].
$$

The Kuramoto model retains only the leading sinusoidal component:

$$
\Gamma_{ij}(\phi)
\approx
K_{ij}\sin\phi.
$$

It is therefore the first Fourier-harmonic approximation to a broader class of weakly coupled phase-oscillator models.

### 2.3 An Alternative: The Swing Equation

The same noise-covariance framework can be applied to other oscillator dynamics. In power-grid models, for example, one commonly uses the second-order swing equation

$$
M_i\ddot{\theta}_i
+
D_i\dot{\theta}_i
=
P_i
+
\sum_j K_{ij}\sin(\theta_j-\theta_i)
+
\eta_i(t).
$$

Here:

- $M_i$ is inertia.
- $D_i$ is damping.
- $P_i$ is the net mechanical or electrical power injection.

The Kuramoto equation is first order in phase, whereas the swing equation includes angular velocity as an additional state variable. The optimization logic remains similar: linearize the chosen dynamics around a synchronized operating state, determine how noise covariance propagates into state covariance, and optimize the resulting synchronization measure.

---

## 3. Why a Covariance Matrix Must Be Positive Semidefinite

A valid covariance matrix must satisfy

$$
C=C^T
$$

and

$$
C\succeq 0.
$$

The reason is that every linear combination of the noise variables must have nonnegative variance. For any vector $a$,

$$
a^T\eta
$$

is a scalar random input. Its variance is

$$
\operatorname{Var}(a^T\eta)
=
\left\langle
(a^T\eta)^2
\right\rangle
=
a^T C a.
$$

A variance cannot be negative, so

$$
a^T C a\ge 0
\qquad
\text{for every }a.
$$

This is exactly the definition of positive semidefiniteness.

For two oscillators,

$$
C=
\begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2\\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}.
$$

The PSD condition gives

$$
\det C
=
\sigma_1^2\sigma_2^2(1-\rho^2)
\ge 0,
$$

and therefore

$$
-1\le \rho\le 1.
$$

The endpoints $\rho=\pm1$ are singular, rank-one covariance matrices. They represent perfectly correlated or perfectly anticorrelated noise.

---

## 4. Two-Oscillator Reduction

Consider two oscillators:

$$
\dot{\theta}_1
=
\omega_1
+
\frac{K}{2}\sin(\theta_2-\theta_1)
+
\eta_1,
$$

$$
\dot{\theta}_2
=
\omega_2
+
\frac{K}{2}\sin(\theta_1-\theta_2)
+
\eta_2.
$$

Define the phase difference

$$
\delta=\theta_1-\theta_2
$$

and frequency difference

$$
\Delta\omega=\omega_1-\omega_2.
$$

Subtracting the equations gives

$$
\dot{\delta}
=
\Delta\omega
-
K\sin\delta
+
\Delta\eta,
$$

where

$$
\Delta\eta=\eta_1-\eta_2.
$$

Thus the synchronization of the pair depends only on the **differential noise**, not on noise in the common phase direction.

The differential-noise variance is

$$
\sigma_\Delta^2
=
\left\langle
(\eta_1-\eta_2)^2
\right\rangle.
$$

Expanding,

$$
\sigma_\Delta^2
=
C_{11}+C_{22}-2C_{12}.
$$

Writing

$$
C_{11}=\sigma_1^2,
\qquad
C_{22}=\sigma_2^2,
\qquad
C_{12}=\rho\sigma_1\sigma_2,
$$

gives

$$
\sigma_\Delta^2
=
\sigma_1^2+\sigma_2^2
-
2\rho\sigma_1\sigma_2.
$$

Therefore the correlation corresponding to a desired differential-noise variance is

$$
\boxed{
\rho^*
=
\frac{
\sigma_1^2+\sigma_2^2-\sigma_{\Delta,\mathrm{opt}}^2
}{
2\sigma_1\sigma_2
}
}.
$$

Here $\sigma_{\Delta,\mathrm{opt}}^2$ is the effective differential-noise strength that optimizes the synchronization measure for the reduced one-dimensional phase-difference dynamics.

Because a physical covariance matrix requires

$$
-1\le\rho\le1,
$$

the actual optimum is

$$
\boxed{
\rho_{\mathrm{phys}}^*
=
\operatorname{clip}
\left(
\frac{
\sigma_1^2+\sigma_2^2-\sigma_{\Delta,\mathrm{opt}}^2
}{
2\sigma_1\sigma_2
},
-1,
1
\right).
}
$$

### Why Clipping Favors Extreme Covariance Matrices

Suppose the unconstrained calculation asks for an effective differential-noise variance outside the range realizable by any valid correlation.

The allowed range is

$$
(\sigma_1-\sigma_2)^2
\le
\sigma_\Delta^2
\le
(\sigma_1+\sigma_2)^2.
$$

The lower endpoint occurs at

$$
\rho=1,
$$

and the upper endpoint occurs at

$$
\rho=-1.
$$

If the unconstrained optimum lies below the attainable interval, clipping gives $\rho=1$. If it lies above the interval, clipping gives $\rho=-1$.

This explains why optimal solutions often occur at extreme covariance structures:

- The objective may prefer more or less noise in a particular collective direction than the PSD constraint permits.
- The constrained optimum is then pushed to the boundary of the covariance set.
- Boundary points of the PSD cone are often singular or low rank.

In the two-oscillator case, the boundary consists of the perfectly correlated and perfectly anticorrelated matrices. In larger networks, the analogous phenomenon produces covariance matrices with extreme eigenvalues and often low-rank structure.

---

## 5. Synchronization Measure

The Kuramoto order parameter is

$$
R(t)e^{i\psi(t)}
=
\frac{1}{N}
\sum_{j=1}^{N}
e^{i\theta_j(t)}.
$$

Its squared magnitude is

$$
R^2
=
\frac{1}{N^2}
\sum_{i,j}
\cos(\theta_i-\theta_j).
$$

Interpretation:

- $R=1$: all phases are equal.
- $R\approx0$: phases are broadly dispersed.
- Intermediate values quantify partial synchrony.

The quantity optimized in the paper is the expected squared order parameter,

$$
\langle R^2\rangle.
$$

---

## 6. Expansion Around a Phase-Locked State

Assume the oscillators have reached a frequency-locked state with common angular frequency $\Omega$. Write

$$
\theta_i(t)
=
\Omega t
+
\bar{\theta}_i
+
\epsilon_i(t),
$$

where:

- $\bar{\theta}_i$ is the fixed phase offset in the co-rotating frame.
- $\epsilon_i(t)$ is a small noise-induced deviation.

The locked phases satisfy

$$
0
=
\omega_i-\Omega
+
\sum_j
K_{ij}\sin(\bar{\theta}_j-\bar{\theta}_i).
$$

The common term $\Omega t$ cancels from all phase differences, so

$$
R^2(\theta)
=
R^2(\bar{\theta}+\epsilon).
$$

Taylor-expand around $\bar{\theta}$:

$$
R^2(\bar{\theta}+\epsilon)
=
R_0^2
+
J^T\epsilon
+
\frac{1}{2}\epsilon^T
\left[
\nabla^2 R^2
\right]_{\bar{\theta}}
\epsilon
+
O(\|\epsilon\|^3).
$$

Define

$$
H
=
\frac{1}{2}
\left[
\nabla^2R^2
\right]_{\bar{\theta}}.
$$

Then

$$
R^2
\approx
R_0^2
+
J^T\epsilon
+
\epsilon^T H\epsilon.
$$

At the stationary noisy state,

$$
\langle\epsilon\rangle=0,
$$

so the linear term disappears:

$$
\langle R^2\rangle
\approx
R_0^2
+
\left\langle
\epsilon^T H\epsilon
\right\rangle.
$$

Using

$$
E
=
\langle\epsilon\epsilon^T\rangle,
$$

and the trace identity

$$
\epsilon^T H\epsilon
=
\operatorname{tr}
\left(
H\epsilon\epsilon^T
\right),
$$

we obtain

$$
\boxed{
\langle R^2\rangle
\approx
R_0^2
+
\operatorname{tr}(HE).
}
$$

Because $R_0^2$ is fixed once the deterministic operating state is fixed, maximizing synchronization at this order is equivalent to maximizing

$$
\operatorname{tr}(HE).
$$

---

## 7. Linearized Fluctuation Dynamics

Insert

$$
\theta_i
=
\Omega t+\bar{\theta}_i+\epsilon_i
$$

into the Kuramoto equation.

For small deviations,

$$
\sin\left[
(\bar{\theta}_j+\epsilon_j)
-
(\bar{\theta}_i+\epsilon_i)
\right]
$$

is approximated by

$$
\sin(\bar{\theta}_j-\bar{\theta}_i)
+
\cos(\bar{\theta}_j-\bar{\theta}_i)
(\epsilon_j-\epsilon_i).
$$

The zeroth-order terms cancel because $\bar{\theta}$ is a locked solution. The remaining dynamics are

$$
\dot{\epsilon}_i
=
\sum_j
K_{ij}
\cos(\bar{\theta}_j-\bar{\theta}_i)
(\epsilon_j-\epsilon_i)
+
\eta_i.
$$

Define the linearized matrix $L$ by

$$
L_{ij}
=
K_{ij}
\cos(\bar{\theta}_j-\bar{\theta}_i),
\qquad i\ne j,
$$

and

$$
L_{ii}
=
-
\sum_{j\ne i}
K_{ij}
\cos(\bar{\theta}_j-\bar{\theta}_i).
$$

Then

$$
\boxed{
\dot{\epsilon}
=
L\epsilon+\eta.
}
$$

For a stable locked state, every nontrivial eigenvalue of $L$ has negative real part. One zero mode remains because a uniform phase shift does not change the physical state. This mode is removed by working in the centered or co-rotating subspace.

---

## 8. From Noise Covariance $C$ to Phase Covariance $E$

Define the stationary covariance of the phase deviations:

$$
E
=
\langle
\epsilon(t)\epsilon(t)^T
\rangle.
$$

The linear stochastic dynamics are

$$
\dot{\epsilon}
=
L\epsilon+\eta.
$$

Rather than reproduce the full stochastic-calculus derivation, use the formal solution of the stable linear system. After transients have decayed,

$$
\epsilon(t)
=
\int_{-\infty}^{t}
e^{L(t-s)}
\eta(s)\,ds.
$$

Substitute this into $E$:

$$
E
=
\left\langle
\int_{-\infty}^{t}
e^{L(t-s)}
\eta(s)\,ds
\int_{-\infty}^{t}
\eta(s')^T
e^{L^T(t-s')}\,ds'
\right\rangle.
$$

Because the noise is white,

$$
\langle
\eta(s)\eta(s')^T
\rangle
=
C\delta(s-s'),
$$

the double integral collapses to

$$
\boxed{
E
=
\int_0^\infty
e^{Ls}
C
e^{L^Ts}
\,ds.
}
$$

This expression shows directly how the dynamics filter the input covariance $C$ into the state covariance $E$.

Now multiply by $L$ on the left and $L^T$ on the right:

$$
LE+EL^T
=
\int_0^\infty
\left[
Le^{Ls}Ce^{L^Ts}
+
e^{Ls}Ce^{L^Ts}L^T
\right]ds.
$$

The integrand is a total derivative:

$$
\frac{d}{ds}
\left(
e^{Ls}Ce^{L^Ts}
\right)
=
Le^{Ls}Ce^{L^Ts}
+
e^{Ls}Ce^{L^Ts}L^T.
$$

Therefore,

$$
LE+EL^T
=
\left[
e^{Ls}Ce^{L^Ts}
\right]_{s=0}^{s=\infty}.
$$

Stability makes the upper limit vanish, while the lower limit is $C$. Hence

$$
\boxed{
LE+EL^T=-C.
}
$$

If the coupling is undirected and $L=L^T$, this becomes

$$
\boxed{
LE+EL=-C.
}
$$

This is the continuous Lyapunov equation.

It gives the required link:

$$
C
\longrightarrow
E
\longrightarrow
\langle R^2\rangle.
$$

The noise covariance determines the phase covariance through the linearized dynamics, and the phase covariance determines the synchronization correction through the Hessian.

---

## 9. Semidefinite Optimization Problem

The second-order synchronization objective is

$$
\max_{C,E}
\operatorname{tr}(HE)
$$

subject to

$$
LE+EL^T=-C.
$$

The covariance must satisfy

$$
C\succeq0.
$$

To prevent the optimizer from increasing the total noise power without bound, fix the marginal noise variances:

$$
C_{ii}=1
\qquad
\text{for all }i.
$$

Because the global phase mode is physically irrelevant, impose centered-frame constraints such as

$$
C\mathbf{1}=0.
$$

The resulting problem is a semidefinite program because:

- The objective is linear in $E$.
- The Lyapunov equation is linear in $C$ and $E$.
- The diagonal and centering conditions are linear.
- $C\succeq0$ is a semidefinite-cone constraint.

---

## 10. What COSMO Solves

COSMO is a conic optimization solver. Its generic problem form is

$$
\min_x
\frac{1}{2}x^TPx+q^Tx
$$

subject to

$$
Ax+s=b,
\qquad
s\in\mathcal{K},
$$

where $\mathcal{K}$ is a product of cones, such as:

- The zero cone for equality constraints.
- The nonnegative orthant.
- Second-order cones.
- Positive-semidefinite cones.

Our problem is linear, so

$$
P=0.
$$

Because COSMO is written in terms of vectors, symmetric matrix variables must be represented as vectors.

---

## 11. Vectorizing Symmetric Matrices

For a symmetric matrix $X$, define

$$
x=\operatorname{svec}(X),
$$

where $\operatorname{svec}$ stores the upper or lower triangular entries and scales off-diagonal terms by $\sqrt{2}$.

For example,

$$
X=
\begin{pmatrix}
x_{11} & x_{12}\\
x_{12} & x_{22}
\end{pmatrix}
$$

may be mapped to

$$
\operatorname{svec}(X)
=
\begin{pmatrix}
x_{11}\\
\sqrt{2}x_{12}\\
x_{22}
\end{pmatrix}.
$$

The $\sqrt{2}$ scaling preserves the matrix inner product:

$$
\operatorname{tr}(XY)
=
\operatorname{svec}(X)^T
\operatorname{svec}(Y).
$$

Therefore,

$$
\operatorname{tr}(HE)
=
\operatorname{svec}(H)^T
\operatorname{svec}(E).
$$

Let

$$
e=\operatorname{svec}(E),
\qquad
c=\operatorname{svec}(C),
$$

and combine them into one decision vector:

$$
x=
\begin{pmatrix}
e\\
c
\end{pmatrix}.
$$

---

## 12. Vectorizing the Lyapunov Equation

The map

$$
E\mapsto LE+EL^T
$$

is linear. Therefore there exists a matrix $A_L$ such that

$$
\operatorname{svec}(LE+EL^T)
=
A_L\operatorname{svec}(E).
$$

The Lyapunov constraint becomes

$$
A_Le+c=0.
$$

Equivalently,

$$
\begin{pmatrix}
A_L & I
\end{pmatrix}
\begin{pmatrix}
e\\
c
\end{pmatrix}
=
0.
$$

The objective can be written as

$$
\max
\begin{pmatrix}
\operatorname{svec}(H)\\
0
\end{pmatrix}^T
x.
$$

Since COSMO uses minimization, the linear objective vector is

$$
q=
-
\begin{pmatrix}
\operatorname{svec}(H)\\
0
\end{pmatrix}.
$$

The PSD constraint is expressed by requiring the portion of the conic variable corresponding to $c$ to lie in the semidefinite cone:

$$
C=\operatorname{smat}(c)\succeq0.
$$
