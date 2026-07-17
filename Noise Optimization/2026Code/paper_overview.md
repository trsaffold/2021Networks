# Optimizing Synchronization by Shaping Correlated Noise

## 1. The Problem

Consider a network of coupled oscillators driven by Gaussian white noise. The oscillator dynamics and network couplings are assumed to be fixed. The control variable is instead the **spatial covariance structure of the noise** applied across the oscillators.

The central question is:

> Given fixed oscillator dynamics, fixed network couplings, and fixed marginal noise amplitudes, how should the noise acting on different oscillators be correlated in order to maximize synchronization?

Write the noise vector as

```text
η(t) = [η₁(t), ..., η_N(t)]ᵀ
```

with

```text
<η(t)> = 0
```

and

```text
<η(t) η(t')ᵀ> = C δ(t - t')
```

The matrix `C` is the noise covariance matrix. Its diagonal entries specify the variance applied to each oscillator, while its off-diagonal entries specify the correlations between the noise inputs.

The optimization problem is therefore not to remove noise, but to choose an admissible `C` whose correlations make the oscillator network as synchronized as possible.

---

## 2. Oscillator Dynamics

### 2.1 Kuramoto Dynamics

The paper uses the noisy network Kuramoto model

```text
dθᵢ/dt = ωᵢ + Σⱼ Kᵢⱼ sin(θⱼ - θᵢ) + ηᵢ(t)
```

Here:

- `θᵢ(t)` is the phase of oscillator `i`.
- `ωᵢ` is its natural frequency.
- `Kᵢⱼ` is the coupling strength from oscillator `j` to oscillator `i`.
- `ηᵢ(t)` is the Gaussian white-noise input.

The coupling depends only on phase differences, so adding the same phase to every oscillator does not change the dynamics.

### 2.2 Kuramoto as the First Fourier Harmonic

The sine interaction is not the only possible coupling law. A general phase-reduced oscillator network can be written as

```text
dθᵢ/dt = ωᵢ + Σⱼ Γᵢⱼ(θⱼ - θᵢ) + ηᵢ(t)
```

where each `2π`-periodic coupling function can be expanded in a Fourier series:

```text
Γᵢⱼ(φ) = aᵢⱼ,₀ + Σₘ₌₁..∞ [aᵢⱼ,ₘ cos(mφ) + bᵢⱼ,ₘ sin(mφ)]
```

The Kuramoto model retains only the leading sinusoidal component:

```text
Γᵢⱼ(φ) ≈ Kᵢⱼ sin(φ)
```

It is therefore the first Fourier-harmonic approximation to a broader class of weakly coupled phase-oscillator models.

### 2.3 An Alternative: The Swing Equation

The same noise-covariance framework can be applied to other oscillator dynamics. In power-grid models, for example, one commonly uses the second-order swing equation

```text
Mᵢ d²θᵢ/dt² + Dᵢ dθᵢ/dt = Pᵢ + Σⱼ Kᵢⱼ sin(θⱼ - θᵢ) + ηᵢ(t)
```

Here:

- `Mᵢ` is inertia.
- `Dᵢ` is damping.
- `Pᵢ` is the net mechanical or electrical power injection.

The Kuramoto equation is first order in phase, whereas the swing equation includes angular velocity as an additional state variable. The optimization logic remains similar: linearize the chosen dynamics around a synchronized operating state, determine how noise covariance propagates into state covariance, and optimize the resulting synchronization measure.

---

## 3. Why a Covariance Matrix Must Be Positive Semidefinite

A valid covariance matrix must satisfy

```text
C = Cᵀ
C ⪰ 0
```

The reason is that every linear combination of the noise variables must have nonnegative variance. For any vector `a`, the scalar random input is

```text
aᵀη
```

Its variance is

```text
Var(aᵀη) = <(aᵀη)²> = aᵀ C a
```

A variance cannot be negative, so

```text
aᵀ C a ≥ 0 for every a
```

This is exactly the definition of positive semidefiniteness.

For two oscillators,

```text
C = [ σ₁²        ρσ₁σ₂ ]
    [ ρσ₁σ₂     σ₂²   ]
```

The PSD condition gives

```text
det(C) = σ₁² σ₂² (1 - ρ²) ≥ 0
```

and therefore

```text
-1 ≤ ρ ≤ 1
```

The endpoints `ρ = ±1` are singular, rank-one covariance matrices. They represent perfectly correlated or perfectly anticorrelated noise.

---

## 4. Two-Oscillator Reduction

Consider two oscillators:

```text
dθ₁/dt = ω₁ + (K/2) sin(θ₂ - θ₁) + η₁
dθ₂/dt = ω₂ + (K/2) sin(θ₁ - θ₂) + η₂
```

Define the phase difference

```text
δ = θ₁ - θ₂
```

and frequency difference

```text
Δω = ω₁ - ω₂
```

Subtracting the equations gives

```text
dδ/dt = Δω - K sin(δ) + Δη
```

where

```text
Δη = η₁ - η₂
```

Thus the synchronization of the pair depends only on the **differential noise**, not on noise in the common phase direction.

The differential-noise variance is

```text
σ²_Δ = <(η₁ - η₂)²>
```

Expanding,

```text
σ²_Δ = C₁₁ + C₂₂ - 2C₁₂
```

Writing

```text
C₁₁ = σ₁²
C₂₂ = σ₂²
C₁₂ = ρσ₁σ₂
```

gives

```text
σ²_Δ = σ₁² + σ₂² - 2ρσ₁σ₂
```

Therefore the correlation corresponding to a desired differential-noise variance is

```text
ρ* = (σ₁² + σ₂² - σ²_Δ,opt) / (2σ₁σ₂)
```

Here `σ²_Δ,opt` is the effective differential-noise strength that optimizes the synchronization measure for the reduced one-dimensional phase-difference dynamics.

Because a physical covariance matrix requires `-1 ≤ ρ ≤ 1`, the actual optimum is

```text
ρ*_phys = clip((σ₁² + σ₂² - σ²_Δ,opt) / (2σ₁σ₂), -1, 1)
```

### Why Clipping Favors Extreme Covariance Matrices

Suppose the unconstrained calculation asks for an effective differential-noise variance outside the range realizable by any valid correlation.

The allowed range is

```text
(σ₁ - σ₂)² ≤ σ²_Δ ≤ (σ₁ + σ₂)²
```

The lower endpoint occurs at `ρ = 1`, and the upper endpoint occurs at `ρ = -1`.

If the unconstrained optimum lies below the attainable interval, clipping gives `ρ = 1`. If it lies above the interval, clipping gives `ρ = -1`.

This explains why optimal solutions often occur at extreme covariance structures:

- The objective may prefer more or less noise in a particular collective direction than the PSD constraint permits.
- The constrained optimum is then pushed to the boundary of the covariance set.
- Boundary points of the PSD cone are often singular or low rank.

In the two-oscillator case, the boundary consists of the perfectly correlated and perfectly anticorrelated matrices. In larger networks, the analogous phenomenon produces covariance matrices with extreme eigenvalues and often low-rank structure.

---

## 5. Synchronization Measure

The Kuramoto order parameter is

```text
R(t) e^(iψ(t)) = (1/N) Σⱼ e^(iθⱼ(t))
```

Its squared magnitude is

```text
R² = (1/N²) Σᵢⱼ cos(θᵢ - θⱼ)
```

Interpretation:

- `R = 1`: all phases are equal.
- `R ≈ 0`: phases are broadly dispersed.
- Intermediate values quantify partial synchrony.

The quantity optimized in the paper is the expected squared order parameter,

```text
<R²>
```

---

## 6. Expansion Around a Phase-Locked State

Assume the oscillators have reached a frequency-locked state with common angular frequency `Ω`. Write

```text
θᵢ(t) = Ωt + θ̄ᵢ + εᵢ(t)
```

where:

- `θ̄ᵢ` is the fixed phase offset in the co-rotating frame.
- `εᵢ(t)` is a small noise-induced deviation.

The locked phases satisfy

```text
0 = ωᵢ - Ω + Σⱼ Kᵢⱼ sin(θ̄ⱼ - θ̄ᵢ)
```

The common term `Ωt` cancels from all phase differences, so

```text
R²(θ) = R²(θ̄ + ε)
```

Taylor-expand around `θ̄`:

```text
R²(θ̄ + ε)
≈ R₀² + Jᵀε + (1/2) εᵀ [∇²R²]_(θ̄) ε + O(||ε||³)
```

Define

```text
H = (1/2) [∇²R²]_(θ̄)
```

Then

```text
R² ≈ R₀² + Jᵀε + εᵀHε
```

At the stationary noisy state, `<ε> = 0`, so the linear term disappears:

```text
<R²> ≈ R₀² + <εᵀHε>
```

Using

```text
E = <εεᵀ>
```

and the trace identity

```text
εᵀHε = tr(Hεεᵀ)
```

we obtain

```text
<R²> ≈ R₀² + tr(HE)
```

Because `R₀²` is fixed once the deterministic operating state is fixed, maximizing synchronization at this order is equivalent to maximizing

```text
tr(HE)
```

---

## 7. Linearized Fluctuation Dynamics

Insert

```text
θᵢ = Ωt + θ̄ᵢ + εᵢ
```

into the Kuramoto equation.

For small deviations,

```text
sin((θ̄ⱼ + εⱼ) - (θ̄ᵢ + εᵢ))
```

is approximated by

```text
sin(θ̄ⱼ - θ̄ᵢ) + cos(θ̄ⱼ - θ̄ᵢ)(εⱼ - εᵢ)
```

The zeroth-order terms cancel because `θ̄` is a locked solution. The remaining dynamics are

```text
dεᵢ/dt = Σⱼ Kᵢⱼ cos(θ̄ⱼ - θ̄ᵢ)(εⱼ - εᵢ) + ηᵢ
```

Define the linearized matrix `L` by

```text
Lᵢⱼ = Kᵢⱼ cos(θ̄ⱼ - θ̄ᵢ),     i ≠ j
```

and

```text
Lᵢᵢ = -Σⱼ≠ᵢ Kᵢⱼ cos(θ̄ⱼ - θ̄ᵢ)
```

Then

```text
dε/dt = Lε + η
```

For a stable locked state, every nontrivial eigenvalue of `L` has negative real part. One zero mode remains because a uniform phase shift does not change the physical state. This mode is removed by working in the centered or co-rotating subspace.

---

## 8. From Noise Covariance `C` to Phase Covariance `E`

Define the stationary covariance of the phase deviations:

```text
E = <ε(t) ε(t)ᵀ>
```

The linear stochastic dynamics are

```text
dε/dt = Lε + η
```

Rather than reproduce the full stochastic-calculus derivation, use the formal solution of the stable linear system. After transients have decayed,

```text
ε(t) = ∫ from -∞ to t of exp(L(t-s)) η(s) ds
```

Substitute this into `E`:

```text
E =
<
  ∫ exp(L(t-s)) η(s) ds
  ∫ η(s')ᵀ exp(Lᵀ(t-s')) ds'
>
```

Because the noise is white,

```text
<η(s)η(s')ᵀ> = C δ(s - s')
```

the double integral collapses to

```text
E = ∫ from 0 to ∞ of exp(Ls) C exp(Lᵀs) ds
```

This expression shows directly how the dynamics filter the input covariance `C` into the state covariance `E`.

Now multiply by `L` on the left and `Lᵀ` on the right:

```text
LE + ELᵀ =
∫ from 0 to ∞ of [L exp(Ls) C exp(Lᵀs) + exp(Ls) C exp(Lᵀs) Lᵀ] ds
```

The integrand is a total derivative:

```text
d/ds [exp(Ls) C exp(Lᵀs)]
= L exp(Ls) C exp(Lᵀs) + exp(Ls) C exp(Lᵀs) Lᵀ
```

Therefore,

```text
LE + ELᵀ = [exp(Ls) C exp(Lᵀs)] from s=0 to s=∞
```

Stability makes the upper limit vanish, while the lower limit is `C`. Hence

```text
LE + ELᵀ = -C
```

If the coupling is undirected and `L = Lᵀ`, this becomes

```text
LE + EL = -C
```

This is the continuous Lyapunov equation.

It gives the required link:

```text
C → E → <R²>
```

The noise covariance determines the phase covariance through the linearized dynamics, and the phase covariance determines the synchronization correction through the Hessian.

---

## 9. Semidefinite Optimization Problem

The second-order synchronization objective is

```text
maximize tr(HE) over C and E
```

subject to

```text
LE + ELᵀ = -C
```

The covariance must satisfy

```text
C ⪰ 0
```

To prevent the optimizer from increasing the total noise power without bound, fix the marginal noise variances:

```text
Cᵢᵢ = 1 for all i
```

Because the global phase mode is physically irrelevant, impose centered-frame constraints such as

```text
C 1 = 0
```

The resulting problem is a semidefinite program because:

- The objective is linear in `E`.
- The Lyapunov equation is linear in `C` and `E`.
- The diagonal and centering conditions are linear.
- `C ⪰ 0` is a semidefinite-cone constraint.

---

## 10. What COSMO Solves

COSMO is a conic optimization solver. Its generic problem form is

```text
minimize (1/2)xᵀPx + qᵀx
```

subject to

```text
Ax + s = b,
s ∈ K
```

where `K` is a product of cones, such as:

- The zero cone for equality constraints.
- The nonnegative orthant.
- Second-order cones.
- Positive-semidefinite cones.

Our problem is linear, so

```text
P = 0
```

Because COSMO is written in terms of vectors, symmetric matrix variables must be represented as vectors.

---

## 11. Vectorizing Symmetric Matrices

For a symmetric matrix `X`, define

```text
x = svec(X)
```

where `svec` stores the upper or lower triangular entries and scales off-diagonal terms by `√2`.

For example,

```text
X = [ x₁₁   x₁₂ ]
    [ x₁₂   x₂₂ ]
```

may be mapped to

```text
svec(X) = [x₁₁, √2 x₁₂, x₂₂]ᵀ
```

The `√2` scaling preserves the matrix inner product:

```text
tr(XY) = svec(X)ᵀ svec(Y)
```

Therefore,

```text
tr(HE) = svec(H)ᵀ svec(E)
```

Let

```text
e = svec(E)
c = svec(C)
x = [e, c]ᵀ
```

---

## 12. Vectorizing the Lyapunov Equation

The map

```text
E ↦ LE + ELᵀ
```

is linear. Therefore there exists a matrix `A_L` such that

```text
svec(LE + ELᵀ) = A_L svec(E)
```

The Lyapunov constraint becomes

```text
A_L e + c = 0
```

Equivalently,

```text
[A_L  I] [e, c]ᵀ = 0
```

The objective can be written as

```text
maximize [svec(H), 0]ᵀ x
```

Since COSMO uses minimization, the linear objective vector is

```text
q = -[svec(H), 0]
```

The PSD constraint is expressed by requiring the portion of the conic variable corresponding to `c` to lie in the semidefinite cone:

```text
C = smat(c) ⪰ 0
```
