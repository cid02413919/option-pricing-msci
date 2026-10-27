# Heston Model Study Guide: Three Levels from Intuition to Implementation

## Purpose and scope

This guide explains the uploaded chapter **“The Heston Model for European Options”** in three passes. It assumes you already understand the basic meaning of an SDE, Brownian motion, Itô's lemma, and the geometric-Brownian-motion stock model at a high level. It builds the rest from there.

The target is not merely to memorise the Heston formula. By the end, you should be able to:

- state exactly what the model assumes under both real-world and pricing measures;
- explain why Heston improves on constant-volatility Black–Scholes;
- read and derive the Heston pricing PDE;
- understand what a characteristic function is and why it replaces a density calculation;
- see why the exponential-affine ansatz works and why it produces a Riccati ODE;
- implement and test a European-option pricer;
- understand what a Heston-informed PINN is enforcing, and what it is *not* learning.

The guide focuses on the Heston chapter. The earlier *Risk, randomness, crashes and quants* paper supplies the conceptual path: random walks lead to Brownian motion; Brownian motion leads to geometric Brownian motion; Black–Scholes uses constant volatility; observed changing volatility and non-normal return behaviour motivate stochastic-volatility models such as Heston.

---

# Notation first: avoid the most common confusion

Different books reuse symbols differently. In this guide, use the following convention under the **risk-neutral pricing measure** $$\mathbb Q$$:

$$
dS_t=(r-q)S_t\,dt+\sqrt{v_t}\,S_t\,dW^S_t,
$$

$$
dv_t=\kappa(\theta-v_t)\,dt+\xi\sqrt{v_t}\,dW^v_t,
$$

$$
dW^S_t\,dW^v_t=\rho\,dt.
$$

| Symbol | Meaning | Typical restriction |
|---|---|---|
| $$S_t$$ | asset or share price | $$S_t>0$$ |
| $$v_t$$ | **instantaneous variance** | $$v_t\geq0$$ |
| $$\sqrt{v_t}$$ | instantaneous volatility | non-negative |
| $$r$$ | continuously compounded risk-free rate | often treated as deterministic |
| $$q$$ | continuous dividend yield | often $$q=0$$ in a first derivation |
| $$\kappa$$ | speed of mean reversion of variance | $$\kappa>0$$ |
| $$\theta$$ | long-run mean variance | $$\theta>0$$ |
| $$\xi$$ | volatility of variance, usually called *vol-of-vol* | $$\xi>0$$ |
| $$\rho$$ | correlation of price and variance shocks | $$-1\leq\rho\leq1$$ |
| $$v_0$$ | current or initial variance | $$v_0\geq0$$ |
| $$T$$ | expiry time | $$T>t$$ |
| $$\tau=T-t$$ | time to expiry | $$\tau\geq0$$ |
| $$K$$ | strike price | $$K>0$$ |

The uploaded chapter calls the vol-of-vol parameter $$\sigma$$. Here it is called $$\xi$$ so it cannot be confused with Black–Scholes volatility. In Black–Scholes, volatility is constant. In Heston, **variance is random**, and the volatility at a moment is $$\sqrt{v_t}$$.

---

# Level 1 — Conceptual understanding

## 1. The entire story in one paragraph

Black–Scholes assumes the asset has one fixed volatility. Real markets do not: turbulent periods and calm periods cluster, and implied volatilities differ across strikes and maturities. Heston keeps the same basic random asset-price idea, but makes its instantaneous variance a second random process. That variance is mean-reverting and non-negative. The correlation between shocks to returns and shocks to variance creates skew. For European options, Heston is unusually useful because—even though variance is random—the characteristic function of log price can be obtained analytically up to simple complex-valued functions. Option prices then follow from one-dimensional numerical integrals.

## 2. Why Black–Scholes is insufficient

Under Black–Scholes, under $$\mathbb Q$$,

$$
dS_t=(r-q)S_tdt+\sigma_{\mathrm{BS}}S_tdW_t.
$$

The constant $$\sigma_{\mathrm{BS}}$$ implies that log returns over a fixed horizon are normal and that one volatility should price every strike and maturity for the same underlying.

In market option quotes, if you invert Black–Scholes separately for each option, the implied values of $$\sigma_{\mathrm{BS}}$$ form a **smile** or **skew**, not a flat surface. Equity markets commonly have a left skew: lower-strike puts imply higher volatility. The model is not literally saying each option has a different physical volatility; rather, the constant-volatility model is unable to fit the entire cross-section with one value.

## 3. The Heston fix

Heston replaces constant variance with a process:

$$
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW^v_t.
$$

Interpret each piece:

- $$\kappa(\theta-v_t)dt$$ pulls variance toward its typical level $$\theta$$. If variance is high, the drift is negative; if it is low, the drift is positive.
- $$\xi\sqrt{v_t}dW^v_t$$ adds random variation to variance itself.
- Multiplying the random term by $$\sqrt{v_t}$$ helps prevent variance from becoming negative.

The asset receives random shocks through $$dW^S_t$$. Variance receives shocks through $$dW^v_t$$. Their correlation is $$\rho$$.

- $$\rho<0$$: adverse price shocks tend to coincide with higher volatility. This is the usual equity *leverage effect* and generates a negatively skewed implied-volatility surface.
- Larger $$\xi$$: variance changes more violently, usually giving more curvature in the smile.
- Larger $$\kappa$$: volatility shocks fade faster, especially affecting term structure.
- Larger $$\theta$$: raises long-horizon average variance.
- Larger $$v_0$$: raises very short-maturity variance.

## 4. Physical versus risk-neutral worlds

This distinction is essential.

### Physical measure $$\mathbb P$$

This is intended to describe actual historical evolution. It is relevant for forecasting, historical simulation, and estimating realised dynamics.

### Risk-neutral measure $$\mathbb Q$$

This is a mathematical pricing measure. Under no arbitrage, prices of tradable assets, after discounting, must be martingales under $$\mathbb Q$$. Hence the asset drift changes to:

$$
r-q.
$$

A European payoff $$g(S_T)$$ has no-arbitrage price:

$$
V(S_t,v_t,t)=\mathbb E^{\mathbb Q}\left[e^{-r(T-t)}g(S_T)\mid S_t,v_t\right].
$$

This is **not** a statement that investors truly expect the asset to grow at $$r-q$$. It is a valuation device derived from hedging and no-arbitrage.

Variance is not directly traded. Consequently, its risk-neutral drift cannot be inferred solely by replacing the stock drift. A market price of variance risk must be specified or absorbed into the risk-neutral parameters. In the chapter, an affine choice produces risk-neutral parameters commonly denoted by $$\kappa^*$$ and $$\theta^*$$. In calibration to option prices, you usually estimate the risk-neutral parameters directly.

## 5. What is being priced?

For a European call, the expiry payoff is:

$$
(S_T-K)^+ = \max(S_T-K,0).
$$

It has no early exercise. Its time-$$t$$ price is:

$$
C_t=e^{-r\tau}\mathbb E^{\mathbb Q}\left[(S_T-K)^+\mid S_t,v_t\right].
$$

The entire chapter is a route to evaluate this expectation quickly and accurately.

## 6. Three equivalent viewpoints

The same price can be viewed in three ways:

1. **Expectation view**: discounted expected payoff under $$\mathbb Q$$.
2. **PDE view**: solve a backward equation with a terminal payoff condition.
3. **Characteristic-function view**: obtain probabilities or expectations by Fourier inversion.

For European Heston options, the characteristic-function route is popular. For PINNs, the PDE route is especially natural. Monte Carlo offers a fourth numerical view, but is often slower for vanilla European options.

## 7. The final pricing structure

For a call with continuous dividend yield, the Heston price has a Black–Scholes-like form:

$$
C_t=S_te^{-q\tau}P_1-Ke^{-r\tau}P_2.
$$

The quantities $$P_1$$ and $$P_2$$ are not usually simple normal-CDF values, unlike Black–Scholes. They are recovered from the characteristic function by numerical integration.

## 8. The key message for ML and PINNs

A Heston-informed PINN can learn an approximation to the function:

$$
V(S,v,t),
$$

while penalising violation of the Heston PDE and its terminal/boundary conditions. It does **not** automatically solve the harder statistical problem of identifying the correct risk-neutral parameter vector from sparse and noisy option prices. Calibration, PDE solution, and forecasting are different tasks.

---

# Level 2 — Working mathematical understanding

## 9. Prerequisite A: the CIR variance process

The Heston variance equation is the Cox–Ingersoll–Ross, or CIR, square-root process:

$$
dv_t=\kappa(\theta-v_t)dt+\xi\sqrt{v_t}dW^v_t.
$$

It resembles the Ornstein–Uhlenbeck mean-reverting equation, except the noise amplitude is $$\sqrt{v_t}$$ rather than a constant.

### Conditional mean

Conditioned on $$v_s$$ at time $$s$$, the mean at later time $$t$$ is:

$$
\mathbb E[v_t\mid v_s]
=
\theta+(v_s-\theta)e^{-\kappa(t-s)}.
$$

This tells you exactly how mean reversion works:

- as $$t-s\to0$$, the mean is approximately $$v_s$$;
- as $$t-s\to\infty$$, the mean tends to $$\theta$$;
- the characteristic reversion timescale is roughly $$1/\kappa$$.

### Variance positivity and the Feller condition

A sufficient condition for the CIR process to stay strictly positive is:

$$
2\kappa\theta\geq\xi^2.
$$

This is the **Feller condition**. If it holds, zero is unattainable from a positive start in the ideal continuous-time model. If it fails, the process can touch zero, though it remains non-negative under the appropriate mathematical interpretation.

Important practical point: a calibrated Heston model may violate the Feller condition and still be used. But simulation schemes and numerical methods must then handle the boundary carefully. Do not treat Feller as a universal calibration constraint without considering the application.

### Units sanity check

If time is measured in years, then:

- $$v$$ has units of inverse years;
- $$\theta$$ has units of inverse years;
- $$\kappa$$ has units of inverse years;
- $$\xi$$ has units of inverse years;
- $$\sqrt{v}dW$$ is dimensionless in a return equation.

This is one reason variance, not volatility, is modelled linearly in the drift.

## 10. Prerequisite B: correlated Brownian motions

The relation

$$
dW^S_tdW^v_t=\rho dt
$$

is Itô shorthand for instantaneous correlation. To simulate it using independent Brownian motions $$W^1$$ and $$W^2$$, write:

$$
dW^v_t=\rho dW^1_t+\sqrt{1-\rho^2}\,dW^2_t,
$$

$$
dW^S_t=dW^1_t.
$$

Thus the same shock partly drives both asset returns and variance. This shared component is how correlation enters the PDE and characteristic function.

## 11. Log-price dynamics

Set:

$$
x_t=\ln S_t.
$$

Apply Itô's lemma to the risk-neutral asset equation. Since the diffusion coefficient of $$S_t$$ is $$S_t\sqrt{v_t}$$, the log-price equation is:

$$
dx_t=\left(r-q-\frac{1}{2}v_t\right)dt+\sqrt{v_t}dW^S_t.
$$

The term $$-v_t/2$$ is the Itô correction. Conditional on the entire variance path, log price is Gaussian, but unconditionally it is a mixture over random variance paths. This mixture creates richer distributions than one fixed normal law.

## 12. The pricing PDE from a generator

For a sufficiently smooth function $$u(S,v,t)$$ of the state variables, the Heston infinitesimal generator under $$\mathbb Q$$ is:

$$
\mathcal L u
=
(r-q)S\frac{\partial u}{\partial S}
+
\kappa(\theta-v)\frac{\partial u}{\partial v}
+
\frac{1}{2}vS^2\frac{\partial^2u}{\partial S^2}
+
\rho\xi vS\frac{\partial^2u}{\partial S\partial v}
+
\frac{1}{2}\xi^2v\frac{\partial^2u}{\partial v^2}.
$$

The pricing PDE is:

$$
\frac{\partial V}{\partial t}+\mathcal LV-rV=0.
$$

Expanded:

$$
\frac{\partial V}{\partial t}
+
\frac{1}{2}vS^2\frac{\partial^2V}{\partial S^2}
+
\rho\xi vS\frac{\partial^2V}{\partial S\partial v}
+
\frac{1}{2}\xi^2v\frac{\partial^2V}{\partial v^2}
+
(r-q)S\frac{\partial V}{\partial S}
+
\kappa(\theta-v)\frac{\partial V}{\partial v}
-rV=0.
$$

For the call option, the terminal condition is:

$$
V(S,v,T)=(S-K)^+.
$$

### Where each second derivative comes from

The second-derivative terms are not arbitrary:

- $$V_{SS}$$ comes from random motion in $$S$$;
- $$V_{vv}$$ comes from random motion in $$v$$;
- $$V_{Sv}$$ comes from correlated shocks, because $$dS_tdv_t=\rho\xi v_tS_tdt$$.

This cross derivative is often a technical challenge for finite-difference solvers and PINNs.

### Why the chapter uses two derivatives in its hedge

In Black–Scholes, one Brownian risk can be hedged with the stock. Heston has two Brownian sources. The chapter introduces a second variance-sensitive derivative $$U(S,v,t)$$ and forms a portfolio involving $$V$$, $$U$$, and the stock to eliminate both local diffusion terms. This motivates the PDE and exposes the need to specify a variance risk premium.

In modern treatments, one may derive the same PDE more compactly by declaring the risk-neutral SDE and applying Feynman–Kac. Both are valid routes. The hedge derivation should not be read as a literal claim that every market has a perfectly liquid instrument with which variance risk can always be continuously hedged.

## 13. Backward time form

Define time to maturity:

$$
\tau=T-t.
$$

Then:

$$
\frac{\partial V}{\partial t}=-\frac{\partial V}{\partial\tau}.
$$

The PDE becomes an initial-value problem in $$\tau$$ with known initial condition:

$$
V(S,v,0)=(S-K)^+.
$$

PINN implementations often prefer this form because the known payoff is imposed at $$\tau=0$$.

## 14. A characteristic function from scratch

For a real random variable $$X$$, its characteristic function is:

$$
\varphi_X(u)=\mathbb E[e^{iuX}],
$$

where:

$$
i^2=-1.
$$

This is a Fourier transform of the probability distribution.

Why is it useful?

- A probability density may be difficult to derive explicitly.
- Expectations of exponentials often remain tractable in affine stochastic models.
- Fourier inversion can recover the density, distribution function, or the option probabilities needed for price calculation.

### Familiar normal example

If:

$$
X\sim N(m,s^2),
$$

then:

$$
\varphi_X(u)=\exp\left(iu m-\frac{1}{2}u^2s^2\right).
$$

The characteristic function uniquely determines the distribution. This is analogous to how the Fourier transform uniquely encodes a signal.

### What Heston needs

The relevant random variable is terminal log price:

$$
x_T=\ln S_T.
$$

Heston gives a tractable formula for:

$$
f(u;\tau,x_t,v_t)=\mathbb E^{\mathbb Q}[e^{iux_T}\mid x_t,v_t].
$$

It is conditional because the current state $$x_t,v_t$$ matters.

## 15. From characteristic function to exercise probability

For a suitable characteristic function, a Gil–Pelaez/Fourier inversion formula gives:

$$
\Pr(x_T>\ln K)
=
\frac12+\frac1\pi\int_0^\infty
\operatorname{Re}\left[
\frac{e^{-iu\ln K}\varphi_{x_T}(u)}{iu}
\right]du.
$$

The symbols mean:

- $$u$$ is the Fourier integration variable, not a variance;
- $$\operatorname{Re}$$ takes the real part of a complex number;
- the integral reconstructs a cumulative probability from the characteristic function.

For option pricing, the two terms $$P_1$$ and $$P_2$$ arise under two related measures. A useful safe interpretation is:

- $$P_2$$ is the risk-neutral probability that the call expires in the money;
- $$P_1$$ is an asset-numeraire-weighted analogue, not generally the same ordinary probability.

This distinction prevents a common misconception: both are between zero and one in ordinary settings, but they do not describe identical events under the same probability weighting.

## 16. Numeraire and the two measures: enough to read the chapter

A **numeraire** is the asset used as the unit in which values are measured.

- Under the usual risk-neutral measure, the money-market account is the numeraire.
- Under the stock/asset measure, the stock adjusted for dividends plays the numeraire role.

Changing numeraire changes the probability measure but leaves correctly priced assets consistent. The payoff decomposition explains why:

$$
C_t=e^{-r\tau}\mathbb E^{\mathbb Q}[S_T\mathbf1_{S_T>K}]
-Ke^{-r\tau}\mathbb Q(S_T>K).
$$

The first expectation is naturally rewritten under the stock-related measure, yielding:

$$
C_t=S_te^{-q\tau}P_1-Ke^{-r\tau}P_2.
$$

You do not need measure-change machinery to code the usual standard Heston formula, but you need this idea to understand why the call price contains **two** Fourier inversions/characteristic functions in the chapter's derivation.

---

# Level 3 — Full Heston derivation and implementation-level understanding

## 17. Real-world versus risk-neutral Heston dynamics

A physical-measure model might be written:

$$
dS_t=\mu S_tdt+\sqrt{v_t}S_tdW^{S,\mathbb P}_t,
$$

$$
dv_t=\kappa_{\mathbb P}(\theta_{\mathbb P}-v_t)dt+\xi\sqrt{v_t}dW^{v,\mathbb P}_t.
$$

Under the pricing measure, write directly:

$$
dS_t=(r-q)S_tdt+\sqrt{v_t}S_tdW^{S,\mathbb Q}_t,
$$

$$
dv_t=\kappa_{\mathbb Q}(\theta_{\mathbb Q}-v_t)dt+\xi\sqrt{v_t}dW^{v,\mathbb Q}_t.
$$

The chapter starts with a particular volatility-risk-premium specification that transforms physical parameters into starred risk-neutral parameters. For option pricing/calibration, use the latter directly and often drop the $$\mathbb Q$$ subscript for readability.

**Important modelling point:** the stock is traded but instantaneous variance is not. The market is therefore generally incomplete, so an additional assumption about the variance risk premium is needed to select a pricing measure. Option calibration implicitly selects risk-neutral variance dynamics.

## 18. Deriving the Heston PDE using multivariate Itô calculus

Let $$V=V(S,v,t)$$. Multivariate Itô's lemma gives:

$$
dV=V_tdt+V_SdS+V_vdv
+\frac12V_{SS}(dS)^2+V_{Sv}dSdv+\frac12V_{vv}(dv)^2.
$$

Using the SDEs:

$$
(dS)^2=vS^2dt,
$$

$$
(dv)^2=\xi^2vdt,
$$

$$
dSdv=\rho\xi vSdt.
$$

Substituting gives the drift and the two diffusion terms. Under the risk-neutral dynamics, Feynman–Kac immediately yields the PDE in Section 12. The chapter instead creates a two-derivative hedge to eliminate both diffusion sources, then imposes that the locally riskless portfolio earns $$r$$. Both produce the same risk-neutral equation once the variance risk premium is fixed.

## 19. Log-price Heston PDE

Put:

$$
x=\ln S.
$$

Useful derivative identities are:

$$
V_S=\frac{1}{S}V_x,
$$

$$
V_{SS}=\frac{1}{S^2}(V_{xx}-V_x),
$$

$$
V_{Sv}=\frac{1}{S}V_{xv}.
$$

The PDE becomes:

$$
V_t
+
\frac12vV_{xx}
+
\rho\xi vV_{xv}
+
\frac12\xi^2vV_{vv}
+
\left(r-q-\frac12v\right)V_x
+
\kappa(\theta-v)V_v
-rV=0.
$$

This form is important for Heston PINNs because it removes large scaling factors involving $$S$$ and defines the computational state in the natural log-price coordinate.

## 20. Why Heston is called an affine model

The coefficients in the log-price/variance generator are either:

- constant, or
- linear in $$v$$.

For example, the diffusion variance of log price is $$v$$; the variance drift is $$\kappa\theta-\kappa v$$; the correlation term is $$\rho\xi v$$.

This affine structure makes an exponential-affine conditional transform plausible:

$$
f(u;\tau,x,v)
=
\mathbb E^{\mathbb Q}[e^{iux_T}\mid x_t=x,v_t=v]
=
\exp\left(C(\tau,u)+D(\tau,u)v+iux\right).
$$

This is the central ansatz. It says the logarithm of the characteristic function is affine in the current state $$x,v$$.

Why it makes sense:

- dependence on $$x$$ must include $$e^{iux}$$ because terminal log price shifts one-for-one with current log price;
- the remaining coefficients of the generator are at most linear in $$v$$, so after differentiating the exponential ansatz, the PDE can be separated into a constant term and a term proportional to $$v$$;
- setting the coefficients of each independently to zero gives ODEs for $$C$$ and $$D$$.

## 21. The Riccati equation appears

Use time to maturity $$\tau$$. For a generic transform satisfying the relevant backward equation, substitution of the ansatz yields an equation of the schematic form:

$$
\frac{\partial D}{\partial\tau}
=
A(u)+B(u)D+\frac12\xi^2D^2,
$$

$$
\frac{\partial C}{\partial\tau}
=
i u(r-q)+\kappa\theta D.
$$

The first is a **Riccati differential equation**: a first-order ODE containing a quadratic term in the unknown function.

For the ordinary risk-neutral characteristic function of $$x_T$$, one common convention is:

$$
D'(\tau)
=
\frac12\xi^2D(\tau)^2+
(\rho\xi iu-\kappa)D(\tau)
-
\frac12(u^2+iu),
$$

$$
C'(\tau)=iu(r-q)+\kappa\theta D(\tau),
$$

with:

$$
D(0)=0,
$$

$$
C(0)=0.
$$

Different sources have superficially different versions because they use different signs for Fourier transforms, different definitions of $$\tau$$, and sometimes two different measures for $$P_1$$ and $$P_2$$. Check conventions before copying code across sources.

## 22. Solving the Riccati equation

For a generic constant-coefficient Riccati equation,

$$
y'=P+Qy+Ry^2,
$$

a standard substitution converts it to a second-order linear ODE. In Heston, the resulting solution is conveniently written using complex quantities:

$$
d(u)=\sqrt{(\rho\xi iu-\kappa)^2+\xi^2(u^2+iu)},
$$

$$
g(u)=\frac{\kappa-\rho\xi iu-d(u)}{\kappa-\rho\xi iu+d(u)}.
$$

Under one common convention:

$$
D(\tau,u)=
\frac{\kappa-\rho\xi iu-d(u)}{\xi^2}
\frac{1-e^{-d(u)\tau}}{1-g(u)e^{-d(u)\tau},
$$

$$
C(\tau,u)=iu(r-q)\tau+
\frac{\kappa\theta}{\xi^2}
\left[
(\kappa-\rho\xi iu-d(u))\tau
-2\ln\left(\frac{1-g(u)e^{-d(u)\tau}}{1-g(u)}\right)
\right].
$$

Then:

$$
\varphi(u)=\exp\left(C(\tau,u)+D(\tau,u)v_0+iu\ln S_0\right).
$$

These expressions are the mathematical heart of the semi-closed Heston formula.

### Complex square roots and logarithms are not a cosmetic detail

Complex functions have branches. A naive implementation of the formulas can show discontinuities as parameters or Fourier arguments vary—the “little Heston trap” and related stability issues. Robust production implementations use a stable formulation, ensure a continuous choice of the square-root/log branches, or use alternative characteristic-function formulations such as those associated with Lord–Kahl or Cui et al. Always validate prices across a grid of strikes and maturities rather than trusting one output.

## 23. Fourier pricing: practical standard formula

A widely used one-characteristic-function formulation is:

$$
C_0=S_0e^{-qT}\Pi_1-Ke^{-rT}\Pi_2,
$$

where:

$$
\Pi_2
=
\frac12+
\frac1\pi\int_0^\infty
\operatorname{Re}\left[
\frac{e^{-iu\ln K}\varphi(u)}{iu}
\right]du,
$$

and:

$$
\Pi_1
=
\frac12+
\frac1\pi\int_0^\infty
\operatorname{Re}\left[
\frac{e^{-iu\ln K}\varphi(u-i)}{iu\,\varphi(-i)}
\right]du.
$$

This is a compact way to account for the two measures. Equivalent formulas use two separately parameterised characteristic functions, as in the chapter.

The integrals are real-valued after taking the real part, but their integrands are formed from complex arithmetic.

### Numerical quadrature

A simple prototype can use adaptive quadrature or a carefully chosen finite upper limit and Simpson/trapezoidal integration. Production methods may use Gauss–Laguerre quadrature, COS methods, Carr–Madan FFT, or other Fourier schemes.

Basic numerical checks:

- the call is non-negative;
- the call is no larger than $$S_0e^{-qT}$$;
- put–call parity holds;
- price converges when the integration grid is refined;
- price converges as the upper cutoff increases;
- Heston tends to Black–Scholes in the deterministic-variance limit described below.

## 24. Dividend yield and put–call parity

With continuous yield $$q$$, replace the risk-neutral asset drift by $$r-q$$.

Put–call parity is:

$$
C_0-P_0=S_0e^{-qT}-Ke^{-rT}.
$$

Equivalently:

$$
P_0=C_0+Ke^{-rT}-S_0e^{-qT}.
$$

This identity is an excellent implementation test and lets you compute a put from a call without a separate Fourier inversion.

## 25. Black–Scholes as a Heston limit

Set vol-of-vol to zero:

$$
\xi=0.
$$

Then variance is no longer random:

$$
dv_t=\kappa(\theta-v_t)dt.
$$

Its solution is deterministic:

$$
v_t=\theta+(v_0-\theta)e^{-\kappa t}.
$$

If additionally:

$$
v_0=\theta,
$$

then:

$$
v_t=\theta
$$

for every time. The asset then has constant volatility:

$$
\sigma_{\mathrm{BS}}=\sqrt{\theta},
$$

and Heston reduces to Black–Scholes.

More generally, deterministic but time-varying variance gives a Black–Scholes-type lognormal model with integrated variance:

$$
\int_0^T v_sds.
$$

This limit is a vital unit test for both pricing code and PINN results.

---

# Relating every chapter section to the core logic

## 26. Model dynamics

The chapter begins with physical-measure stock and variance equations. It notes that if one defines volatility as $$h_t=\sqrt{v_t}$$, applying Itô gives a more awkward process for $$h_t$$. That is why Heston models **variance** directly: the CIR form is analytically tractable and maintains non-negativity more naturally.

It then changes to risk-neutral dynamics. The most important reading rule is: for option pricing, use risk-neutral parameters. The physical drift $$\mu$$ is not an input to the usual no-arbitrage European option price.

## 27. Properties of the variance process

The chapter states that the conditional distribution of future variance is a scaled non-central chi-squared distribution. This is a stronger result than you need to use the characteristic-function pricer, but it confirms that CIR is a well-studied positive process with known transition law. The conditional mean and variance describe how $$\kappa$$ controls persistence and $$\xi$$ controls uncertainty.

## 28. The European call price and two probabilities

The payoff decomposition is:

$$
(S_T-K)^+=S_T\mathbf1_{S_T>K}-K\mathbf1_{S_T>K}.
$$

Discounted expectation yields the two terms of the call price. The chapter's measure change converts the first term into the current stock price times a probability under an alternative measure. This is why its formula looks like Black–Scholes even though the probabilities are obtained differently.

## 29. The Heston PDE and hedging portfolio

The chapter includes an additional derivative $$U$$ in the hedge because variance risk supplies a second source of randomness. It chooses stock and derivative holdings to cancel coefficients of both $$dS$$ and $$dv$$. The remaining portfolio must earn the risk-free rate. Equating the two drift expressions produces a ratio that must be independent of which derivative was selected; this ratio is written in terms of the variance risk premium. The affine assumption for that premium produces the standard Heston risk-neutral variance drift.

## 30. The PDE for $$P_1$$ and $$P_2$$

The chapter substitutes the Black–Scholes-like call decomposition into the Heston PDE, then matches the separate parts. This yields PDEs for the two probability-like functions. This step can look algebraically unmotivated, but its purpose is precise: it turns pricing into the calculation of transformable quantities whose terminal conditions are indicators of an in-the-money event.

## 31. Characteristic functions

At expiry, the indicator is:

$$
\mathbf1_{x_T>\ln K}.
$$

Fourier inversion transforms this discontinuous payoff/event into an integral involving a smooth characteristic function. The exponential-affine proposal for the characteristic function is selected because the Heston generator is affine in $$v$$.

## 32. Riccati solution

Substitution produces one nonlinear ODE for $$D$$ and one linear ODE for $$C$$. The nonlinear one is Riccati. Its closed-form solution, inserted into the linear equation and integrated, yields the formulas used by the pricing routine.

## 33. Code and numerical integration

The code examples evaluate a complex integrand over a grid of Fourier arguments. The use of a finite interval and trapezoidal rule is pedagogically transparent, but not automatically robust for all parameter sets, especially at short maturity or under extreme parameter values. Numerical convergence must be checked.

## 34. Dividends, puts, and consolidated integrals

Continuous dividends replace $$r$$ by $$r-q$$ in the asset drift and introduce the factor $$e^{-qT}$$. Put–call parity gives puts. The chapter consolidates two integrations into one for efficiency. This changes computational packaging, not the economic model.

## 35. Black–Scholes special case

The chapter verifies that removing variance randomness and fixing variance at its long-run level retrieves the familiar lognormal model, normal CDF, Black–Scholes PDE, and Black–Scholes formula. This is both a conceptual consistency result and the strongest debugging test.

---

# Heston for PINNs and ML: what you need to know

## 36. A Heston PINN problem statement

Choose a domain, for example:

$$
(S,v,\tau)\in[S_{\min},S_{\max}]\times[v_{\min},v_{\max}]\times[0,T_{\max}].
$$

Let a neural network approximate:

$$
V_\vartheta(S,v,\tau).
$$

Using the backward-time PDE, define a residual:

$$
\mathcal R_\vartheta
=
V_\tau
-
\frac12vS^2V_{SS}
-
\rho\xi vSV_{Sv}
-
\frac12\xi^2vV_{vv}
-
(r-q)SV_S
-
\kappa(\theta-v)V_v
+rV.
$$

The sign convention is correct for the backward-time version. It equals zero in the interior.

A typical loss is:

$$
\mathcal L
=
\lambda_{\mathrm{PDE}}\operatorname{MSE}(\mathcal R_\vartheta)
+
\lambda_{\mathrm{term}}\operatorname{MSE}\left(V_\vartheta(S,v,0)-(S-K)^+\right)
+
\lambda_{\mathrm{bc}}\mathcal L_{\mathrm{boundary}}.
$$

Automatic differentiation supplies the first and second derivatives.

### More stable coordinates

Use:

$$
x=\ln(S/K),
$$

and normalise output, for example:

$$
u(x,v,\tau)=\frac{V(S,v,\tau)}{K}.
$$

These transformations can improve conditioning and make a network more reusable across strikes. The log-price PDE is usually the cleaner starting point.

## 37. Boundary conditions deserve care

The terminal condition is exact:

$$
V(S,v,0)=(S-K)^+.
$$

Useful asymptotic economic conditions for a call include:

$$
V(0,v,\tau)=0,
$$

$$
V(S,v,\tau)\sim Se^{-q\tau}-Ke^{-r\tau}
\quad\text{as }S\to\infty.
$$

The boundary at $$v=0$$ is delicate because the coefficient of $$V_{vv}$$ vanishes. It is a degenerate PDE boundary. Whether an explicit boundary condition is required depends on the Feller regime and the PDE formulation. Do not casually impose an arbitrary Dirichlet value there merely to make a neural network train.

## 38. Common PINN difficulties specific to Heston

1. **Degeneracy near $$v=0$$**: diffusion vanishes in the variance direction.
2. **Non-smooth terminal payoff**: the kink at $$S=K$$ makes derivatives discontinuous at expiry.
3. **Multi-scale inputs**: stock level, variance, and time can differ radically in scale.
4. **Mixed derivative**: $$V_{Sv}$$ can be difficult for network optimisation.
5. **Loss imbalance**: a small PDE residual does not guarantee accurate prices around the strike.
6. **Parameter dependence**: a network trained for one parameter vector does not automatically price another.
7. **No-arbitrage shape constraints**: raw neural outputs may violate monotonicity or convexity in strike.

Useful mitigations include adaptive collocation, payoff smoothing only for controlled experiments, curriculum learning in maturity, input/output normalisation, hard enforcement of terminal conditions where possible, residual weighting, and comparisons against a trusted Fourier pricer.

## 39. Comparing Fourier Heston, PDE solvers, Monte Carlo, and ML

| Method | Strong use case | Main limitation |
|---|---|---|
| Characteristic-function quadrature | European vanilla calls/puts | Requires stable complex arithmetic and integration |
| Finite differences | PDE values and some early-exercise/path features | Grid cost grows with dimension; mixed derivative and boundaries need care |
| Monte Carlo | Path-dependent/high-dimensional extensions | Slow for vanilla calibration; variance simulation needs care |
| PINN | Mesh-free PDE approximation, parametric surrogates, inverse problems | Optimisation can be unstable; accuracy is not guaranteed by low training loss |
| Supervised ML surrogate | Fast repeated pricing after training | Needs quality labelled data and can extrapolate poorly |

For your proposed comparison, the semi-analytic Heston Fourier price is the natural benchmark for European vanilla options. It can generate high-quality labels and independently validate PINN solutions.

## 40. Calibration is separate from pricing

Given market option prices or implied volatilities, calibration seeks:

$$
\Theta=(\kappa,\theta,\xi,\rho,v_0)
$$

that minimises an error such as:

$$
\min_{\Theta}
\sum_jw_j\left(C_{\mathrm{Heston}}(K_j,T_j;\Theta)-C_{\mathrm{mkt},j}\right)^2.
$$

Often implied-volatility errors are used instead of price errors. Parameters are constrained by:

$$
\kappa>0,
\quad
\theta>0,
\quad
\xi>0,
\quad
-1<\rho<1,
\quad
v_0>0.
$$

Calibration is frequently ill-conditioned: different parameter sets can yield similar option surfaces, particularly with sparse data. Short maturities tend to identify $$v_0$$ and aspects of $$\xi,\rho$$ more strongly; longer maturities contain more information about $$\kappa,\theta$$. This is a practical identifiability statement, not an absolute rule.

A PINN can be used either:

- as a **forward solver**, with parameters fixed; or
- in an **inverse PINN**, where parameters are trainable alongside network weights and market prices are added to the loss.

The second is much harder and must be tested for identifiability rather than judged only by in-sample fit.

---

# A recommended learning sequence

## Stage 1: get fluent with the economics

Be able to explain without equations:

- call, put, payoff, strike, maturity, discounting;
- risk-neutral pricing versus forecasting;
- why Black–Scholes uses fixed volatility;
- why stochastic variance helps produce skew/smile;
- the qualitative jobs of $$v_0,\theta,\kappa,\xi,\rho$$.

## Stage 2: get fluent with the PDE

Work through:

1. multivariate Itô's lemma;
2. the three quadratic variations $$dS^2$$, $$dv^2$$, and $$dSdv$$;
3. the generator;
4. the Heston PDE and terminal payoff;
5. change from $$S$$ to $$x=\ln S$$;
6. Black–Scholes as the $$\xi=0$$, $$v_0=\theta$$ limit.

## Stage 3: get fluent with transforms

Work through:

1. characteristic function of a normal variable;
2. Fourier inversion at a conceptual level;
3. exponential-affine ansatz;
4. coefficient matching in the PDE;
5. Riccati ODE and its closed form;
6. numerical quadrature and complex-number stability.

## Stage 4: implement in increasing difficulty

1. Implement Black–Scholes and put–call parity.
2. Implement Heston characteristic function.
3. Implement a call price with adaptive quadrature.
4. Verify the Black–Scholes limit.
5. Compare with Monte Carlo under a robust variance scheme.
6. Implement a finite-difference or PINN Heston PDE solver.
7. Compare prices, Greeks, convergence, and runtime across methods.
8. Attempt calibration only after the forward pricer is thoroughly validated.

---

# Final checklist: “I understand Heston fully enough to work with it”

You are ready to compare Heston with ML and Heston-informed PINNs when you can answer yes to these:

- [ ] I can distinguish volatility from variance.
- [ ] I know why the variance process is CIR and what each parameter means.
- [ ] I know what negative $$\rho$$ does economically and geometrically.
- [ ] I can distinguish physical and risk-neutral parameters.
- [ ] I can write the risk-neutral Heston SDEs.
- [ ] I can derive the mixed derivative in the pricing PDE.
- [ ] I know the terminal and economically meaningful boundary conditions for a European call.
- [ ] I understand why Heston is affine and why that leads to an exponential-affine characteristic function.
- [ ] I know what a Riccati ODE is and why it appears here.
- [ ] I know that the formula is semi-closed because Fourier integrals remain numerical.
- [ ] I can explain the Black–Scholes limit and use it as a test.
- [ ] I distinguish pricing, calibration, simulation, and forecasting.
- [ ] I know that a PINN’s PDE residual is not by itself evidence of a financially accurate or well-calibrated model.

## One final compact summary

Heston is:

$$
\text{Black–Scholes price dynamics}
+
\text{random mean-reverting variance}
+
\text{correlation between return and variance shocks}.
$$

Its major practical advantage is:

$$
\text{richer implied-volatility behaviour}
+
\text{tractable characteristic function}.
$$

Its major practical caveat is:

$$
\text{calibration and numerical stability matter as much as the formula itself}.
$$
