# Methods and empirical analysis plan

## Main constructs

### Corporate diversification

Preferred construction:

```text
DIV = 1 - HHI
```

This is important because HHI is a concentration index. If HHI increases, the firm is more focused. The transformed variable `DIV` increases with diversification.

If segment sales are available, compute:

```text
HHI = sum(segment sales share^2)
DIV = 1 - HHI
```

An entropy index can be used as a robustness measure.

### Information asymmetry

Preferred proxy:

```text
Amihud illiquidity = mean(abs(daily return) / daily value traded)
```

Compute this by firm-year from daily trading data. Larger Amihud values indicate lower liquidity and greater information asymmetry.

### Earnings quality

The Modified Jones model estimates discretionary accruals. Higher absolute discretionary accruals indicate lower earnings quality. For clarity:

```text
EQ = -abs(discretionary accruals)
```

This makes larger `EQ` mean better earnings quality.

If the existing variable is `EAR` and it measures earnings management, use:

```text
EQ = -abs(EAR)
```

### Corporate governance quality

Construct a governance index from:

- Board independence
- Board size
- Audit committee independence
- CEO duality
- Institutional ownership

Standardize each component. Reverse CEO duality so larger values indicate stronger governance. Average the standardized components.

## Econometric models

### H1: Direct effect

```text
EQ_it = b0 + b1 DIV_it + controls_it + firm FE + year FE + error_it
```

### H2: First-stage mediation path

```text
IA_it = a0 + a1 DIV_it + controls_it + firm FE + year FE + error_it
```

### H3--H5: Outcome model with moderation

```text
EQ_it = c0 + c1 DIV_it + c2 IA_it + c3 CGQ_it + c4 IA_it*CGQ_it + controls_it + firm FE + year FE + error_it
```

### H6: Conditional indirect effect

```text
Indirect effect = a1 * (c2 + c4*CGQ)
```

Use bootstrap confidence intervals at low, mean, and high governance levels.

## Recommended robustness checks

1. Use HHI instead of `DIV` and reverse interpretation.
2. Use entropy index if segment sales are available.
3. Use alternative information asymmetry proxies if available: bid-ask spread, turnover, zero-return days.
4. Use lagged independent variables: `DIV_{t-1}`, `IA_{t-1}`, and `CGQ_{t-1}`.
5. Winsorize continuous variables at 1% and 99%.
6. Cluster standard errors by firm.
7. Include firm and year fixed effects.
8. Exclude financial firms or highly regulated sectors as sensitivity analysis.
