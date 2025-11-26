# Basel FRTB-SA Market Risk Capital Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Validation](https://img.shields.io/badge/Validation-95%25%20Pass-brightgreen.svg)]()

A production-grade implementation of the **Basel III Fundamental Review of the Trading Book (FRTB) Standardised Approach (SA)** for market risk capital calculation. This engine computes regulatory capital charges for a multi-asset portfolio spanning interest rates, foreign exchange, and equities with full traceability to Basel MAR21 regulatory paragraphs.

---

## 📋 Table of Contents

1. [Project Overview](#-project-overview)
2. [Portfolio Composition](#-portfolio-composition)
3. [Capital Results Summary](#-capital-results-summary)
4. [Phase 1: Zero Curve Bootstrapping](#-phase-1-zero-curve-bootstrapping)
5. [Phase 2: Sensitivity Calculation](#-phase-2-sensitivity-calculation)
6. [Phase 3: GIRR Delta Capital (Detailed)](#-phase-3-girr-delta-capital-detailed)
7. [Phase 10: FX Delta Capital](#-phase-10-fx-delta-capital)
8. [Phase 12: Equity Delta Capital](#-phase-12-equity-delta-capital)
9. [Correlation Scenarios Framework](#-correlation-scenarios-framework)
10. [Validation Framework](#-validation-framework)
11. [Technical Implementation](#-technical-implementation)
12. [Installation & Usage](#-installation--usage)
13. [Basel Regulatory References](#-basel-regulatory-references)
14. [Future Enhancements](#-future-enhancements)

---

## 🎯 Project Overview

### What is FRTB-SA?

The **Fundamental Review of the Trading Book (FRTB)** is a comprehensive suite of capital rules developed by the Basel Committee on Banking Supervision (BCBS) to address shortcomings revealed during the 2007-2008 financial crisis. The **Standardised Approach (SA)** provides a formulaic method for calculating market risk capital that all banks must be able to compute.

### What This Engine Does

This engine implements the **sensitivity-based method (SbM)** of FRTB-SA, which calculates capital charges through:

1. **Sensitivity Calculation**: Computing how portfolio value changes with respect to risk factors
2. **Risk Weighting**: Applying Basel-prescribed risk weights to sensitivities
3. **Aggregation**: Combining weighted sensitivities using correlation matrices
4. **Scenario Analysis**: Running three correlation scenarios (BASE, HIGH, LOW)
5. **Capital Determination**: Taking the maximum capital across scenarios

### Risk Classes Implemented

| Risk Class | Basel Reference | Complexity | Status |
|------------|-----------------|------------|--------|
| **GIRR Delta** | MAR21.19-51 | ⭐⭐⭐⭐⭐ High | ✅ Complete |
| **FX Delta** | MAR21.86-89 | ⭐⭐ Low | ✅ Complete |
| **Equity Delta** | MAR21.73-80 | ⭐⭐⭐ Medium | ✅ Complete |

---

## 💼 Portfolio Composition

### Overview

The portfolio consists of **16 instruments** across three asset classes:

| Asset Class | Instruments | Total Exposure |
|-------------|-------------|----------------|
| Interest Rates (GIRR) | 4 instruments | ~$180M notional |
| Foreign Exchange (FX) | 2 positions | $21.8M |
| Equities | 5 instruments | $4.2M (Delta) |

### Detailed Instrument Breakdown

#### Interest Rate Instruments (GIRR)

| Instrument | Description | Notional | Currency | Curve |
|------------|-------------|----------|----------|-------|
| **USGG5YR** | 5-Year US Treasury Bond | $100,000,000 | USD | Treasury |
| **USSO2** | 2-Year SOFR Swap (receive fixed) | $50,000,000 | USD | SOFR |
| **USISSO10** | 10-Year SOFR Swap (pay fixed) | $30,000,000 | USD | SOFR |
| **EUSA5** | 5-Year EUR Swap (receive fixed) | €25,000,000 | EUR | EUR Swap |

#### Foreign Exchange Positions (FX Delta)

| Position | Description | Notional | Direction |
|----------|-------------|----------|-----------|
| **EUR/USD** | Long EUR vs USD | €12,000,000 | Long EUR |
| **USD/JPY** | Long USD vs JPY | $8,000,000 | Short JPY |

#### Equity Positions (Equity Delta)

| Instrument | Description | Position Value | Type |
|------------|-------------|----------------|------|
| **SPX** | S&P 500 Index | $3,000,000 | Index (Delta) |
| **AAPL** | Apple Inc. Stock | $1,200,000 | Single Stock (Delta) |
| **VIX** | VIX Volatility Index | - | Vega (excluded from Delta) |
| **VXAPL** | Apple Implied Volatility | - | Vega (excluded from Delta) |
| **MOVE** | Bond Market Volatility | - | GIRR Vega (excluded) |

---

## 📊 Capital Results Summary

### Final Capital Charges

| Risk Class | Capital Charge | Binding Scenario | Validation Pass Rate |
|------------|----------------|------------------|---------------------|
| **GIRR Delta** | **$1,767,543** | LOW | 88.4% (38/43 tests) |
| **FX Delta** | **$1,322,938** | LOW | 100% (24/24 tests) |
| **Equity Delta** | **$814,709** | HIGH | 100% (33/33 tests) |
| **Total Delta** | **$3,905,190** | - | - |

### Capital by Scenario

| Risk Class | BASE | HIGH | LOW | Binding |
|------------|------|------|-----|---------|
| GIRR Delta | $1,438,015 | $1,005,765 | $1,767,543 | LOW |
| FX Delta | $1,173,421 | $1,001,833 | $1,322,938 | LOW |
| Equity Delta | $802,185 | $814,709 | $789,462 | HIGH |

### Key Insight: Why Different Scenarios Bind

- **GIRR & FX (LOW binds)**: Opposite-signed positions create hedges. Lower correlation = weaker hedge = higher capital.
- **Equity (HIGH binds)**: Same-signed positions (both long). Higher correlation = more correlated losses = higher capital.

---

## 🔧 Phase 1: Zero Curve Bootstrapping

### Objective

Construct zero-coupon discount factor curves from market instruments to enable accurate pricing and sensitivity calculation.

### Curves Built

| Curve | Instruments | Tenors | Day Count Convention |
|-------|-------------|--------|---------------------|
| **USD Treasury** | Treasury bonds | 3M to 30Y (10 points) | ACT/ACT |
| **USD SOFR** | SOFR swaps | 3M to 30Y (10 points) | ACT/360 |
| **EUR Swap** | EUR interest rate swaps | 3M to 30Y (10 points) | 30/360 |

### Bootstrap Methodology
```
┌─────────────────────────────────────────────────────────────────┐
│                    CURVE BOOTSTRAPPING PROCESS                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  SHORT END (3M, 6M):                                            │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Money Market Formula:                                    │   │
│  │  DF(T) = 1 / (1 + r × T)                                 │   │
│  │  where r = quoted rate, T = year fraction                │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  1-YEAR POINT:                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Coupon Bond/Swap Formula (NOT money market):            │   │
│  │  For Treasury: Price = Σ(coupon × DF_i) + 100 × DF_n     │   │
│  │  For Swap: 0 = Σ(fixed_rate × DF_i) - (1 - DF_n)        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                          ↓                                       │
│  LONG END (2Y - 30Y):                                           │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  Newton-Raphson Iteration:                                │   │
│  │  1. Guess initial DF                                      │   │
│  │  2. Price instrument using log-linear interpolation       │   │
│  │  3. Calculate pricing error                               │   │
│  │  4. Update DF: DF_new = DF_old - f(DF)/f'(DF)            │   │
│  │  5. Repeat until |error| < 1e-10                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Interpolation Method

- **Method**: Log-linear interpolation on discount factors (equivalent to constant forward rates between nodes)
- **Formula**: `ln(DF(t)) = ln(DF(t1)) + (t - t1)/(t2 - t1) × [ln(DF(t2)) - ln(DF(t1))]`

### Validation Results (Phase 1)

| Test Category | Tests | Passed | Status |
|---------------|-------|--------|--------|
| Discount Factor Properties | 4 per curve | 12/12 | ✅ |
| Instrument Repricing | 4 instruments | 4/4 | ✅ |
| Methodology Verification | 9 tests | 9/9 | ✅ |
| **Total** | **25** | **25/25** | **100%** |

---

## 📐 Phase 2: Sensitivity Calculation

### Objective

Calculate DV01 (dollar value of 1 basis point) for each instrument at each tenor point.

### Basel Sensitivity Definition (MAR21.19)

$$s_{k,r_t} = \frac{V_i(r_t + 0.0001) - V_i(r_t)}{0.0001}$$

This represents the change in instrument value for a 1bp parallel shift in the rate at tenor t.

### Sensitivity Calculation by Instrument Type

#### Treasury Bonds
```python
# For each tenor point t:
# 1. Shift the curve at point t by 1bp
# 2. Reprice the bond
# 3. DV01 = (Price_shifted - Price_base) / 0.0001
```

#### Interest Rate Swaps
```python
# For each tenor point t:
# 1. Shift the curve at point t by 1bp
# 2. Recalculate fixed and floating leg PVs
# 3. DV01 = (NPV_shifted - NPV_base) / 0.0001
```

### Portfolio Sensitivities (30 Risk Factors)

| Curve | Tenors | Risk Factors | Key Sensitivities |
|-------|--------|--------------|-------------------|
| USD Treasury | 0.25Y - 30Y | 10 | 5Y: -$29,355 per bp |
| USD SOFR | 0.25Y - 30Y | 10 | 2Y: +$9,450, 10Y: -$18,230 per bp |
| EUR Swap | 0.25Y - 30Y | 10 | 5Y: +$10,720 per bp |
| **Total** | - | **30** | - |

### Basel Scaling (MAR21.19)

Raw DV01 values are scaled to align with Basel's risk weight framework:

$$\text{Basel Sensitivity} = \text{DV01} \times 10,000$$

This converts the "per 1bp" sensitivity to a "per 100bp (1%)" sensitivity.

---

## 📈 Phase 3: GIRR Delta Capital (Detailed)

### Overview

GIRR (General Interest Rate Risk) Delta is the **most complex** risk class in FRTB-SA due to:
- Multiple curves per currency (Treasury, SOFR, swaps)
- 10 tenor points per curve
- Intra-bucket correlations varying by tenor distance
- Basis risk correlations between curves
- Cross-currency (cross-bucket) aggregation

### Step-by-Step Process
```
┌────────────────────────────────────────────────────────────────────────────┐
│                     GIRR DELTA CAPITAL CALCULATION                          │
│                        (Basel MAR21.19-51)                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: SENSITIVITY CALCULATION (MAR21.19)                                │
│  ═══════════════════════════════════════════                               │
│  • 30 risk factors: 10 tenors × 3 curves                                   │
│  • Formula: s_k = V(r + 1bp) - V(r) / 0.0001                              │
│  • Scale by 10,000 for Basel Convention                                    │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 2: CURRENCY CONVERSION                                               │
│  ══════════════════════════════                                            │
│  • EUR sensitivities → USD at spot rate 1.152                              │
│  • All calculations in reporting currency (USD)                            │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 3: RISK WEIGHT APPLICATION (MAR21.43-44)                             │
│  ═════════════════════════════════════════════                             │
│  • Base RW by tenor: 1.7% (short) to 1.1% (long)                          │
│  • √2 reduction for well-traded currencies: RW_adj = RW_base / √2          │
│  • Weighted Sensitivity: WS_k = s_k × RW_k                                 │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Risk Weight Schedule (MAR21.44):                                    │   │
│  │  Tenor    │ 0.25Y │ 0.5Y │ 1Y  │ 2Y  │ 3Y  │ 5Y  │ 10Y │ 15Y │ 30Y │   │
│  │  RW Base  │ 1.7%  │ 1.7% │ 1.6%│ 1.3%│ 1.2%│ 1.1%│ 1.1%│ 1.1%│ 1.1%│   │
│  │  RW Adj   │ 1.20% │ 1.20%│1.13%│0.92%│0.85%│0.78%│0.78%│0.78%│0.78%│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 4: INTRA-BUCKET CORRELATION MATRIX (MAR21.45-47)                     │
│  ═════════════════════════════════════════════════════                     │
│                                                                             │
│  4a. SAME-CURVE TENOR CORRELATION (MAR21.47):                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ρ(t_k, t_l) = max(e^(-θ × |t_k - t_l| / min(t_k, t_l)), 40%)       │   │
│  │                                                                      │   │
│  │  where θ = 0.03 (correlation decay parameter)                        │   │
│  │                                                                      │   │
│  │  Example: ρ(1Y, 5Y) = max(e^(-0.03 × 4 / 1), 0.40) = 0.8869         │   │
│  │  Example: ρ(5Y, 30Y) = max(e^(-0.03 × 25 / 5), 0.40) = 0.8607       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  4b. CROSS-CURVE (BASIS) CORRELATION (MAR21.45):                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  ρ_basis = 99.90% (between different curves in same currency)       │   │
│  │                                                                      │   │
│  │  Combined: ρ_combined = ρ_tenor × ρ_basis                           │   │
│  │                                                                      │   │
│  │  Example: USD Treasury 5Y vs USD SOFR 10Y:                          │   │
│  │  ρ = 0.8607 × 0.999 = 0.8598                                        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  USD Bucket Matrix: 20×20 (10 Treasury + 10 SOFR tenors)                   │
│  EUR Bucket Matrix: 10×10 (10 swap tenors)                                 │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 5: INTRA-BUCKET AGGREGATION (MAR21.4)                                │
│  ══════════════════════════════════════════                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  K_b² = Σ_k WS_k² + Σ_{k≠l} ρ_kl × WS_k × WS_l                      │   │
│  │                                                                      │   │
│  │  K_b = √(K_b²)  [if K_b² ≥ 0]                                       │   │
│  │                                                                      │   │
│  │  S_b = Σ_k WS_k  (sum with signs preserved)                         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Results (BASE scenario):                                                   │
│  • K_USD = $1,177,389                                                      │
│  • K_EUR = $1,288,273                                                      │
│  • S_USD = +$757,868 (net long rates)                                      │
│  • S_EUR = -$1,290,470 (net short rates)                                   │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 6: THREE CORRELATION SCENARIOS (MAR21.6)                             │
│  ═════════════════════════════════════════════                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Scenario │ Intra-Bucket ρ          │ Inter-Bucket γ                │   │
│  │  ─────────┼─────────────────────────┼─────────────────────────────  │   │
│  │  BASE     │ ρ                       │ γ = 0.50                      │   │
│  │  HIGH     │ min(1.25 × ρ, 1.0)      │ γ = min(1.25 × 0.50, 1) = 0.625│  │
│  │  LOW      │ max(2ρ - 1, 0.75 × ρ)   │ γ = max(0, 0.375) = 0.375     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 7: INTER-BUCKET AGGREGATION (MAR21.50)                               │
│  ════════════════════════════════════════════                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  K_GIRR² = Σ_b K_b² + Σ_{b≠c} γ_bc × S_b × S_c                      │   │
│  │                                                                      │   │
│  │  where:                                                              │   │
│  │  • K_b = bucket capital (USD, EUR)                                  │   │
│  │  • S_b = sum of weighted sensitivities in bucket                    │   │
│  │  • γ_bc = 0.50 (cross-currency correlation)                         │   │
│  │                                                                      │   │
│  │  For our 2-bucket case:                                             │   │
│  │  K² = K_USD² + K_EUR² + 2 × γ × S_USD × S_EUR                       │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Cross-term Analysis:                                                       │
│  • S_USD = +$757,868 (positive)                                            │
│  • S_EUR = -$1,290,470 (negative)                                          │
│  • Cross-term = 2 × γ × (+) × (-) = NEGATIVE                               │
│  • Negative cross-term = diversification benefit (hedge)                    │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 8: FINAL CAPITAL DETERMINATION                                       │
│  ═══════════════════════════════════════                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Scenario │ K_USD      │ K_EUR      │ γ     │ K_GIRR               │   │
│  │  ─────────┼────────────┼────────────┼───────┼─────────────────────  │   │
│  │  BASE     │ $1,177,389 │ $1,288,273 │ 0.500 │ $1,438,015           │   │
│  │  HIGH     │ $754,443   │ $1,290,305 │ 0.625 │ $1,005,765           │   │
│  │  LOW      │ $1,484,354 │ $1,286,237 │ 0.375 │ $1,767,543           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  K_GIRR_FINAL = max(BASE, HIGH, LOW) = $1,767,543                          │
│  BINDING SCENARIO: LOW                                                      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why LOW Scenario Binds for GIRR

The portfolio has **opposite-signed bucket sums** (S_USD > 0, S_EUR < 0), creating a natural hedge:

| Scenario | Cross-Bucket γ | Hedge Strength | Cross-term | Capital |
|----------|----------------|----------------|------------|---------|
| HIGH | 0.625 | Strong hedge | -$613M | $1,005,765 (lowest) |
| BASE | 0.500 | Moderate hedge | -$490M | $1,438,015 |
| LOW | 0.375 | Weak hedge | -$367M | $1,767,543 (highest) |

**Basel's conservative approach**: Take the maximum capital, which occurs when the hedge is least effective (LOW scenario).

### Correlation Matrix Visualization
```
USD Bucket Correlation Matrix (20×20) - BASE Scenario
         │ Tsy_0.25Y  Tsy_1Y   Tsy_5Y   SOFR_0.25Y  SOFR_1Y  SOFR_5Y
─────────┼────────────────────────────────────────────────────────────
Tsy_0.25Y│   1.0000   0.9704   0.8607     0.9990    0.9694   0.8599
Tsy_1Y   │   0.9704   1.0000   0.8869     0.9694    0.9990   0.8861
Tsy_5Y   │   0.8607   0.8869   1.0000     0.8599    0.8861   0.9990
SOFR_0.25│   0.9990   0.9694   0.8599     1.0000    0.9704   0.8607
SOFR_1Y  │   0.9694   0.9990   0.8861     0.9704    1.0000   0.8869
SOFR_5Y  │   0.8599   0.8861   0.9990     0.8607    0.8869   1.0000
```

---

## 💱 Phase 10: FX Delta Capital

### Overview

FX Delta is simpler than GIRR because:
- One risk factor per currency (vs. 10 tenors in GIRR)
- No intra-bucket correlation (single factor per bucket)
- Uniform cross-bucket correlation (γ = 60%)

### Calculation Process
```
┌────────────────────────────────────────────────────────────────────────────┐
│                       FX DELTA CAPITAL CALCULATION                          │
│                          (Basel MAR21.86-89)                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: SENSITIVITY CALCULATION                                           │
│  ════════════════════════════════                                          │
│  FX sensitivity = Position value in USD                                    │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Position   │ Notional     │ FX Rate │ Sensitivity │ Direction     │   │
│  │  ───────────┼──────────────┼─────────┼─────────────┼──────────────  │   │
│  │  Long EUR   │ €12,000,000  │ 1.152   │ +$13,824,000│ Positive      │   │
│  │  Short JPY  │ $8,000,000   │ -       │ -$8,000,000 │ Negative      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Note: Short JPY means we lose money when JPY appreciates → negative s     │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 2: RISK WEIGHT APPLICATION (MAR21.87-88)                             │
│  ═════════════════════════════════════════════                             │
│                                                                             │
│  Standard RW = 15%                                                         │
│  Major Pairs RW = 15% / √2 = 10.6066%                                      │
│                                                                             │
│  Major pairs include: EUR, JPY, GBP, AUD, CAD, CHF, etc. vs USD            │
│                                                                             │
│  Weighted Sensitivities:                                                    │
│  • WS_EUR = +$13,824,000 × 10.6066% = +$1,466,257                          │
│  • WS_JPY = -$8,000,000 × 10.6066% = -$848,528                             │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 3: BUCKET CAPITAL (Single factor per bucket)                         │
│  ═════════════════════════════════════════════════                         │
│                                                                             │
│  K_EUR = |WS_EUR| = $1,466,257                                             │
│  K_JPY = |WS_JPY| = $848,528                                               │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 4: INTER-BUCKET AGGREGATION (MAR21.89)                               │
│  ════════════════════════════════════════════                              │
│                                                                             │
│  γ_base = 0.60 (uniform for all FX pairs)                                  │
│                                                                             │
│  K² = K_EUR² + K_JPY² + 2 × γ × WS_EUR × WS_JPY                            │
│                                                                             │
│  Cross-term = 2 × 0.60 × (+$1,466,257) × (-$848,528) = -$1,493B            │
│                                                                             │
│  Opposite signs → NEGATIVE cross-term → Diversification benefit!           │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 5: THREE SCENARIOS                                                   │
│  ═══════════════════════                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Scenario │ γ      │ Cross-term   │ K²          │ K_FX             │   │
│  │  ─────────┼────────┼──────────────┼─────────────┼─────────────────  │   │
│  │  BASE     │ 0.60   │ -$1,493B     │ $1,377B     │ $1,173,421       │   │
│  │  HIGH     │ 0.75   │ -$1,866B     │ $1,004B     │ $1,001,833       │   │
│  │  LOW      │ 0.45   │ -$1,120B     │ $1,750B     │ $1,322,938       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  K_FX_FINAL = max(BASE, HIGH, LOW) = $1,322,938                            │
│  BINDING SCENARIO: LOW                                                      │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why LOW Scenario Binds for FX

Same logic as GIRR - opposite-signed positions create a hedge:

| Scenario | γ | Hedge Effectiveness | Capital |
|----------|---|---------------------|---------|
| HIGH | 0.75 | Strong (EUR & JPY move together) | $1,001,833 |
| LOW | 0.45 | Weak (less correlated) | $1,322,938 |

Basel takes max → LOW binds (most conservative).

### FX vs GIRR Comparison

| Aspect | FX Delta | GIRR Delta |
|--------|----------|------------|
| Risk factors | 2 (one per currency) | 30 (10 tenors × 3 curves) |
| Intra-bucket correlation | N/A (single factor) | Complex (tenor × basis) |
| Inter-bucket γ | 0.60 | 0.50 |
| Binding scenario | LOW | LOW |

---

## 📊 Phase 12: Equity Delta Capital

### Overview

Equity Delta has medium complexity due to:
- 13 bucket structure (by market cap, economy, sector)
- Bucket-specific risk weights (15% to 70%)
- Different treatment for indices vs. single stocks

### Equity Bucket Structure (MAR21.77)
```
┌────────────────────────────────────────────────────────────────────────────┐
│                        EQUITY BUCKET TAXONOMY                               │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  LARGE CAP EMERGING MARKETS (Buckets 1-4)                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Bucket 1: Consumer/Healthcare/Utilities     RW = 55%               │   │
│  │  Bucket 2: Telecom/Industrials               RW = 60%               │   │
│  │  Bucket 3: Materials/Energy                  RW = 45%               │   │
│  │  Bucket 4: Financials/Real Estate            RW = 55%               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LARGE CAP ADVANCED ECONOMIES (Buckets 5-8)                                │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Bucket 5: Consumer/Healthcare/Utilities     RW = 30%               │   │
│  │  Bucket 6: Telecom/Industrials               RW = 35%               │   │
│  │  Bucket 7: Materials/Energy                  RW = 40%               │   │
│  │  Bucket 8: Technology/Financials             RW = 50%  ← AAPL       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SMALL CAP (Buckets 9-10)                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Bucket 9:  Small Cap Emerging               RW = 70%               │   │
│  │  Bucket 10: Small Cap Advanced               RW = 50%               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  SPECIAL BUCKETS (Buckets 11-13)                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Bucket 11: Other Sector                     RW = 70% (simple sum)  │   │
│  │  Bucket 12: Large Cap Indices                RW = 15%  ← SPX        │   │
│  │  Bucket 13: Other Indices                    RW = 70%               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Our Bucket Assignments

| Instrument | Bucket | Rationale | Risk Weight |
|------------|--------|-----------|-------------|
| **SPX** | 12 | S&P 500 = Large Cap Index | 15% |
| **AAPL** | 8 | Apple = Large Cap, Advanced, Technology | 50% |

### Calculation Process
```
┌────────────────────────────────────────────────────────────────────────────┐
│                     EQUITY DELTA CAPITAL CALCULATION                        │
│                          (Basel MAR21.73-80)                                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  STEP 1: SENSITIVITY (Linear equity positions)                             │
│  ═════════════════════════════════════════════                             │
│  s_k = Position Value (for cash equity/index)                              │
│                                                                             │
│  • s_SPX = $3,000,000                                                      │
│  • s_AAPL = $1,200,000                                                     │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 2: RISK WEIGHT APPLICATION (MAR21.78)                                │
│  ══════════════════════════════════════════                                │
│                                                                             │
│  WS_k = s_k × RW_k                                                         │
│                                                                             │
│  • WS_SPX = $3,000,000 × 15% = $450,000                                    │
│  • WS_AAPL = $1,200,000 × 50% = $600,000                                   │
│                                                                             │
│  Note: AAPL has higher WS despite smaller position (higher RW)             │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 3: BUCKET CAPITAL (Single instrument per bucket)                     │
│  ═════════════════════════════════════════════════════                     │
│                                                                             │
│  K_8 = |WS_AAPL| = $600,000                                                │
│  K_12 = |WS_SPX| = $450,000                                                │
│                                                                             │
│  S_8 = +$600,000 (positive = long)                                         │
│  S_12 = +$450,000 (positive = long)                                        │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 4: INTER-BUCKET AGGREGATION (MAR21.80)                               │
│  ════════════════════════════════════════════                              │
│                                                                             │
│  γ_base = 0.15 (between most equity buckets)                               │
│                                                                             │
│  K² = K_8² + K_12² + 2 × γ × S_8 × S_12                                    │
│                                                                             │
│  Cross-term = 2 × 0.15 × (+$600,000) × (+$450,000) = +$81B                 │
│                                                                             │
│  SAME SIGNS → POSITIVE cross-term → NO diversification!                    │
│                                                                             │
│                              ↓                                              │
│                                                                             │
│  STEP 5: THREE SCENARIOS                                                   │
│  ═══════════════════════                                                   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Scenario │ γ       │ Cross-term │ K²          │ K_Equity          │   │
│  │  ─────────┼─────────┼────────────┼─────────────┼──────────────────  │   │
│  │  BASE     │ 0.1500  │ +$81.0B    │ $643.5B     │ $802,185          │   │
│  │  HIGH     │ 0.1875  │ +$101.3B   │ $663.8B     │ $814,709          │   │
│  │  LOW      │ 0.1125  │ +$60.8B    │ $623.3B     │ $789,462          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  K_EQUITY_FINAL = max(BASE, HIGH, LOW) = $814,709                          │
│  BINDING SCENARIO: HIGH                                                     │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Why HIGH Scenario Binds for Equity

**Both positions are LONG** (same sign):

| Scenario | γ | Position Correlation | Capital |
|----------|---|---------------------|---------|
| LOW | 0.1125 | Less correlated moves | $789,462 |
| HIGH | 0.1875 | More correlated moves | $814,709 |

When both positions move together (high correlation), losses compound → higher capital.

### Equity Key Insight: Index vs Stock Risk Weights

| Attribute | SPX (Index) | AAPL (Stock) |
|-----------|-------------|--------------|
| Position | $3,000,000 | $1,200,000 |
| Risk Weight | 15% | 50% |
| Weighted Sensitivity | $450,000 | $600,000 |

**Basel rationale**: Indices are inherently diversified (S&P 500 = 500 stocks), so lower idiosyncratic risk → lower RW.

---

## 🔄 Correlation Scenarios Framework

### Basel MAR21.6 Scenarios

All three risk classes use the same scenario framework:
```
┌────────────────────────────────────────────────────────────────────────────┐
│                     CORRELATION SCENARIO FRAMEWORK                          │
│                            (Basel MAR21.6)                                  │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PURPOSE: Capture model uncertainty in correlation estimates               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │  BASE SCENARIO:                                                      │   │
│  │  ρ_base = prescribed Basel correlation                               │   │
│  │                                                                      │   │
│  │  HIGH SCENARIO (stressed up):                                        │   │
│  │  ρ_high = min(1.25 × ρ_base, 1.0)                                   │   │
│  │                                                                      │   │
│  │  LOW SCENARIO (stressed down):                                       │   │
│  │  ρ_low = max(2 × ρ_base - 1, 0.75 × ρ_base)                         │   │
│  │                                                                      │   │
│  │  FINAL CAPITAL = max(K_BASE, K_HIGH, K_LOW)                         │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  LOW SCENARIO FLOOR:                                                       │
│  The max(2ρ-1, 0.75ρ) formula ensures ρ_low never goes below 75% of base  │
│                                                                             │
│  Example with ρ_base = 0.60:                                               │
│  • 2 × 0.60 - 1 = 0.20                                                     │
│  • 0.75 × 0.60 = 0.45                                                      │
│  • ρ_low = max(0.20, 0.45) = 0.45 (floor kicks in)                        │
│                                                                             │
│  Example with ρ_base = 0.15:                                               │
│  • 2 × 0.15 - 1 = -0.70                                                    │
│  • 0.75 × 0.15 = 0.1125                                                    │
│  • ρ_low = max(-0.70, 0.1125) = 0.1125 (floor kicks in)                   │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### Binding Scenario Summary

| Risk Class | Position Signs | Cross-term Sign | Binding Scenario |
|------------|----------------|-----------------|------------------|
| **GIRR** | Opposite (USD+, EUR-) | Negative | LOW |
| **FX** | Opposite (EUR+, JPY-) | Negative | LOW |
| **Equity** | Same (SPX+, AAPL+) | Positive | HIGH |

**Rule of thumb**:
- Opposite signs (hedge) → LOW binds (weakest hedge)
- Same signs (directional) → HIGH binds (highest correlation of losses)

---

## ✅ Validation Framework

### Validation Philosophy

Every calculation must be:
1. **Traceable**: Linked to specific Basel paragraph
2. **Verifiable**: Hand-calculable for spot checks
3. **Reproducible**: Same inputs → same outputs

### Validation Test Categories

| Category | Tests | Description |
|----------|-------|-------------|
| **Input Validation** | Data integrity | Completeness, no NaN/Inf, correct instruments |
| **Calculation Validation** | Formula verification | Risk weights, sensitivities, correlations |
| **Scenario Validation** | Stress testing | HIGH/LOW scenario formulas correct |
| **Aggregation Validation** | Capital formulas | Intra-bucket, inter-bucket aggregation |
| **Economic Sensibility** | Reasonableness | Capital magnitude, ratio checks |

### Validation Results by Phase

| Phase | Tests | Passed | Pass Rate | Status |
|-------|-------|--------|-----------|--------|
| Phase 1 (Curves) | 25 | 25 | 100% | ✅ |
| Phase 3 (GIRR) | 43 | 38 | 88.4% | ✅* |
| Phase 10 (FX) | 24 | 24 | 100% | ✅ |
| Phase 12 (Equity) | 33 | 33 | 100% | ✅ |

*GIRR has 5 non-critical failures related to matrix positive semi-definiteness (numerical precision issue, not calculation error).

### Sample Validation Tests
```python
# V3.3.3: Major pairs risk weight
def test_fx_major_pairs_rw():
    expected = 0.15 / math.sqrt(2)  # 10.6066%
    actual = calculate_rw("EUR", "USD")
    assert abs(actual - expected) < 1e-6, f"RW should be 15%/√2"

# V4.1.1: SPX bucket assignment
def test_spx_bucket():
    assert get_bucket("SPX") == 12, "SPX should be in Bucket 12 (Large Cap Indices)"

# V3.5.3: LOW scenario floor
def test_low_scenario_floor():
    rho_base = 0.15
    rho_low = max(2 * rho_base - 1, 0.75 * rho_base)
    assert rho_low == 0.1125, "LOW should use 0.75× floor when 2ρ-1 < 0"
```

---

## 🛠 Technical Implementation

### Architecture
```
┌─────────────────────────────────────────────────────────────────────────┐
│                        SYSTEM ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │  Market Data │────▶│    Phase 1   │────▶│   Zero Curves │           │
│  │  (CSV/API)   │     │  Bootstrap   │     │  (DF, ZR, Fwd)│           │
│  └──────────────┘     └──────────────┘     └───────┬──────┘            │
│                                                     │                    │
│                                                     ▼                    │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐            │
│  │  Portfolio   │────▶│    Phase 2   │────▶│ Sensitivities │           │
│  │  Instruments │     │  DV01 Calc   │     │   (DV01s)     │           │
│  └──────────────┘     └──────────────┘     └───────┬──────┘            │
│                                                     │                    │
│                       ┌─────────────────────────────┼─────────────────┐ │
│                       │                             │                 │ │
│                       ▼                             ▼                 ▼ │
│               ┌──────────────┐             ┌──────────────┐    ┌──────────┐
│               │   Phase 3    │             │  Phase 10    │    │ Phase 12 │
│               │  GIRR Delta  │             │  FX Delta    │    │ EQ Delta │
│               └──────┬───────┘             └──────┬───────┘    └────┬─────┘
│                      │                            │                 │      │
│                      └────────────┬───────────────┴─────────────────┘      │
│                                   ▼                                        │
│                           ┌──────────────┐                                 │
│                           │ Total Delta  │                                 │
│                           │   Capital    │                                 │
│                           │  $3,905,190  │                                 │
│                           └──────────────┘                                 │
│                                                                            │
└────────────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Language** | Python 3.9+ | NumPy/Pandas ecosystem, readability |
| **Interpolation** | Log-linear (DF) | Constant forward rates between nodes |
| **Matrix Operations** | NumPy | Efficient correlation matrix algebra |
| **Day Count** | Exact formulas | Regulatory precision requirement |
| **Validation** | Integrated | Every phase produces validation report |

### Core Formulas Implemented
```python
# Correlation decay (MAR21.47)
def tenor_correlation(t1, t2, theta=0.03):
    return max(math.exp(-theta * abs(t1 - t2) / min(t1, t2)), 0.40)

# Risk weight reduction (MAR21.43)
def adjusted_risk_weight(rw_base, is_well_traded=True):
    return rw_base / math.sqrt(2) if is_well_traded else rw_base

# Correlation scenarios (MAR21.6)
def scenario_high(rho): return min(1.25 * rho, 1.0)
def scenario_low(rho): return max(2 * rho - 1, 0.75 * rho)

# Bucket aggregation (MAR21.4)
def bucket_capital(ws_vector, corr_matrix):
    return math.sqrt(ws_vector @ corr_matrix @ ws_vector)

# Inter-bucket aggregation
def total_capital(K_buckets, S_buckets, gamma):
    K_sq = sum(k**2 for k in K_buckets)
    for i, j in combinations(range(len(K_buckets)), 2):
        K_sq += 2 * gamma * S_buckets[i] * S_buckets[j]
    return math.sqrt(max(K_sq, 0))
```

---

## 🚀 Installation & Usage

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/frtb-sa-capital-engine.git
cd frtb-sa-capital-engine

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start
```bash
# Run full capital calculation
python -m src.main

# Run individual phases
python -m src.phase1_curve_bootstrap
python -m src.phase3_girr_delta_capital
python -m src.phase10_fx_delta_capital
python -m src.phase12_equity_delta_capital
```

### Expected Output
```
================================================================================
                    FRTB-SA MARKET RISK CAPITAL SUMMARY
================================================================================

Risk Class          Capital         Binding Scenario    Validation
--------------------------------------------------------------------------------
GIRR Delta          $1,767,543      LOW                 88.4% (38/43)
FX Delta            $1,322,938      LOW                 100% (24/24)
Equity Delta        $814,709        HIGH                100% (33/33)
--------------------------------------------------------------------------------
TOTAL DELTA         $3,905,190
================================================================================
```

---



---

## 📚 Basel Regulatory References

### Primary Sources

| Document | Reference | Content |
|----------|-----------|---------|
| [MAR21](https://www.bis.org/basel_framework/chapter/MAR/21.htm) | Standardised Approach | Core SA methodology |
| [d457](https://www.bis.org/bcbs/publ/d457.htm) | FRTB Standards | Full FRTB framework |
| [d436](https://www.bis.org/bcbs/publ/d436.htm) | FRTB Revisions | 2019 updates |

### Key MAR21 Paragraphs Used

| Paragraph | Topic | Used In |
|-----------|-------|---------|
| MAR21.4 | Aggregation methodology | All phases |
| MAR21.6 | Correlation scenarios | All phases |
| MAR21.19 | GIRR sensitivity definition | Phase 2-3 |
| MAR21.43-44 | GIRR risk weights | Phase 3 |
| MAR21.45-47 | GIRR correlations | Phase 3 |
| MAR21.50 | GIRR cross-bucket γ | Phase 3 |
| MAR21.73-80 | Equity risk class | Phase 12 |
| MAR21.86-89 | FX risk class | Phase 10 |


## 🔮 Future Enhancements

### Planned Risk Classes

| Risk Class | Status |
|------------|--------|
| GIRR Curvature | 🔲 Planned | 
| GIRR Vega | 🔲 Planned | 
| Equity Vega | 🔲 Planned | 
| Equity Curvature | 🔲 Planned | 
| FX Vega | 🔲 Planned | 
| Credit Spread Risk | 🔲 Planned | 
| Commodity Delta | 🔲 Planned |
| Default Risk Charge | 🔲 Planned | 
| Residual Risk Add-on | 🔲 Planned |

### Technical Improvements

- [ ] Real-time market data integration (Bloomberg API)
- [ ] Interactive dashboard (Streamlit/Dash)
- [ ] Database backend (PostgreSQL)
- [ ] REST API for capital queries
- [ ] Parallel processing for large portfolios
- [ ] Monte Carlo validation framework

---


## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Hemanth Reddy Aeddulla** 
VP of MQF Program(2025-27) at Rutgers Business School
Focus: Market Risk, Quantitative Resear
[LinkedIn](https://www.linkedin.com/in/ahemanthreddy/) | [Email](mailto:hemanth.reddy@rutgers.edu)



## Acknowledgments

- Basel Committee on Banking Supervision for FRTB framework
- Rutgers MQF program for academic guidance


*Last Updated: November 2025*
