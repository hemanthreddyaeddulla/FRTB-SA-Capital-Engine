# FRTB-SA-Capital-Engine
Basel III FRTB Standardised Approach (SA) market risk capital calculation engine implementing GIRR, FX, and Equity Delta risk classes with full regulatory validation

# Basel FRTB-SA Capital Engine

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Production-grade implementation of the **Basel III Fundamental Review of the Trading Book (FRTB) Standardised Approach (SA)** for market risk capital calculation.

## Overview

This engine calculates regulatory capital charges for market risk under the Basel FRTB-SA framework, implementing:

| Risk Class | Basel Reference | Status |
|------------|-----------------|--------|
| **GIRR Delta** | MAR21.19-51 | ✅ Complete |
| **FX Delta** | MAR21.86-89 | ✅ Complete |
| **Equity Delta** | MAR21.73-80 | ✅ Complete |
| GIRR Curvature | MAR21.5-8 | 🔲 Planned |
| Equity Vega | MAR21.73-80 | 🔲 Planned |

## Key Features

- **Full Basel Compliance**: Every calculation traceable to MAR21 paragraphs
- **Three Correlation Scenarios**: BASE, HIGH, LOW per MAR21.6
- **Comprehensive Validation**: 100+ automated tests with regulatory references
- **Production-Ready**: Modular design, extensive documentation, audit trail

## Capital Results Summary

| Risk Class | Capital Charge | Binding Scenario | Validation |
|------------|----------------|------------------|------------|
| GIRR Delta | $1,767,543 | LOW | 88.4% (38/43 tests) |
| FX Delta | $1,322,938 | LOW | 100% (24/24 tests) |
| Equity Delta | $814,709 | HIGH | 100% (33/33 tests) |

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourusername/frtb-sa-capital-engine.git
cd frtb-sa-capital-engine

# Install dependencies
pip install -r requirements.txt

# Run capital calculation
python -m src.phase12_equity_delta_capital
```

## Project Structure
```
frtb-sa-capital-engine/
├── src/                    # Core calculation modules
├── docs/                   # Methodology & validation reports
├── tests/                  # Unit & integration tests
├── config/                 # Basel parameters & portfolio specs
├── data/                   # Market data & outputs
└── notebooks/              # Analysis notebooks
```

## Methodology

### GIRR Delta (Interest Rate Risk)
- **Instruments**: Treasury bonds, SOFR swaps, EUR swaps
- **Risk Factors**: 30 (10 tenors × 3 curves)
- **Correlation**: Exponential decay θ=0.03 (MAR21.47)
- **Aggregation**: Intra-bucket correlation + cross-bucket γ=0.50

### FX Delta (Foreign Exchange Risk)
- **Instruments**: EUR/USD, USD/JPY positions
- **Risk Weight**: 15%/√2 = 10.6% for major pairs (MAR21.88)
- **Cross-Bucket γ**: 0.60 (MAR21.89)

### Equity Delta
- **Instruments**: SPX Index (Bucket 12), AAPL Stock (Bucket 8)
- **Risk Weights**: 15% (indices), 50% (large cap tech)
- **Cross-Bucket γ**: 0.15 (MAR21.80)

## Correlation Scenarios (MAR21.6)

| Scenario | Formula | Purpose |
|----------|---------|---------|
| BASE | ρ | Standard correlation |
| HIGH | min(1.25×ρ, 1.0) | Stressed correlation (up) |
| LOW | max(2ρ−1, 0.75×ρ) | Stressed correlation (down) |

Final Capital = max(K_BASE, K_HIGH, K_LOW)

## Validation Framework

Each phase includes comprehensive validation tests:

- **Input Validation**: Data integrity, completeness
- **Calculation Validation**: Basel formula verification
- **Scenario Validation**: Correlation stress tests
- **Economic Sensibility**: Reasonableness checks

See `docs/validation_reports/` for detailed test results.

## Basel References

Key regulatory documents:
- [MAR21: Standardised Approach](https://www.bis.org/basel_framework/chapter/MAR/21.htm)
- [FRTB Standards (d457)](https://www.bis.org/bcbs/publ/d457.htm)

## Requirements

- Python 3.9+
- NumPy, Pandas, SciPy
- See `requirements.txt` for full dependencies

## License

MIT License - see [LICENSE](LICENSE) for details.

## Author

**Hemanth Reddy Aeddulla**  
Master of Quantitative Finance, Rutgers Business School  
[LinkedIn](https://www.linkedin.com/in/ahemanthreddy/) | [Email](mailto:hemanth.reddy@rutgers.edu)

## Acknowledgments

- Basel Committee on Banking Supervision for FRTB framework
- Rutgers MQF program for academic guidance
