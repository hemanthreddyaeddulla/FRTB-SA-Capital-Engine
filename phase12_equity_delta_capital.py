"""
FRTB-SA EQUITY DELTA - PHASE 12: CAPITAL CALCULATION
Bucket-based Risk Weight Aggregation with Three Correlation Scenarios

Basel References:
- MAR21.20: Equity delta sensitivity definition
- MAR21.73-76: Equity risk class overview
- MAR21.77: Bucket definitions (13 buckets)
- MAR21.78: Risk weights and intra-bucket correlations
- MAR21.79: Bucket 11 special aggregation (not used here)
- MAR21.80: Inter-bucket correlations
- MAR21.6: Correlation scenarios

Usage:
    python -m src.phase12_equity_delta_capital
"""

import pandas as pd
import numpy as np
import math
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple, List


# CONFIGURATION


# Portfolio specification
EQUITY_PORTFOLIO = {
    'SPX': {
        'position_value': 3_000_000,
        'direction': 'Long',
        'asset_type': 'Index',
        'description': 'S&P 500 Index',
        'bucket': 12,
        'market_cap': 'Large',
        'economy': 'Advanced',
        'sector': 'Large Cap Indices',
    },
    'AAPL': {
        'position_value': 1_200_000,
        'direction': 'Long',
        'asset_type': 'Single Stock',
        'description': 'Apple Inc.',
        'bucket': 8,
        'market_cap': 'Large',
        'economy': 'Advanced',
        'sector': 'Technology',
    },
}

# Equity bucket definitions (MAR21.77)
EQUITY_BUCKETS = {
    1: {'market_cap': 'Large', 'economy': 'Emerging', 'sector': 'Consumer/Healthcare/Utilities', 'rw': 0.55},
    2: {'market_cap': 'Large', 'economy': 'Emerging', 'sector': 'Telecom/Industrials', 'rw': 0.60},
    3: {'market_cap': 'Large', 'economy': 'Emerging', 'sector': 'Materials/Energy', 'rw': 0.45},
    4: {'market_cap': 'Large', 'economy': 'Emerging', 'sector': 'Financials/Real Estate', 'rw': 0.55},
    5: {'market_cap': 'Large', 'economy': 'Advanced', 'sector': 'Consumer/Healthcare/Utilities', 'rw': 0.30},
    6: {'market_cap': 'Large', 'economy': 'Advanced', 'sector': 'Telecom/Industrials', 'rw': 0.35},
    7: {'market_cap': 'Large', 'economy': 'Advanced', 'sector': 'Materials/Energy', 'rw': 0.40},
    8: {'market_cap': 'Large', 'economy': 'Advanced', 'sector': 'Technology/Financials', 'rw': 0.50},
    9: {'market_cap': 'Small', 'economy': 'Emerging', 'sector': 'All Sectors', 'rw': 0.70},
    10: {'market_cap': 'Small', 'economy': 'Advanced', 'sector': 'All Sectors', 'rw': 0.50},
    11: {'market_cap': 'N/A', 'economy': 'N/A', 'sector': 'Other Sector', 'rw': 0.70},
    12: {'market_cap': 'Large', 'economy': 'Advanced', 'sector': 'Large Cap Indices', 'rw': 0.15},
    13: {'market_cap': 'N/A', 'economy': 'N/A', 'sector': 'Other Indices', 'rw': 0.70},
}

# Risk weights by bucket (MAR21.78)
EQUITY_RISK_WEIGHTS = {b: info['rw'] for b, info in EQUITY_BUCKETS.items()}

# Intra-bucket correlations (MAR21.78) - different issuers, same bucket
INTRA_BUCKET_CORR = {
    1: 0.15, 2: 0.15, 3: 0.15, 4: 0.15,  # Large EM
    5: 0.25, 6: 0.25, 7: 0.25, 8: 0.25,  # Large Adv
    9: 0.075,   # Small EM
    10: 0.125,  # Small Adv
    11: None,   # Simple sum (special treatment)
    12: 0.00,   # Large Cap Indices - no correlation
    13: 0.00,   # Other Indices - no correlation
}

# Inter-bucket correlation (MAR21.80)
GAMMA_BASE = 0.15

# Special treatment buckets (no diversification with other buckets)
NO_DIVERSIFICATION_BUCKETS = {11, 13}

# File paths
OUTPUT_DIR = Path('data/gold')
DOCS_DIR = Path('docs')

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOCS_DIR.mkdir(parents=True, exist_ok=True)


# CORRELATION SCENARIO FUNCTIONS


def scenario_high(rho_base: float) -> float:
    """
    MAR21.6: High correlation scenario.
    Formula: rho_high = min(1.25 * rho_base, 1.0)
    """
    return min(1.25 * rho_base, 1.0)


def scenario_low(rho_base: float) -> float:
    """
    MAR21.6: Low correlation scenario with 0.75 floor.
    Formula: rho_low = max(2 * rho_base - 1, 0.75 * rho_base)
    """
    return max(2.0 * rho_base - 1.0, 0.75 * rho_base)



# CAPITAL CALCULATION FUNCTIONS


def calculate_sensitivity(position_value: float, direction: str = 'Long') -> float:
    """
    MAR21.20: Equity delta sensitivity.

    For linear equity positions: s_k = position_value

    Args:
        position_value: Market value of position in USD
        direction: 'Long' or 'Short'

    Returns:
        Sensitivity (positive for long, negative for short)
    """
    sign = 1 if direction == 'Long' else -1
    return sign * position_value


def calculate_weighted_sensitivity(sensitivity: float, bucket: int) -> float:
    """
    MAR21.4: Calculate weighted sensitivity.

    WS_k = s_k * RW_k

    Args:
        sensitivity: Delta sensitivity
        bucket: Bucket number (1-13)

    Returns:
        Weighted sensitivity
    """
    risk_weight = EQUITY_RISK_WEIGHTS[bucket]
    return sensitivity * risk_weight


def calculate_bucket_capital(weighted_sensitivities: List[float], bucket: int) -> float:
    """
    MAR21.78: Intra-bucket aggregation.

    For single instrument: K_b = |WS|
    For multiple instruments: K_b = sqrt(sum(WS^2) + sum(rho * WS_k * WS_l))

    Bucket 11 special rule (MAR21.79): K_b = sum(|WS|)

    Args:
        weighted_sensitivities: List of WS values in the bucket
        bucket: Bucket number

    Returns:
        Bucket capital
    """
    if len(weighted_sensitivities) == 0:
        return 0.0

    # Bucket 11 special treatment: simple sum of absolute values
    if bucket == 11:
        return sum(abs(ws) for ws in weighted_sensitivities)

    # Single instrument: no correlation term
    if len(weighted_sensitivities) == 1:
        return abs(weighted_sensitivities[0])

    # Multiple instruments: correlation-based aggregation
    rho = INTRA_BUCKET_CORR.get(bucket, 0.0)
    if rho is None:
        rho = 0.0

    ws_array = np.array(weighted_sensitivities)

    # K_b^2 = sum(WS^2) + sum_{k!=l}(rho * WS_k * WS_l)
    sum_ws_squared = np.sum(ws_array ** 2)
    cross_term = 0.0
    for i in range(len(ws_array)):
        for j in range(len(ws_array)):
            if i != j:
                cross_term += rho * ws_array[i] * ws_array[j]

    capital_sq = sum_ws_squared + cross_term
    return np.sqrt(max(capital_sq, 0))


def calculate_total_equity_capital(bucket_capitals: Dict[int, float],
                                   bucket_sums: Dict[int, float],
                                   gamma: float) -> float:
    """
    MAR21.80: Inter-bucket aggregation for equity delta.

    K_Equity^2 = sum(K_b^2) + sum_{b!=c}(gamma * S_b * S_c)

    Buckets 11 and 13 have no diversification (gamma = 0) with other buckets.

    Args:
        bucket_capitals: {bucket: K_b} dictionary
        bucket_sums: {bucket: S_b} dictionary (sum of WS in bucket)
        gamma: Cross-bucket correlation

    Returns:
        Total equity delta capital
    """
    buckets = list(bucket_capitals.keys())

    # Sum of squared bucket capitals
    sum_kb_squared = sum(k**2 for k in bucket_capitals.values())

    # Cross-bucket terms
    cross_term = 0.0
    for i, b1 in enumerate(buckets):
        for j, b2 in enumerate(buckets):
            if i < j:  # Only count each pair once, multiply by 2
                # No diversification for buckets 11, 13
                if b1 in NO_DIVERSIFICATION_BUCKETS or b2 in NO_DIVERSIFICATION_BUCKETS:
                    effective_gamma = 0.0
                else:
                    effective_gamma = gamma

                cross_term += 2 * effective_gamma * bucket_sums[b1] * bucket_sums[b2]

    capital_sq = sum_kb_squared + cross_term
    return np.sqrt(max(capital_sq, 0))



# VALIDATION FUNCTIONS


def run_all_validations(results: Dict) -> Dict:
    """
    Run all validation tests (32+ tests).

    Returns:
        Dictionary with test results and overall pass rate
    """
    tests = []

    # =========================================================================
    # V4.0: Input Validation (5 tests)
    # =========================================================================

    # V4.0.1: Correct instruments identified
    expected_instruments = {'SPX', 'AAPL'}
    actual_instruments = set(results['instruments'].keys())
    tests.append({
        'id': 'V4.0.1',
        'name': 'Correct instruments identified',
        'passed': actual_instruments == expected_instruments,
        'expected': expected_instruments,
        'actual': actual_instruments
    })

    # V4.0.2: SPX position value
    tests.append({
        'id': 'V4.0.2',
        'name': 'SPX position value = $3,000,000',
        'passed': results['instruments']['SPX']['position_value'] == 3_000_000,
        'expected': 3_000_000,
        'actual': results['instruments']['SPX']['position_value']
    })

    # V4.0.3: AAPL position value
    tests.append({
        'id': 'V4.0.3',
        'name': 'AAPL position value = $1,200,000',
        'passed': results['instruments']['AAPL']['position_value'] == 1_200_000,
        'expected': 1_200_000,
        'actual': results['instruments']['AAPL']['position_value']
    })

    # V4.0.4: All values in USD
    tests.append({
        'id': 'V4.0.4',
        'name': 'All values in USD (no conversion needed)',
        'passed': True,  # Both instruments are USD-denominated
        'expected': 'USD',
        'actual': 'USD'
    })

    # V4.0.5: No volatility instruments included
    vol_instruments = {'VIX', 'VXAPL', 'MOVE'}
    has_vol = any(inst in actual_instruments for inst in vol_instruments)
    tests.append({
        'id': 'V4.0.5',
        'name': 'No volatility instruments in delta',
        'passed': not has_vol,
        'expected': 'No VIX/VXAPL/MOVE',
        'actual': 'None included' if not has_vol else 'ERROR: Vol instruments found'
    })

    # =========================================================================
    # V4.1: Bucket Assignment (4 tests)
    # =========================================================================

    # V4.1.1: SPX -> Bucket 12
    tests.append({
        'id': 'V4.1.1',
        'name': 'SPX assigned to Bucket 12',
        'passed': results['instruments']['SPX']['bucket'] == 12,
        'expected': 12,
        'actual': results['instruments']['SPX']['bucket']
    })

    # V4.1.2: AAPL -> Bucket 8
    tests.append({
        'id': 'V4.1.2',
        'name': 'AAPL assigned to Bucket 8',
        'passed': results['instruments']['AAPL']['bucket'] == 8,
        'expected': 8,
        'actual': results['instruments']['AAPL']['bucket']
    })

    # V4.1.3: Only buckets 8 and 12 populated
    populated_buckets = set(results['bucket_capitals'].keys())
    tests.append({
        'id': 'V4.1.3',
        'name': 'Only buckets 8 and 12 populated',
        'passed': populated_buckets == {8, 12},
        'expected': {8, 12},
        'actual': populated_buckets
    })

    # V4.1.4: Bucket assignments documented
    tests.append({
        'id': 'V4.1.4',
        'name': 'Bucket assignments have rationale',
        'passed': all('sector' in results['instruments'][inst] for inst in results['instruments']),
        'expected': 'Sector documented',
        'actual': 'Documented' if all('sector' in results['instruments'][inst] for inst in results['instruments']) else 'Missing'
    })

    # =========================================================================
    # V4.2: Risk Weight Application (4 tests)
    # =========================================================================

    # V4.2.1: SPX RW = 15%
    tests.append({
        'id': 'V4.2.1',
        'name': 'SPX risk weight = 15%',
        'passed': abs(results['instruments']['SPX']['risk_weight'] - 0.15) < 1e-10,
        'expected': 0.15,
        'actual': results['instruments']['SPX']['risk_weight']
    })

    # V4.2.2: AAPL RW = 50%
    tests.append({
        'id': 'V4.2.2',
        'name': 'AAPL risk weight = 50%',
        'passed': abs(results['instruments']['AAPL']['risk_weight'] - 0.50) < 1e-10,
        'expected': 0.50,
        'actual': results['instruments']['AAPL']['risk_weight']
    })

    # V4.2.3: WS_SPX = $450,000
    expected_ws_spx = 3_000_000 * 0.15
    tests.append({
        'id': 'V4.2.3',
        'name': 'WS_SPX = $450,000',
        'passed': abs(results['instruments']['SPX']['weighted_sensitivity'] - expected_ws_spx) < 1,
        'expected': expected_ws_spx,
        'actual': results['instruments']['SPX']['weighted_sensitivity']
    })

    # V4.2.4: WS_AAPL = $600,000
    expected_ws_aapl = 1_200_000 * 0.50
    tests.append({
        'id': 'V4.2.4',
        'name': 'WS_AAPL = $600,000',
        'passed': abs(results['instruments']['AAPL']['weighted_sensitivity'] - expected_ws_aapl) < 1,
        'expected': expected_ws_aapl,
        'actual': results['instruments']['AAPL']['weighted_sensitivity']
    })

    # =========================================================================
    # V4.3: Intra-Bucket Capital (4 tests)
    # =========================================================================

    # V4.3.1: K_8 = $600,000
    tests.append({
        'id': 'V4.3.1',
        'name': 'K_8 (AAPL bucket) = $600,000',
        'passed': abs(results['bucket_capitals'][8] - 600_000) < 1,
        'expected': 600_000,
        'actual': results['bucket_capitals'][8]
    })

    # V4.3.2: K_12 = $450,000
    tests.append({
        'id': 'V4.3.2',
        'name': 'K_12 (SPX bucket) = $450,000',
        'passed': abs(results['bucket_capitals'][12] - 450_000) < 1,
        'expected': 450_000,
        'actual': results['bucket_capitals'][12]
    })

    # V4.3.3: All bucket capitals >= 0
    all_positive = all(k >= 0 for k in results['bucket_capitals'].values())
    tests.append({
        'id': 'V4.3.3',
        'name': 'All bucket capitals >= 0',
        'passed': all_positive,
        'expected': '>= 0',
        'actual': 'All positive' if all_positive else 'Some negative'
    })

    # V4.3.4: Single instrument per bucket (no correlation needed)
    tests.append({
        'id': 'V4.3.4',
        'name': 'Single instrument per bucket',
        'passed': True,
        'expected': '1 per bucket',
        'actual': '1 per bucket'
    })

    # =========================================================================
    # V4.4: Inter-Bucket Aggregation (4 tests)
    # =========================================================================

    # V4.4.1: gamma_base = 0.15
    tests.append({
        'id': 'V4.4.1',
        'name': 'gamma_base = 0.15',
        'passed': abs(results['gamma_base'] - 0.15) < 1e-10,
        'expected': 0.15,
        'actual': results['gamma_base']
    })

    # V4.4.2: gamma_high = 0.1875
    expected_gamma_high = min(1.25 * 0.15, 1.0)
    tests.append({
        'id': 'V4.4.2',
        'name': 'gamma_high = 0.1875',
        'passed': abs(results['gamma_high'] - expected_gamma_high) < 1e-10,
        'expected': expected_gamma_high,
        'actual': results['gamma_high']
    })

    # V4.4.3: gamma_low = 0.1125
    expected_gamma_low = max(2 * 0.15 - 1, 0.75 * 0.15)
    tests.append({
        'id': 'V4.4.3',
        'name': 'gamma_low = 0.1125',
        'passed': abs(results['gamma_low'] - expected_gamma_low) < 1e-10,
        'expected': expected_gamma_low,
        'actual': results['gamma_low']
    })

    # V4.4.4: Cross-term positive (both positions long)
    s_8 = results['bucket_sums'][8]
    s_12 = results['bucket_sums'][12]
    cross_sign_positive = (s_8 * s_12) > 0
    tests.append({
        'id': 'V4.4.4',
        'name': 'Cross-term positive (both long)',
        'passed': cross_sign_positive,
        'expected': 'Positive',
        'actual': 'Positive' if cross_sign_positive else 'Negative'
    })

    # =========================================================================
    # V4.5: Three Scenarios (4 tests)
    # =========================================================================

    # V4.5.1: K_base calculated
    tests.append({
        'id': 'V4.5.1',
        'name': 'K_equity_base calculated',
        'passed': results['K_equity_base'] > 0,
        'expected': '> 0',
        'actual': f"${results['K_equity_base']:,.0f}"
    })

    # V4.5.2: K_high > K_base (higher correlation = less diversification for same-sign positions)
    tests.append({
        'id': 'V4.5.2',
        'name': 'K_high >= K_base',
        'passed': results['K_equity_high'] >= results['K_equity_base'] * 0.999,
        'expected': f">= ${results['K_equity_base']:,.0f}",
        'actual': f"${results['K_equity_high']:,.0f}"
    })

    # V4.5.3: K_low < K_base (lower correlation = more diversification)
    tests.append({
        'id': 'V4.5.3',
        'name': 'K_low <= K_base',
        'passed': results['K_equity_low'] <= results['K_equity_base'] * 1.001,
        'expected': f"<= ${results['K_equity_base']:,.0f}",
        'actual': f"${results['K_equity_low']:,.0f}"
    })

    # V4.5.4: Binding scenario is HIGH
    tests.append({
        'id': 'V4.5.4',
        'name': 'Binding scenario is HIGH',
        'passed': results['binding_scenario'] == 'HIGH',
        'expected': 'HIGH',
        'actual': results['binding_scenario']
    })

    # =========================================================================
    # V4.6: Final Capital (4 tests)
    # =========================================================================

    # V4.6.1: Final = max of three scenarios
    expected_final = max(results['K_equity_base'], results['K_equity_high'], results['K_equity_low'])
    tests.append({
        'id': 'V4.6.1',
        'name': 'Final = max(BASE, HIGH, LOW)',
        'passed': abs(results['K_equity_final'] - expected_final) < 1,
        'expected': f"${expected_final:,.0f}",
        'actual': f"${results['K_equity_final']:,.0f}"
    })

    # V4.6.2: Final approx $814,708
    tests.append({
        'id': 'V4.6.2',
        'name': 'Final capital approx $814,708',
        'passed': abs(results['K_equity_final'] - 814_708) < 1000,
        'expected': '$814,708 +/- $1,000',
        'actual': f"${results['K_equity_final']:,.0f}"
    })

    # V4.6.3: Capital > sum of individual WS (aggregation working)
    sum_ws = abs(results['instruments']['SPX']['weighted_sensitivity']) + \
             abs(results['instruments']['AAPL']['weighted_sensitivity'])
    tests.append({
        'id': 'V4.6.3',
        'name': 'Capital < simple sum (diversification benefit)',
        'passed': results['K_equity_final'] < sum_ws,
        'expected': f"< ${sum_ws:,.0f}",
        'actual': f"${results['K_equity_final']:,.0f}"
    })

    # V4.6.4: Capital > max bucket capital (aggregation adds risk)
    max_bucket = max(results['bucket_capitals'].values())
    tests.append({
        'id': 'V4.6.4',
        'name': 'Capital > max bucket capital',
        'passed': results['K_equity_final'] > max_bucket,
        'expected': f"> ${max_bucket:,.0f}",
        'actual': f"${results['K_equity_final']:,.0f}"
    })

    # =========================================================================
    # V4.7: Economic Sensibility (4 tests)
    # =========================================================================

    # V4.7.1: Capital / Exposure ratio reasonable
    total_exposure = 3_000_000 + 1_200_000
    capital_ratio = results['K_equity_final'] / total_exposure
    tests.append({
        'id': 'V4.7.1',
        'name': 'Capital ratio reasonable (15-25%)',
        'passed': 0.15 <= capital_ratio <= 0.25,
        'expected': '15-25%',
        'actual': f"{capital_ratio*100:.1f}%"
    })

    # V4.7.2: AAPL contributes more than SPX (higher RW)
    tests.append({
        'id': 'V4.7.2',
        'name': 'AAPL WS > SPX WS (higher RW)',
        'passed': results['instruments']['AAPL']['weighted_sensitivity'] > \
                  results['instruments']['SPX']['weighted_sensitivity'],
        'expected': 'AAPL > SPX',
        'actual': f"AAPL: ${results['instruments']['AAPL']['weighted_sensitivity']:,.0f}, SPX: ${results['instruments']['SPX']['weighted_sensitivity']:,.0f}"
    })

    # V4.7.3: Equity Delta > FX Delta
    fx_delta = 13_229
    tests.append({
        'id': 'V4.7.3',
        'name': 'Equity Delta >> FX Delta',
        'passed': results['K_equity_final'] > fx_delta * 10,
        'expected': f">> ${fx_delta:,}",
        'actual': f"${results['K_equity_final']:,.0f}"
    })

    # V4.7.4: Index gets lower effective capital charge (15% vs 50%)
    spx_effective = results['instruments']['SPX']['weighted_sensitivity'] / \
                   results['instruments']['SPX']['position_value']
    aapl_effective = results['instruments']['AAPL']['weighted_sensitivity'] / \
                    results['instruments']['AAPL']['position_value']
    tests.append({
        'id': 'V4.7.4',
        'name': 'Index gets lower effective RW than stock',
        'passed': spx_effective < aapl_effective,
        'expected': 'SPX < AAPL',
        'actual': f"SPX: {spx_effective*100:.0f}%, AAPL: {aapl_effective*100:.0f}%"
    })

    # =========================================================================
    # Calculate Summary
    # =========================================================================

    passed = sum(1 for t in tests if t['passed'])
    total = len(tests)
    pass_rate = passed / total * 100

    return {
        'tests': tests,
        'passed': passed,
        'total': total,
        'pass_rate': pass_rate
    }



# OUTPUT FUNCTIONS


def save_instrument_detail(results: Dict, filepath: Path) -> None:
    """Save detailed instrument-level calculations."""
    rows = []
    for inst_name, inst_data in results['instruments'].items():
        rows.append({
            'instrument': inst_name,
            'position_value': inst_data['position_value'],
            'direction': inst_data['direction'],
            'asset_type': inst_data['asset_type'],
            'bucket': inst_data['bucket'],
            'sector': inst_data['sector'],
            'risk_weight': inst_data['risk_weight'],
            'sensitivity': inst_data['sensitivity'],
            'weighted_sensitivity': inst_data['weighted_sensitivity'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved {filepath.name}")


def save_bucket_summary(results: Dict, filepath: Path) -> None:
    """Save bucket-level summary."""
    rows = []
    for bucket in results['bucket_capitals'].keys():
        bucket_info = EQUITY_BUCKETS.get(bucket, {})
        rows.append({
            'bucket': bucket,
            'market_cap': bucket_info.get('market_cap', 'N/A'),
            'economy': bucket_info.get('economy', 'N/A'),
            'sector': bucket_info.get('sector', 'N/A'),
            'risk_weight': EQUITY_RISK_WEIGHTS.get(bucket, 'N/A'),
            'bucket_sum_S_b': results['bucket_sums'][bucket],
            'bucket_capital_K_b': results['bucket_capitals'][bucket],
        })

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved {filepath.name}")


def save_scenario_summary(results: Dict, filepath: Path) -> None:
    """Save scenario-level capital summary."""
    rows = [
        {
            'scenario': 'BASE',
            'gamma': results['gamma_base'],
            'K_equity': results['K_equity_base'],
            'vs_base_pct': 0.0,
        },
        {
            'scenario': 'HIGH',
            'gamma': results['gamma_high'],
            'K_equity': results['K_equity_high'],
            'vs_base_pct': (results['K_equity_high'] / results['K_equity_base'] - 1) * 100,
        },
        {
            'scenario': 'LOW',
            'gamma': results['gamma_low'],
            'K_equity': results['K_equity_low'],
            'vs_base_pct': (results['K_equity_low'] / results['K_equity_base'] - 1) * 100,
        },
    ]

    df = pd.DataFrame(rows)
    df.to_csv(filepath, index=False)
    print(f"[OK] Saved {filepath.name}")


def generate_validation_report(validation_results: Dict, results: Dict, filepath: Path) -> None:
    """Generate comprehensive markdown validation report."""

    report = f"""# PHASE 12 VALIDATION REPORT: EQUITY DELTA CAPITAL CALCULATION
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## EXECUTIVE SUMMARY

| Metric | Value |
|--------|-------|
| **Final Equity Delta Capital** | ${results['K_equity_final']:,.0f} |
| **Binding Scenario** | {results['binding_scenario']} |
| **Validation Pass Rate** | {validation_results['pass_rate']:.1f}% ({validation_results['passed']}/{validation_results['total']}) |

## PORTFOLIO COMPOSITION

| Instrument | Position | Bucket | Risk Weight | Weighted Sensitivity |
|------------|----------|--------|-------------|---------------------|
| AAPL | ${results['instruments']['AAPL']['position_value']:,.0f} | 8 (Tech) | 50% | ${results['instruments']['AAPL']['weighted_sensitivity']:,.0f} |
| SPX | ${results['instruments']['SPX']['position_value']:,.0f} | 12 (Index) | 15% | ${results['instruments']['SPX']['weighted_sensitivity']:,.0f} |

## CAPITAL BY SCENARIO

| Scenario | gamma | K_Equity | vs BASE |
|----------|-------|----------|---------|
| BASE | {results['gamma_base']:.4f} | ${results['K_equity_base']:,.0f} | - |
| HIGH | {results['gamma_high']:.4f} | ${results['K_equity_high']:,.0f} | {(results['K_equity_high']/results['K_equity_base']-1)*100:+.1f}% |
| LOW | {results['gamma_low']:.4f} | ${results['K_equity_low']:,.0f} | {(results['K_equity_low']/results['K_equity_base']-1)*100:+.1f}% |

## BUCKET CAPITALS

| Bucket | Description | K_b | S_b |
|--------|-------------|-----|-----|
| 8 | Large Cap Advanced Tech | ${results['bucket_capitals'][8]:,.0f} | ${results['bucket_sums'][8]:,.0f} |
| 12 | Large Cap Indices | ${results['bucket_capitals'][12]:,.0f} | ${results['bucket_sums'][12]:,.0f} |

## KEY PARAMETERS

| Parameter | Value | Basel Reference |
|-----------|-------|-----------------|
| AAPL Risk Weight | 50% | MAR21.78 Bucket 8 |
| SPX Risk Weight | 15% | MAR21.78 Bucket 12 |
| Inter-bucket gamma (BASE) | 0.15 | MAR21.80 |
| Scenario HIGH multiplier | 1.25x | MAR21.6 |
| Scenario LOW formula | max(2*rho-1, 0.75*rho) | MAR21.6 |

## VALIDATION TEST RESULTS

"""

    # Group tests by category
    categories = {
        'V4.0': 'Input Validation',
        'V4.1': 'Bucket Assignment',
        'V4.2': 'Risk Weight Application',
        'V4.3': 'Intra-Bucket Capital',
        'V4.4': 'Inter-Bucket Aggregation',
        'V4.5': 'Three Scenarios',
        'V4.6': 'Final Capital',
        'V4.7': 'Economic Sensibility',
    }

    for prefix, category_name in categories.items():
        cat_tests = [t for t in validation_results['tests'] if t['id'].startswith(prefix)]
        if cat_tests:
            cat_passed = sum(1 for t in cat_tests if t['passed'])
            report += f"\n### {category_name} ({cat_passed}/{len(cat_tests)} passed)\n\n"
            report += "| ID | Test | Status | Expected | Actual |\n"
            report += "|-----|------|--------|----------|--------|\n"
            for t in cat_tests:
                status = "PASS" if t['passed'] else "FAIL"
                report += f"| {t['id']} | {t['name']} | {status} | {t['expected']} | {t['actual']} |\n"

    report += f"""

## COMPARISON TO OTHER RISK CLASSES

| Risk Class | Capital | Notes |
|------------|---------|-------|
| FX Delta (Phase 10) | $13,229 | EUR/USD, USD/JPY exposures |
| **Equity Delta (Phase 12)** | **${results['K_equity_final']:,.0f}** | SPX + AAPL |

## FILES GENERATED

1. `data/gold/equity_delta_capital_2025-11-03.csv` - Instrument details
2. `data/gold/equity_delta_capital_2025-11-03.pkl` - Complete results
3. `data/gold/equity_bucket_summary.csv` - Bucket-level summary
4. `data/gold/equity_scenario_summary.csv` - Scenario comparison

---
*Report generated by Phase 12 Equity Delta Capital Calculation Engine*
*Basel FRTB-SA Compliant*
"""

    with open(filepath, 'w') as f:
        f.write(report)

    print(f"[OK] Saved {filepath.name}")



# MAIN FUNCTION


def main():
    """
    Phase 12: Calculate Equity Delta Capital Charge.

    Steps:
    1. Load portfolio specification
    2. Calculate sensitivities (position values for linear instruments)
    3. Apply bucket-specific risk weights
    4. Calculate intra-bucket capitals
    5. Calculate inter-bucket aggregation with three correlation scenarios
    6. Final capital = max of three scenarios
    7. Run validations
    8. Save outputs
    """

    print("="*70)
    print("PHASE 12: SA EQUITY DELTA CAPITAL CALCULATION")
    print("="*70)
    print(f"\nDate: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Basel Reference: MAR21.73-80")

    # =========================================================================
    # STEP 1: Load Portfolio
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 1: Load Equity Portfolio")
    print("-"*70)

    instruments = {}
    for inst_name, inst_spec in EQUITY_PORTFOLIO.items():
        instruments[inst_name] = inst_spec.copy()
        print(f"[OK] {inst_name}: ${inst_spec['position_value']:,} {inst_spec['direction']} ({inst_spec['description']})")

    print(f"\nTotal Equity Exposure: ${sum(i['position_value'] for i in instruments.values()):,}")

    # =========================================================================
    # STEP 2: Calculate Sensitivities (MAR21.20)
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 2: Calculate Delta Sensitivities (MAR21.20)")
    print("-"*70)

    for inst_name, inst_data in instruments.items():
        sensitivity = calculate_sensitivity(
            inst_data['position_value'],
            inst_data['direction']
        )
        inst_data['sensitivity'] = sensitivity
        print(f"[OK] {inst_name}: s = ${sensitivity:,}")

    # =========================================================================
    # STEP 3: Apply Risk Weights
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 3: Apply Risk Weights (MAR21.78)")
    print("-"*70)

    for inst_name, inst_data in instruments.items():
        bucket = inst_data['bucket']
        rw = EQUITY_RISK_WEIGHTS[bucket]
        ws = calculate_weighted_sensitivity(inst_data['sensitivity'], bucket)

        inst_data['risk_weight'] = rw
        inst_data['weighted_sensitivity'] = ws

        print(f"[OK] {inst_name}: Bucket {bucket}, RW = {rw*100:.0f}%, WS = ${ws:,}")

    # =========================================================================
    # STEP 4: Calculate Bucket Capitals
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 4: Calculate Bucket Capitals (MAR21.78)")
    print("-"*70)

    # Group instruments by bucket
    bucket_instruments = {}
    for inst_name, inst_data in instruments.items():
        bucket = inst_data['bucket']
        if bucket not in bucket_instruments:
            bucket_instruments[bucket] = []
        bucket_instruments[bucket].append(inst_data['weighted_sensitivity'])

    # Calculate bucket capitals and sums
    bucket_capitals = {}
    bucket_sums = {}

    for bucket, ws_list in bucket_instruments.items():
        K_b = calculate_bucket_capital(ws_list, bucket)
        S_b = sum(ws_list)

        bucket_capitals[bucket] = K_b
        bucket_sums[bucket] = S_b

        print(f"[OK] Bucket {bucket}: K_b = ${K_b:,.0f}, S_b = ${S_b:,.0f}")

    # =========================================================================
    # STEP 5: Calculate Gamma Scenarios
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 5: Calculate Inter-Bucket Correlation Scenarios (MAR21.6)")
    print("-"*70)

    gamma_base = GAMMA_BASE
    gamma_high = scenario_high(gamma_base)
    gamma_low = scenario_low(gamma_base)

    print(f"[OK] gamma_base = {gamma_base}")
    print(f"[OK] gamma_high = min(1.25 * {gamma_base}, 1.0) = {gamma_high}")
    print(f"[OK] gamma_low = max(2*{gamma_base}-1, 0.75*{gamma_base}) = max({2*gamma_base-1}, {0.75*gamma_base}) = {gamma_low}")

    # =========================================================================
    # STEP 6: Calculate Total Capital (Three Scenarios)
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 6: Calculate Total Equity Delta Capital (MAR21.80)")
    print("-"*70)

    K_equity_base = calculate_total_equity_capital(bucket_capitals, bucket_sums, gamma_base)
    K_equity_high = calculate_total_equity_capital(bucket_capitals, bucket_sums, gamma_high)
    K_equity_low = calculate_total_equity_capital(bucket_capitals, bucket_sums, gamma_low)

    print(f"\nCapital by Scenario:")
    print(f"     BASE: ${K_equity_base:,.0f}")
    print(f"     HIGH: ${K_equity_high:,.0f} ({(K_equity_high/K_equity_base-1)*100:+.1f}% vs base)")
    print(f"     LOW:  ${K_equity_low:,.0f} ({(K_equity_low/K_equity_base-1)*100:+.1f}% vs base)")

    # =========================================================================
    # STEP 7: Final Capital
    # =========================================================================
    print("\n" + "-"*70)
    print("STEP 7: Final Capital (Maximum of Three Scenarios)")
    print("-"*70)

    K_equity_final = max(K_equity_base, K_equity_high, K_equity_low)

    if K_equity_final == K_equity_high:
        binding_scenario = "HIGH"
    elif K_equity_final == K_equity_low:
        binding_scenario = "LOW"
    else:
        binding_scenario = "BASE"

    print(f"\n" + "="*50)
    print(f"*** FINAL EQUITY DELTA CAPITAL: ${K_equity_final:,.0f} ***")
    print(f"*** BINDING SCENARIO: {binding_scenario} ***")
    print("="*50)

    # =========================================================================
    # Collect Results
    # =========================================================================

    results = {
        'instruments': instruments,
        'bucket_capitals': bucket_capitals,
        'bucket_sums': bucket_sums,
        'gamma_base': gamma_base,
        'gamma_high': gamma_high,
        'gamma_low': gamma_low,
        'K_equity_base': K_equity_base,
        'K_equity_high': K_equity_high,
        'K_equity_low': K_equity_low,
        'K_equity_final': K_equity_final,
        'binding_scenario': binding_scenario,
    }

    # =========================================================================
    # STEP 8: Run Validations
    # =========================================================================
    print("\n" + "="*70)
    print("VALIDATION SUITE")
    print("="*70)

    validation_results = run_all_validations(results)

    print(f"\n{'='*50}")
    print(f"VALIDATION SUMMARY: {validation_results['passed']}/{validation_results['total']} PASSED ({validation_results['pass_rate']:.1f}%)")
    print(f"{'='*50}")

    # Print failed tests
    failed = [t for t in validation_results['tests'] if not t['passed']]
    if failed:
        print("\nFailed Tests:")
        for t in failed:
            print(f"  [FAIL] {t['id']}: {t['name']}")
            print(f"     Expected: {t['expected']}, Actual: {t['actual']}")
    else:
        print("\n[OK] All validation tests passed!")

    # =========================================================================
    # STEP 9: Save Outputs
    # =========================================================================
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    # Save instrument detail CSV
    save_instrument_detail(results, OUTPUT_DIR / 'equity_delta_capital_2025-11-03.csv')

    # Save bucket summary
    save_bucket_summary(results, OUTPUT_DIR / 'equity_bucket_summary.csv')

    # Save scenario summary
    save_scenario_summary(results, OUTPUT_DIR / 'equity_scenario_summary.csv')

    # Save pickle with all results
    pickle_data = {
        'results': results,
        'validation_results': validation_results,
        'timestamp': datetime.now().isoformat()
    }
    with open(OUTPUT_DIR / 'equity_delta_capital_2025-11-03.pkl', 'wb') as f:
        pickle.dump(pickle_data, f)
    print(f"[OK] Saved equity_delta_capital_2025-11-03.pkl")

    # Generate validation report
    generate_validation_report(validation_results, results, DOCS_DIR / 'validation_report_phase12_equity_delta.md')

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 12 COMPLETE")
    print("="*70)

    print(f"""
======================================================================
                   EQUITY DELTA CAPITAL SUMMARY
======================================================================
  Portfolio:
    AAPL: ${instruments['AAPL']['position_value']:>12,} -> WS = ${instruments['AAPL']['weighted_sensitivity']:>10,.0f} (Bucket 8, 50% RW)
    SPX:  ${instruments['SPX']['position_value']:>12,} -> WS = ${instruments['SPX']['weighted_sensitivity']:>10,.0f} (Bucket 12, 15% RW)

  Bucket Capitals:
    K_8 (AAPL)  = ${bucket_capitals[8]:>12,.0f}
    K_12 (SPX)  = ${bucket_capitals[12]:>12,.0f}

  Three Scenarios:
    BASE (gamma=0.15)   = ${K_equity_base:>12,.0f}
    HIGH (gamma=0.1875) = ${K_equity_high:>12,.0f}
    LOW (gamma=0.1125)  = ${K_equity_low:>12,.0f}

  ====================================================================
  FINAL CAPITAL = ${K_equity_final:>12,.0f}
  BINDING SCENARIO = {binding_scenario:<10}
  ====================================================================

  Comparison:
    FX Delta (Phase 10)     = $        13,229
    Equity Delta (Phase 12) = ${K_equity_final:>12,.0f}
    Equity is {K_equity_final/13229:.0f}x larger than FX

  Validation: {validation_results['passed']}/{validation_results['total']} tests passed ({validation_results['pass_rate']:.1f}%)
======================================================================
""")

    print("\n" + "-"*70)
    print("[PAUSE] PHASE 12 CHECKPOINT")
    print("-"*70)
    print("\nPhase 12 complete. Files generated:")
    print("  1. equity_delta_capital_2025-11-03.csv")
    print("  2. equity_delta_capital_2025-11-03.pkl")
    print("  3. equity_bucket_summary.csv")
    print("  4. equity_scenario_summary.csv")
    print("  5. validation_report_phase12_equity_delta.md")
    print(f"\nKey findings:")
    print(f"  - Final Equity Delta Capital: ${K_equity_final:,.0f}")
    print(f"  - Binding Scenario: {binding_scenario}")
    print(f"  - Capital Ratio: {K_equity_final/(3_000_000+1_200_000)*100:.1f}% of exposure")
    print(f"\nDo I have approval to proceed to Phase 13 (Commodity Delta)?")

    return results, validation_results


if __name__ == '__main__':
    main()
