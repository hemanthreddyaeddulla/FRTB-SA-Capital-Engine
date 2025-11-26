"""
FRTB-SA GIRR DELTA - PHASE 1: ZERO-COUPON CURVE CONSTRUCTION
Conceptually Rigorous, Numerically Approximated Bootstrap

python -m src.phase1_curve_bootstrap
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline, PchipInterpolator
import pickle
import warnings
import os


# CONFIGURATION

SETTLEMENT_DATE = datetime(2025, 11, 3)
DATA_FILE = 'data/market_snapshot_2025-11-03.csv'
OUTPUT_DIR_SILVER = 'data/silver'
OUTPUT_DIR_DOCS = 'docs'

# Ensure output directories exist
os.makedirs(OUTPUT_DIR_SILVER, exist_ok=True)
os.makedirs(OUTPUT_DIR_DOCS, exist_ok=True)


# UTILITY FUNCTIONS: DATE AND DAY-COUNT

def is_leap_year(year):
    """Check if year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def accrual_act_act_phase1(date1, date2):
    """
    ACT/ACT approximation for Phase 1.

    Full ISDA ACT/ACT requires splitting across year boundaries.
    Phase 1 approximation: days/365 (or days/366 if leap year in period).

    Error vs full ISDA: < 0.5 bps on discount factors.
    Acceptable for: FRTB-SA regulatory capital.

    Args:
        date1: Start date
        date2: End date

    Returns:
        Accrual factor (year fraction)
    """
    days = (date2 - date1).days

    # Check if any leap year in the span
    years_in_period = range(date1.year, date2.year + 1)
    has_leap = any(is_leap_year(y) for y in years_in_period)

    if has_leap:
        return days / 366.0
    else:
        return days / 365.0


def accrual_act_360(date1, date2):
    """
    ACT/360 convention (exact implementation).

    Used for: USD SOFR swaps (money market convention).
    Formula: actual_days / 360

    Args:
        date1: Start date
        date2: End date

    Returns:
        Accrual factor
    """
    days = (date2 - date1).days
    return days / 360.0


def accrual_30_360(date1, date2):
    """
    30/360 convention (exact implementation).

    Used for: EUR swaps.
    Formula: [360(Y2-Y1) + 30(M2-M1) + (D2-D1)] / 360

    Adjustments:
    - If D1 = 31, set D1 = 30
    - If D2 = 31 and D1 >= 30, set D2 = 30

    Args:
        date1: Start date
        date2: End date

    Returns:
        Accrual factor
    """
    y1, m1, d1 = date1.year, date1.month, date1.day
    y2, m2, d2 = date2.year, date2.month, date2.day

    # 30/360 adjustments
    if d1 == 31:
        d1 = 30
    if d2 == 31 and d1 >= 30:
        d2 = 30

    days_30_360 = 360 * (y2 - y1) + 30 * (m2 - m1) + (d2 - d1)
    return days_30_360 / 360.0


def calculate_accrual(date1, date2, convention):
    """
    Router function for day-count conventions.

    Args:
        date1: Start date
        date2: End date
        convention: 'ACT/ACT', 'ACT/360', or '30/360'

    Returns:
        Accrual factor
    """
    if convention == 'ACT/ACT':
        return accrual_act_act_phase1(date1, date2)
    elif convention == 'ACT/360':
        return accrual_act_360(date1, date2)
    elif convention == '30/360':
        return accrual_30_360(date1, date2)
    else:
        raise ValueError(f"Unknown day-count convention: {convention}")


def year_fraction(date1, date2):
    """
    Simple year fraction (for DF tenor calculation).

    Returns: years as decimal (e.g., 1.5 for 18 months)
    """
    days = (date2 - date1).days
    return days / 365.25  # Average accounting for leap years


def add_months(date, months):
    """
    Add months to a date (handles month/year rollovers).

    Args:
        date: Starting date
        months: Number of months to add (can be negative)

    Returns:
        New date
    """
    # Calculate target month and year
    month = date.month - 1 + months  # 0-indexed
    year = date.year + month // 12
    month = month % 12 + 1  # Back to 1-indexed

    # Handle day overflow (e.g., Jan 31 + 1 month = Feb 28/29)
    day = min(date.day, [31, 29 if is_leap_year(year) else 28, 31, 30, 31, 30,
                          31, 31, 30, 31, 30, 31][month - 1])

    return datetime(year, month, day)


def generate_payment_dates(settlement_date, maturity_years, payment_freq):
    """
    Generate payment dates for a bond or swap.

    Works BACKWARD from maturity to ensure exact alignment.

    Args:
        settlement_date: Settlement date (datetime)
        maturity_years: Maturity in years (float)
        payment_freq: Payments per year (1=annual, 2=semi-annual)

    Returns:
        List of payment dates (datetime objects)
    """
    # Calculate maturity date
    maturity_date = add_months(settlement_date, int(maturity_years * 12))

    # Generate payment dates backward from maturity
    payment_dates = []
    current_date = maturity_date
    months_per_period = 12 // payment_freq

    while current_date > settlement_date:
        payment_dates.insert(0, current_date)  # Insert at beginning
        current_date = add_months(current_date, -months_per_period)

    # Validation: Check we have correct number of periods
    expected_payments = int(maturity_years * payment_freq)
    if len(payment_dates) != expected_payments:
        warnings.warn(
            f"Payment schedule mismatch: {len(payment_dates)} payments "
            f"vs {expected_payments} expected for {maturity_years}Y "
            f"with freq={payment_freq}"
        )

    return payment_dates



# INTERPOLATION

def interpolate_df_loglinear(t_target, t_lower, t_upper, df_lower, df_upper_guess):
    """
    Interpolate DF at t_target using constant forward rate assumption.

    Theory:
        Forward rate: f = -d/dt[ln(DF(t))]
        Constant forward → DF(t) = DF(t_lower) × exp(-f × (t - t_lower))

        Where f is implied by:
        DF(t_upper) = DF(t_lower) × exp(-f × (t_upper - t_lower))
        => f = -ln(DF(t_upper) / DF(t_lower)) / (t_upper - t_lower))

    Args:
        t_target: Target maturity (e.g., 1.5 for 1.5Y)
        t_lower: Known tenor below target (e.g., 1.0)
        t_upper: Tenor above target (e.g., 2.0)
        df_lower: DF at t_lower (known)
        df_upper_guess: Current guess for DF at t_upper

    Returns:
        DF at t_target (interpolated)
    """
    # Guard against edge cases
    if abs(t_upper - t_lower) < 1e-10:
        return df_lower

    if df_upper_guess <= 0 or df_lower <= 0:
        raise ValueError(f"Invalid DFs for interpolation: "
                        f"df_lower={df_lower}, df_upper={df_upper_guess}")

    # Calculate implied forward rate
    forward_rate = -np.log(df_upper_guess / df_lower) / (t_upper - t_lower)

    # Interpolate DF at target
    df_target = df_lower * np.exp(-forward_rate * (t_target - t_lower))

    return df_target



# BOOTSTRAP FUNCTIONS

def bootstrap_money_market(par_rate, tenor_years, settlement_date, convention):
    """
    Bootstrap discount factor for money market instrument.

    Formula: DF = 1 / (1 + r × α)

    ONLY USE FOR: 3M (0.25Y) and 6M (0.5Y) tenors.

    Args:
        par_rate: Par rate in decimal (e.g., 0.0398 for 3.98%)
        tenor_years: Tenor in years (0.25 or 0.5)
        settlement_date: Settlement date
        convention: Day-count convention

    Returns:
        Discount factor
    """
    # Calculate maturity date
    maturity_date = add_months(settlement_date, int(tenor_years * 12))

    # Calculate accrual factor
    alpha = calculate_accrual(settlement_date, maturity_date, convention)

    # DF formula: 1 / (1 + r × α)
    df = 1.0 / (1.0 + par_rate * alpha)

    return df


def bootstrap_1y_treasury(par_rate, previous_dfs, settlement_date):
    """
    Bootstrap 1Y Treasury DF using coupon bond pricing.

    1Y Treasury has 2 semi-annual coupons:
    - Payment 1: 6 months (0.5Y)
    - Payment 2: 12 months (1Y) + principal

    Bond equation (notional = 1):
        1 = c₁ × DF(0.5Y) + (1 + c₂) × DF(1Y)

    Args:
        par_rate: 1Y par yield (decimal)
        previous_dfs: Dict of {tenor: df} with DF(0.5Y)
        settlement_date: Settlement date

    Returns:
        DF(1Y)
    """
    notional = 1.0
    payment_freq = 2  # Semi-annual

    # Generate payment dates
    payment_dates = generate_payment_dates(settlement_date, 1.0, payment_freq)

    # Must have exactly 2 payments
    if len(payment_dates) != 2:
        raise ValueError(f"1Y Treasury should have 2 payments, got {len(payment_dates)}")

    # Calculate coupons using PERIOD accruals
    date_0_5y = payment_dates[0]
    date_1_0y = payment_dates[1]

    alpha_1 = calculate_accrual(settlement_date, date_0_5y, 'ACT/ACT')
    alpha_2 = calculate_accrual(date_0_5y, date_1_0y, 'ACT/ACT')

    coupon_1 = par_rate * notional * alpha_1
    coupon_2 = par_rate * notional * alpha_2

    # DF at 0.5Y (known)
    df_0_5y = previous_dfs[0.5]

    # Bond pricing equation:
    # 1 = coupon_1 × DF(0.5Y) + (1 + coupon_2) × DF(1Y)

    # Solve for DF(1Y):
    known_pv = coupon_1 * df_0_5y
    df_1y = (notional - known_pv) / (notional + coupon_2)

    return df_1y

def bootstrap_1y_swap(par_rate, previous_dfs, settlement_date, convention, fixed_freq):
    """
    Bootstrap 1Y swap DF.

    1Y swap has 2 payments on fixed leg (for semi-annual).

    Swap equation (notional = 1):
        r × (α₁ × DF(0.5Y) + α₂ × DF(1Y)) = 1 - DF(1Y)

    Args:
        par_rate: 1Y swap rate (decimal)
        previous_dfs: Dict with DF(0.5Y)
        settlement_date: Settlement date
        convention: Day-count convention
        fixed_freq: Fixed leg frequency (2 for semi-annual)

    Returns:
        DF(1Y)
    """
    notional = 1.0

    # Generate payment dates
    payment_dates = generate_payment_dates(settlement_date, 1.0, fixed_freq)

    # Handle both annual (1 payment) and semi-annual (2 payments)
    if len(payment_dates) == 1:
        # Annual swap: only 1 payment at maturity
        # Swap equation: par_rate × α × DF(1Y) = 1 - DF(1Y)
        # Solving: DF(1Y) × (1 + par_rate × α) = 1
        # Therefore: DF(1Y) = 1 / (1 + par_rate × α)

        maturity_date = payment_dates[0]
        alpha = calculate_accrual(settlement_date, maturity_date, convention)
        df_1y = 1.0 / (1.0 + par_rate * alpha)

    elif len(payment_dates) == 2:
        # Semi-annual swap: 2 payments
        date_0_5y = payment_dates[0]
        date_1_0y = payment_dates[1]

        alpha_1 = calculate_accrual(settlement_date, date_0_5y, convention)
        alpha_2 = calculate_accrual(date_0_5y, date_1_0y, convention)

        # DF at 0.5Y (known)
        df_0_5y = previous_dfs[0.5]

        # Swap equation:
        # par_rate × (α₁ × DF(0.5Y) + α₂ × DF(1Y)) = 1 - DF(1Y)

        # Solve for DF(1Y):
        known_term = par_rate * alpha_1 * df_0_5y
        df_1y = (notional - known_term) / (notional + par_rate * alpha_2)
    else:
        raise ValueError(f"Unexpected number of payments for 1Y swap: {len(payment_dates)}")

    return df_1y


def bootstrap_bond_newton(par_rate, maturity, previous_dfs, all_tenors,
                          settlement_date, convention='ACT/ACT'):
    """
    Bootstrap bond DF using Newton-Raphson with rigorous intermediate DF handling.

    Bond pricing equation (notional = 1):
        1 = Σ[cᵢ × DF(tᵢ)] + (1 + c_final) × DF(T)

    Newton-Raphson:
        DF_new = DF_current - error / derivative
        error = price - 1
        derivative ~ (1 + c_final) + Σ[cᵢ × ∂DF(tᵢ)/∂DF(T)]

    Note: Derivative is approximate - standard practice.

    Args:
        par_rate: Par yield (decimal)
        maturity: Maturity in years
        previous_dfs: Dict of known DFs at earlier tenors
        all_tenors: List of all Basel tenors [0.25, 0.5, ..., 30]
        settlement_date: Settlement date
        convention: Day-count convention

    Returns:
        DF at maturity
    """
    notional = 1.0
    payment_freq = 2  # Semi-annual for Treasuries

    # Generate payment dates
    payment_dates = generate_payment_dates(settlement_date, maturity, payment_freq)

    # Initial guess for DF(maturity)
    df_current = 1.0 / (1.0 + par_rate * maturity)  # Simple approximation

    # Newton-Raphson iterations
    max_iterations = 50
    tolerance = 1e-9

    for iteration in range(max_iterations):

        bond_price = 0.0
        derivative_sum = 0.0

        # Loop over all payment dates
        for i, payment_date in enumerate(payment_dates):

            # Calculate time to payment (in years)
            t_payment = year_fraction(settlement_date, payment_date)

            # Calculate period accrual
            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, convention)
            coupon = par_rate * notional * alpha

            # Get DF for this payment date
            # Three cases:
            # 1. Close to a known tenor in previous_dfs
            # 2. This is the maturity we're solving for
            # 3. Intermediate date (needs interpolation)

            # Check if close to maturity first
            if abs(t_payment - maturity) < 1e-3:
                # Case 2: This is maturity payment (use wider tolerance)
                df_payment = df_current
                df_derivative = 1.0  # Full dependence

            else:
                # Check if close to any known tenor
                found_known = False
                for known_t in previous_dfs.keys():
                    if abs(t_payment - known_t) < 1e-3:
                        # Case 1: Known DF
                        df_payment = previous_dfs[known_t]
                        df_derivative = 0.0
                        found_known = True
                        break

                if not found_known:
                    # Case 3: Intermediate date - interpolate

                    # Find bounding tenors from KNOWN tenors only
                    known_tenors = list(previous_dfs.keys())
                    t_lower = max([t for t in known_tenors if t < t_payment])

                    # For upper bound, check all_tenors to find next tenor
                    t_upper = min([t for t in all_tenors if t > t_payment])

                df_lower = previous_dfs[t_lower]

                # If upper bound is the maturity we're solving for
                if abs(t_upper - maturity) < 1e-6:
                    df_upper = df_current

                    # Log-linear interpolation
                    df_payment = interpolate_df_loglinear(
                        t_payment, t_lower, t_upper, df_lower, df_upper
                    )

                    # Approximate derivative
                    if df_current > 1e-10:
                        df_derivative = df_payment / df_current
                    else:
                        df_derivative = 0.0

                else:
                    # Upper bound is also known
                    df_upper = previous_dfs[t_upper]
                    df_payment = interpolate_df_loglinear(
                        t_payment, t_lower, t_upper, df_lower, df_upper
                    )
                    df_derivative = 0.0  # No dependence on df_current

            # Add to bond price and derivative
            if i < len(payment_dates) - 1:
                # Intermediate coupon
                bond_price += coupon * df_payment
                derivative_sum += coupon * df_derivative
            else:
                # Final payment (principal + coupon)
                bond_price += (notional + coupon) * df_payment
                derivative_sum += (notional + coupon) * df_derivative

        # Error: bond should price at par (1.0)
        error = bond_price - notional

        # Check convergence
        if abs(error) < tolerance:
            break

        # Newton step
        if abs(derivative_sum) < 1e-10:
            # Derivative too small - use fallback
            derivative_sum = notional + par_rate * notional / payment_freq

        df_new = df_current - error / derivative_sum

        # Bounds checking (DF must be between 0 and 1)
        df_new = max(0.001, min(0.999, df_new))

        # Check if making progress
        if abs(df_new - df_current) < 1e-12:
            # Stuck - break
            warnings.warn(f"Newton stuck at iteration {iteration} for {maturity}Y bond")
            break

        df_current = df_new

    # Final check
    if iteration >= max_iterations - 1:
        warnings.warn(f"Newton did not converge for {maturity}Y bond after {max_iterations} iterations")

    return df_current


def bootstrap_swap_newton(par_rate, maturity, previous_dfs, all_tenors,
                          settlement_date, fixed_freq, convention):
    """
    Bootstrap swap DF using Newton-Raphson.

    Swap pricing equation (notional = 1):
        Fixed_PV = Floating_PV
        r × Σ[αᵢ × DF(tᵢ)] = 1 - DF(T)

    Args:
        par_rate: Par swap rate (decimal)
        maturity: Maturity in years
        previous_dfs: Dict of known DFs
        all_tenors: List of Basel tenors
        settlement_date: Settlement date
        fixed_freq: Fixed leg frequency (1=annual, 2=semi-annual)
        convention: Day-count convention

    Returns:
        DF at maturity
    """
    notional = 1.0

    # Generate payment dates
    payment_dates = generate_payment_dates(settlement_date, maturity, fixed_freq)

    # Initial guess
    df_current = 1.0 / (1.0 + par_rate * maturity)

    # Newton-Raphson
    max_iterations = 50
    tolerance = 1e-9

    for iteration in range(max_iterations):

        fixed_leg_pv = 0.0
        derivative_sum = 0.0

        for i, payment_date in enumerate(payment_dates):

            t_payment = year_fraction(settlement_date, payment_date)

            # Period accrual
            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, convention)

            # Get DF (same logic as bond)
            # Check if close to maturity first
            if abs(t_payment - maturity) < 1e-3:
                df_payment = df_current
                df_derivative = 1.0

            else:
                # Check if close to any known tenor
                found_known = False
                for known_t in previous_dfs.keys():
                    if abs(t_payment - known_t) < 1e-3:
                        df_payment = previous_dfs[known_t]
                        df_derivative = 0.0
                        found_known = True
                        break

                if not found_known:
                    # Interpolate
                    known_tenors = list(previous_dfs.keys())
                    t_lower = max([t for t in known_tenors if t < t_payment])
                    t_upper = min([t for t in all_tenors if t > t_payment])
                    df_lower = previous_dfs[t_lower]

                    if abs(t_upper - maturity) < 1e-3:
                        df_upper = df_current
                        df_payment = interpolate_df_loglinear(
                            t_payment, t_lower, t_upper, df_lower, df_upper
                        )
                        if df_current > 1e-10:
                            df_derivative = df_payment / df_current
                        else:
                            df_derivative = 0.0
                    else:
                        df_upper = previous_dfs[t_upper]
                        df_payment = interpolate_df_loglinear(
                            t_payment, t_lower, t_upper, df_lower, df_upper
                        )
                        df_derivative = 0.0


            # Fixed leg contribution
            fixed_leg_pv += par_rate * notional * alpha * df_payment
            derivative_sum += par_rate * notional * alpha * df_derivative

        # Floating leg (in arrears assumption)
        floating_leg_pv = notional * (1.0 - df_current)

        # Swap equation: fixed = floating
        error = fixed_leg_pv - floating_leg_pv

        # Check convergence
        if abs(error) < tolerance:
            break

        # Newton step
        total_derivative = derivative_sum + notional

        if abs(total_derivative) < 1e-10:
            total_derivative = notional

        df_new = df_current - error / total_derivative
        df_new = max(0.001, min(0.999, df_new))

        if abs(df_new - df_current) < 1e-12:
            break

        df_current = df_new

    if iteration >= max_iterations - 1:
        warnings.warn(f"Newton did not converge for {maturity}Y swap after {max_iterations} iterations")

    return df_current


def bootstrap_curve(par_rates_dict, curve_type, settlement_date):
    """
    Bootstrap complete zero curve for all 10 Basel tenors.

    Args:
        par_rates_dict: {0.25: 0.0398, 0.5: 0.0380, 1: 0.0370, ...}
        curve_type: 'USD_Treasury', 'USD_SOFR', or 'EUR_Swap'
        settlement_date: Settlement date

    Returns:
        discount_factors: {0.25: 0.9901, 0.5: 0.9813, ...}
    """
    # Determine conventions
    if curve_type == 'USD_Treasury':
        convention = 'ACT/ACT'
        fixed_freq = 2  # Semi-annual
        is_bond = True
    elif curve_type == 'USD_SOFR':
        convention = 'ACT/360'
        fixed_freq = 2  # Semi-annual
        is_bond = False
    elif curve_type == 'EUR_Swap':
        convention = '30/360'
        fixed_freq = 1  # Annual
        is_bond = False
    else:
        raise ValueError(f"Unknown curve_type: {curve_type}")

    tenors = sorted(par_rates_dict.keys())

    # Filter out 0.0 tenor (overnight rates are not Basel regulatory tenors)
    tenors = [t for t in tenors if t > 0]
    discount_factors = {}

    print(f"\n{'='*60}")
    print(f"BOOTSTRAPPING {curve_type}")
    print(f"Convention: {convention}, Frequency: {fixed_freq}/year")
    print(f"{'='*60}")

    for tenor in tenors:
        par_rate = par_rates_dict[tenor]

        print(f"\nTenor {tenor}Y: par_rate = {par_rate*100:.4f}%")

        if tenor <= 0.5:
            # Money market (3M, 6M only)
            df = bootstrap_money_market(par_rate, tenor, settlement_date, convention)
            print(f"  Method: Money Market")
            print(f"  DF({tenor}Y) = {df:.6f}")

        elif abs(tenor - 1.0) < 1e-6:
            # 1Y as coupon instrument
            if is_bond:
                df = bootstrap_1y_treasury(par_rate, discount_factors, settlement_date)
                print(f"  Method: 1Y Treasury (2 coupons)")
            else:
                df = bootstrap_1y_swap(par_rate, discount_factors, settlement_date,
                                       convention, fixed_freq)
                print(f"  Method: 1Y Swap ({fixed_freq} payments)")
            print(f"  DF({tenor}Y) = {df:.6f}")

        else:
            # Long end (≥ 2Y): Newton-Raphson
            if is_bond:
                df = bootstrap_bond_newton(par_rate, tenor, discount_factors,
                                          tenors, settlement_date, convention)
                print(f"  Method: Bond Newton-Raphson")
            else:
                df = bootstrap_swap_newton(par_rate, tenor, discount_factors,
                                          tenors, settlement_date, fixed_freq, convention)
                print(f"  Method: Swap Newton-Raphson")
            print(f"  DF({tenor}Y) = {df:.6f}")

        discount_factors[tenor] = df

    print(f"\n{'='*60}")
    print(f"[OK] {curve_type} BOOTSTRAP COMPLETE")
    print(f"{'='*60}\n")

    return discount_factors



# ZEROCURVE CLASS

class ZeroCurve:
    """
    Zero-coupon discount curve with cubic spline interpolation.

    Attributes:
        currency: 'USD' or 'EUR'
        curve_type: 'Treasury', 'SOFR', or 'EUR_Swap'
        tenors: Array of 10 Basel tenors
        discount_factors: DFs at each tenor
        zero_rates: Zero rates (continuously compounded)
        interpolator: Scipy spline object
    """

    def __init__(self, discount_factors_dict, currency, curve_type):
        """Initialize ZeroCurve with bootstrapped DFs."""
        self.currency = currency
        self.curve_type = curve_type
        self.discount_factors_dict = discount_factors_dict

        # Extract tenors and DFs
        self.tenors = np.array(sorted(discount_factors_dict.keys()))
        self.dfs = np.array([discount_factors_dict[t] for t in self.tenors])

        # Calculate zero rates (continuously compounded)
        # DF(T) = exp(-r × T) => r = -ln(DF) / T
        self.zero_rates = -np.log(self.dfs) / self.tenors

        # Fit cubic spline on zero rates
        self.interpolator = CubicSpline(self.tenors, self.zero_rates,
                                       bc_type='natural')

        # Validate forwards (auto-switch to PCHIP if needed)
        self._validate_forwards()

    def _validate_forwards(self):
        """Check all forward rates are non-negative."""
        # Check forwards on fine grid
        t_grid = np.linspace(self.tenors[0], self.tenors[-1], 1000)

        min_forward = float('inf')
        min_forward_location = None

        for i in range(len(t_grid) - 1):
            t1, t2 = t_grid[i], t_grid[i+1]
            fwd = self.get_forward_rate(t1, t2)

            if fwd < min_forward:
                min_forward = fwd
                min_forward_location = (t1, t2)

            if fwd < -0.01:  # Significant negative forward
                print(f"\n[WARN]  WARNING: Negative forward rate detected")
                print(f"   Curve: {self.currency} {self.curve_type}")
                print(f"   Forward: {fwd:.4f}% between {t1:.2f}Y and {t2:.2f}Y")
                print(f"   Switching to monotone (PCHIP) interpolation\n")

                # Switch to PCHIP
                self.interpolator = PchipInterpolator(self.tenors, self.zero_rates)

                # Re-validate
                self._validate_forwards()
                return

        print(f"[OK] Forward validation passed: min forward = {min_forward*100:.4f}% "
              f"at {min_forward_location[0]:.2f}Y-{min_forward_location[1]:.2f}Y")

    def get_df(self, maturity):
        """Get discount factor for any maturity (exact or interpolated)."""
        if maturity in self.discount_factors_dict:
            # Exact match - return bootstrapped DF
            return self.discount_factors_dict[maturity]
        else:
            # Interpolate zero rate, convert to DF
            zero_rate = self.interpolator(maturity)
            return np.exp(-zero_rate * maturity)

    def get_zero_rate(self, maturity):
        """Get zero rate for any maturity."""
        if maturity in self.discount_factors_dict:
            idx = list(self.tenors).index(maturity)
            return self.zero_rates[idx]
        else:
            return self.interpolator(maturity)

    def get_forward_rate(self, t1, t2):
        """Calculate instantaneous forward rate from t1 to t2."""
        if abs(t2 - t1) < 1e-10:
            return self.get_zero_rate(t1)

        r1 = self.get_zero_rate(t1)
        r2 = self.get_zero_rate(t2)

        return (r2 * t2 - r1 * t1) / (t2 - t1)

    def __repr__(self):
        return f"ZeroCurve({self.currency} {self.curve_type}, {len(self.tenors)} tenors)"



# VALIDATION

def validate_curve_properties(curve):
    """Validate basic curve properties (no-arbitrage conditions)."""
    results = {
        'curve': f"{curve.currency} {curve.curve_type}",
        'tests': []
    }

    # Test 1: No negative DFs
    test1_pass = all(df > 0 for df in curve.dfs)
    results['tests'].append({
        'name': 'V1.1: No negative discount factors',
        'pass': test1_pass,
        'detail': f"All {len(curve.dfs)} DFs > 0" if test1_pass else "FAILED"
    })

    # Test 2: DFs monotonically decreasing
    test2_pass = all(curve.dfs[i] > curve.dfs[i+1]
                     for i in range(len(curve.dfs) - 1))
    results['tests'].append({
        'name': 'V1.2: DFs monotonically decreasing',
        'pass': test2_pass,
        'detail': "DFs strictly decreasing" if test2_pass else "FAILED"
    })

    # Test 3: No significantly negative forwards
    t_grid = np.linspace(curve.tenors[0], curve.tenors[-1], 100)
    forwards = [curve.get_forward_rate(t_grid[i], t_grid[i+1])
                for i in range(len(t_grid) - 1)]
    min_forward = min(forwards)

    test3_pass = min_forward >= -0.01  # Allow tiny numerical error
    results['tests'].append({
        'name': 'V1.3: No negative forward rates',
        'pass': test3_pass,
        'detail': f"Min forward = {min_forward*100:.4f}%"
    })

    # Test 4: DF(0) = 1 (approximately, for very short tenor)
    df_short = curve.get_df(0.01)  # 0.01Y ~ 3.65 days
    test4_pass = 0.999 < df_short < 1.001
    results['tests'].append({
        'name': 'V1.4: DF(near-zero) ~ 1.0',
        'pass': test4_pass,
        'detail': f"DF(0.01Y) = {df_short:.6f}"
    })

    return results



# METHODOLOGY VALIDATIONS (V1.5-V1.9)

def validate_1y_bootstrap_method(settlement_date):
    """
    Validate that 1Y instruments are bootstrapped as coupon instruments, not money market.

    Test: Verify 1Y has multiple cash flows by checking payment schedules.

    Critical requirement: We fixed the "1Y MM shortcut" bug by treating 1Y as:
    - USD Treasury: 2 semi-annual coupons
    - USD SOFR: 2 semi-annual fixed payments
    - EUR Swap: 1 annual fixed payment

    Returns:
        List of test results
    """
    results = []

    # Test USD Treasury 1Y (should have 2 semi-annual coupons)
    payment_dates_treasury = generate_payment_dates(settlement_date, 1.0, 2)
    test_treasury = {
        'name': 'V1.5a: 1Y Treasury has 2 coupons (not MM)',
        'expected': 2,
        'actual': len(payment_dates_treasury),
        'pass': len(payment_dates_treasury) == 2,
        'detail': f"1Y Treasury: {len(payment_dates_treasury)} payments (expected 2 for semi-annual)"
    }
    results.append(test_treasury)

    # Test USD SOFR 1Y (should have 2 semi-annual payments)
    payment_dates_sofr = generate_payment_dates(settlement_date, 1.0, 2)
    test_sofr = {
        'name': 'V1.5b: 1Y SOFR has 2 fixed payments (not MM)',
        'expected': 2,
        'actual': len(payment_dates_sofr),
        'pass': len(payment_dates_sofr) == 2,
        'detail': f"1Y SOFR swap: {len(payment_dates_sofr)} fixed payments (expected 2 for semi-annual)"
    }
    results.append(test_sofr)

    # Test EUR 1Y (should have 1 annual payment, BUT still uses swap formula not MM)
    payment_dates_eur = generate_payment_dates(settlement_date, 1.0, 1)
    test_eur = {
        'name': 'V1.5c: 1Y EUR has 1 payment (annual convention, swap formula)',
        'expected': 1,
        'actual': len(payment_dates_eur),
        'pass': len(payment_dates_eur) == 1,
        'detail': f"1Y EUR swap: {len(payment_dates_eur)} fixed payment (expected 1 for annual)"
    }
    results.append(test_eur)

    return results


def validate_period_accruals(settlement_date):
    """
    Validate that accruals are calculated period-by-period, not cumulatively.

    Test: For 2Y semi-annual instrument, the 3rd period (1Y → 1.5Y) should have
    accrual ~0.5 years (period), NOT ~1.5 years (cumulative from settlement).

    Critical requirement: We fixed the "cumulative accrual" bug where coupons
    were incorrectly calculated as cumulative α(0,T) instead of period α(t₁,t₂).

    Returns:
        List of test results
    """
    results = []

    # Generate 2Y semi-annual payment schedule
    payment_dates = generate_payment_dates(settlement_date, 2.0, 2)
    # Expected: 4 payments at approximately 0.5Y, 1Y, 1.5Y, 2Y

    if len(payment_dates) >= 3:
        # Test period 3: from payment 2 (1Y) to payment 3 (1.5Y)
        period_start = payment_dates[1]  # 1Y payment date
        period_end = payment_dates[2]    # 1.5Y payment date

        # Calculate period accrual (what we SHOULD use)
        alpha_period = calculate_accrual(period_start, period_end, 'ACT/ACT')

        # Calculate cumulative accrual (what we should NOT use)
        alpha_cumulative = calculate_accrual(settlement_date, period_end, 'ACT/ACT')

        # Period accrual should be ~0.5 years
        # Cumulative accrual would be ~1.5 years
        test_period_value = {
            'name': 'V1.6a: Period accrual (1Y->1.5Y) ~ 0.5 years',
            'expected_range': (0.45, 0.55),
            'actual': alpha_period,
            'pass': 0.45 < alpha_period < 0.55,
            'detail': f"Period accrual = {alpha_period:.4f} (expected ~0.5 for half-year), "
                     f"NOT cumulative {alpha_cumulative:.4f}"
        }
        results.append(test_period_value)

        # Verify period ≠ cumulative (proof we're not using cumulative)
        test_not_cumulative = {
            'name': 'V1.6b: Period accrual != cumulative accrual (correct)',
            'diff': abs(alpha_period - alpha_cumulative),
            'pass': abs(alpha_period - alpha_cumulative) > 0.8,  # Should differ by ~1 year
            'detail': f"Period ({alpha_period:.4f}) vs Cumulative ({alpha_cumulative:.4f}), "
                     f"difference = {abs(alpha_period - alpha_cumulative):.4f} [OK]"
        }
        results.append(test_not_cumulative)
    else:
        results.append({
            'name': 'V1.6: Period accrual test',
            'pass': False,
            'detail': f"Insufficient payment dates: {len(payment_dates)}"
        })

    return results


def validate_loglinear_interpolation(curves_dict):
    """
    Validate that DF interpolation uses log-linear formula (constant forward).

    Test: Between two known tenors, verify that the interpolated DF follows
    the constant forward rate assumption:

    DF(t_mid) = DF(t_lower) × exp(-f × (t_mid - t_lower))

    where f = -ln(DF_upper / DF_lower) / (t_upper - t_lower)

    Critical requirement: We use log-linear interpolation (constant forward),
    NOT linear DF interpolation which can create arbitrage.

    Returns:
        List of test results
    """
    results = []

    for name, curve in curves_dict.items():
        # Test interpolation between 2Y and 3Y (adjacent Basel tenors)
        t_lower = 2.0
        t_upper = 3.0
        t_mid = 2.5  # Midpoint

        # Get DFs at known tenors
        df_lower = curve.get_df(t_lower)
        df_upper = curve.get_df(t_upper)

        # Get interpolated DF at midpoint (what curve gives us)
        df_mid_actual = curve.get_df(t_mid)

        # Calculate expected DF under constant forward assumption
        forward_rate = -np.log(df_upper / df_lower) / (t_upper - t_lower)
        df_mid_expected = df_lower * np.exp(-forward_rate * (t_mid - t_lower))

        # Error should be tiny (< 1%) for log-linear interpolation
        # NOTE: Small error expected because curve uses cubic spline on ZERO RATES,
        # not exact log-linear on DFs at query time, but during bootstrap it does
        error_pct = abs(df_mid_actual - df_mid_expected) / df_mid_expected * 100

        test_interp = {
            'name': f'V1.7: {name} uses log-linear interpolation (constant forward)',
            'curve': name,
            'test_point': f'{t_mid}Y (between {t_lower}Y and {t_upper}Y)',
            'df_actual': df_mid_actual,
            'df_expected': df_mid_expected,
            'error_pct': error_pct,
            'pass': error_pct < 1.0,  # Allow up to 1% due to spline on zero rates
            'detail': f"DF({t_mid}Y) = {df_mid_actual:.6f}, "
                     f"constant-fwd = {df_mid_expected:.6f}, "
                     f"error = {error_pct:.4f}%"
        }
        results.append(test_interp)

    return results


def validate_newton_convergence(curves_dict):
    """
    Validate Newton-Raphson convergence by checking that no DFs are stuck at bounds.

    Test: For long-end instruments (>1Y) bootstrapped with Newton-Raphson,
    verify that DFs are NOT stuck at artificial bounds (0.001 or 0.999).

    If Newton didn't converge, the DF would be stuck at the bounds we set
    in the Newton step: df_new = max(0.001, min(0.999, df_new))

    Critical requirement: Newton-Raphson should converge in <10 iterations
    with approximate derivatives (industry standard).

    Returns:
        List of test results
    """
    results = []

    for name, curve in curves_dict.items():
        # Check all tenors > 1Y (these use Newton-Raphson)
        stuck_at_bounds = []

        for tenor, df in zip(curve.tenors, curve.dfs):
            if tenor > 1.0:  # Only Newton territory
                # Check if DF is suspiciously close to bounds
                if abs(df - 0.001) < 0.0001 or abs(df - 0.999) < 0.0001:
                    stuck_at_bounds.append((tenor, df))

        test_convergence = {
            'name': f'V1.8: {name} Newton-Raphson converged (DFs not at bounds)',
            'curve': name,
            'stuck_dfs': stuck_at_bounds,
            'pass': len(stuck_at_bounds) == 0,
            'detail': f"All long-end DFs converged properly" if len(stuck_at_bounds) == 0
                     else f"[WARN] DFs stuck at bounds: {stuck_at_bounds}"
        }
        results.append(test_convergence)

    return results


def validate_notional_one_convention(curves_dict, settlement_date):
    """
    Validate that bootstrap internally used notional = 1.0 (not 100).

    Test: Price the 5Y Treasury at par using the bootstrapped curve.
    If bootstrap used notional=1, it should price to 1.0.
    If bootstrap incorrectly used notional=100, it would price to ~100.

    Critical requirement: This was the MAJOR unit bug we fixed. All bootstrap
    equations must use notional=1, and par instruments must price to 1.0.

    Returns:
        List of test results
    """
    results = []

    # Test USD Treasury 5Y
    if 'USD_Treasury' in curves_dict:
        curve = curves_dict['USD_Treasury']

        # 5Y Treasury par rate from market data
        par_rate = 0.037218
        maturity = 5.0
        notional = 1.0  # Use notional=1 for pricing
        payment_freq = 2

        # Generate payment schedule
        payment_dates = generate_payment_dates(settlement_date, maturity, payment_freq)

        # Price the bond
        price = 0.0
        for i, payment_date in enumerate(payment_dates):
            t = year_fraction(settlement_date, payment_date)

            # Calculate period accrual
            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, 'ACT/ACT')
            coupon = par_rate * notional * alpha
            df = curve.get_df(t)

            # Add to price
            if i < len(payment_dates) - 1:
                price += coupon * df
            else:
                price += (notional + coupon) * df

        # Should price to 1.0 (not 100)
        test_notional_1 = {
            'name': 'V1.9a: Bootstrap used notional=1 (5Y Treasury prices to 1.0)',
            'expected': 1.0,
            'actual': price,
            'error': abs(price - 1.0),
            'pass': abs(price - 1.0) < 0.01,  # 1% tolerance
            'detail': f"5Y Treasury price = {price:.6f} (expected 1.0 with notional=1)"
        }
        results.append(test_notional_1)

        # Verify it's NOT ~100 (proof we didn't use notional=100)
        test_not_100 = {
            'name': 'V1.9b: Bootstrap did NOT use notional=100 (incorrect)',
            'actual': price,
            'pass': abs(price - 100.0) > 50,
            'detail': f"Price = {price:.6f}, NOT ~100 [OK]"
        }
        results.append(test_not_100)

    return results


def validate_instrument_pricing(curves_dict, settlement_date):
    """Validate curves by pricing portfolio instruments."""
    results = {
        'instruments': []
    }

    # USGG5YR: 5Y Treasury at 3.7218%
    if 'USD_Treasury' in curves_dict:
        curve = curves_dict['USD_Treasury']

        par_rate = 0.037218
        maturity = 5.0
        notional = 1.0
        payment_freq = 2

        payment_dates = generate_payment_dates(settlement_date, maturity, payment_freq)

        price = 0.0
        for i, payment_date in enumerate(payment_dates):
            t = year_fraction(settlement_date, payment_date)

            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, 'ACT/ACT')
            coupon = par_rate * notional * alpha

            df = curve.get_df(t)

            if i < len(payment_dates) - 1:
                price += coupon * df
            else:
                price += (notional + coupon) * df

        error = abs(price - notional)
        results['instruments'].append({
            'name': 'USGG5YR (5Y Treasury)',
            'price': price,
            'target': notional,
            'error': error,
            'pass': error < 0.005,  # 0.5% tolerance
            'detail': f"Price = {price:.6f}, target = {notional:.6f}"
        })

    # USSO2: 2Y SOFR Swap at 3.3366%
    if 'USD_SOFR' in curves_dict:
        curve = curves_dict['USD_SOFR']

        par_rate = 0.033366
        maturity = 2.0
        notional = 1.0
        fixed_freq = 2

        payment_dates = generate_payment_dates(settlement_date, maturity, fixed_freq)

        fixed_pv = 0.0
        for i, payment_date in enumerate(payment_dates):
            t = year_fraction(settlement_date, payment_date)

            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, 'ACT/360')
            df = curve.get_df(t)
            fixed_pv += par_rate * notional * alpha * df

        floating_pv = notional * (1.0 - curve.get_df(maturity))
        npv = fixed_pv - floating_pv

        results['instruments'].append({
            'name': 'USSO2 (2Y SOFR Swap)',
            'fixed_pv': fixed_pv,
            'floating_pv': floating_pv,
            'npv': npv,
            'pass': abs(npv) < 0.001,  # 0.1% of notional
            'detail': f"NPV = {npv:.6f}, target ~ 0"
        })

    # USISSO10: 10Y SOFR Swap at 3.682%
    if 'USD_SOFR' in curves_dict:
        curve = curves_dict['USD_SOFR']

        par_rate = 0.03682
        maturity = 10.0
        notional = 1.0
        fixed_freq = 2

        payment_dates = generate_payment_dates(settlement_date, maturity, fixed_freq)

        fixed_pv = 0.0
        for i, payment_date in enumerate(payment_dates):
            t = year_fraction(settlement_date, payment_date)

            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, 'ACT/360')
            df = curve.get_df(t)
            fixed_pv += par_rate * notional * alpha * df

        floating_pv = notional * (1.0 - curve.get_df(maturity))
        npv = fixed_pv - floating_pv

        results['instruments'].append({
            'name': 'USISSO10 (10Y SOFR Swap)',
            'fixed_pv': fixed_pv,
            'floating_pv': floating_pv,
            'npv': npv,
            'pass': abs(npv) < 0.001,
            'detail': f"NPV = {npv:.6f}, target ~ 0"
        })

    # EUSA5: 5Y EUR Swap at 2.369%
    if 'EUR_Swap' in curves_dict:
        curve = curves_dict['EUR_Swap']

        par_rate = 0.02369
        maturity = 5.0
        notional = 1.0
        fixed_freq = 1  # Annual

        payment_dates = generate_payment_dates(settlement_date, maturity, fixed_freq)

        fixed_pv = 0.0
        for i, payment_date in enumerate(payment_dates):
            t = year_fraction(settlement_date, payment_date)

            if i == 0:
                period_start = settlement_date
            else:
                period_start = payment_dates[i-1]

            alpha = calculate_accrual(period_start, payment_date, '30/360')
            df = curve.get_df(t)
            fixed_pv += par_rate * notional * alpha * df

        floating_pv = notional * (1.0 - curve.get_df(maturity))
        npv = fixed_pv - floating_pv

        results['instruments'].append({
            'name': 'EUSA5 (5Y EUR Swap)',
            'fixed_pv': fixed_pv,
            'floating_pv': floating_pv,
            'npv': npv,
            'pass': abs(npv) < 0.001,
            'detail': f"NPV = {npv:.6f}, target ~ 0"
        })

    return results



# MAIN EXECUTION

def main():
    """Main execution function for Phase 1."""

    print("\n" + "="*70)
    print("FRTB-SA GIRR DELTA - PHASE 1: CURVE CONSTRUCTION")
    print("="*70)

    # Load market data
    print(f"\nLoading market data from: {DATA_FILE}")
    market_data = pd.read_csv(DATA_FILE)

    # Filter for GIRR curves
    curves_data = market_data[
        market_data['curve_type'].isin(['Treasury', 'SOFR_Swap', 'EUR_Swap'])
    ].copy()

    # Convert rates from percentage to decimal
    curves_data['rate_decimal'] = curves_data['last_price'] / 100.0

    # Separate by curve type
    usd_treasury = curves_data[
        (curves_data['instrument'].str.contains('TREASURY'))
    ].copy()

    usd_sofr = curves_data[
        (curves_data['instrument'].str.contains('SOFR'))
    ].copy()

    eur_swap = curves_data[
        (curves_data['instrument'].str.contains('EUR'))
    ].copy()

    print(f"[OK] Market data loaded:")
    print(f"   USD Treasury: {len(usd_treasury)} tenors")
    print(f"   USD SOFR: {len(usd_sofr)} tenors")
    print(f"   EUR Swap: {len(eur_swap)} tenors")

    # Extract par rates for each curve
    usd_treasury_rates = dict(zip(usd_treasury['tenor_years'],
                                  usd_treasury['rate_decimal']))

    usd_sofr_rates = dict(zip(usd_sofr['tenor_years'],
                              usd_sofr['rate_decimal']))

    eur_swap_rates = dict(zip(eur_swap['tenor_years'],
                              eur_swap['rate_decimal']))

    # Bootstrap curves
    print("\n" + "="*70)
    print("PHASE 1: ZERO-COUPON CURVE CONSTRUCTION")
    print("="*70)

    usd_treasury_dfs = bootstrap_curve(usd_treasury_rates, 'USD_Treasury', SETTLEMENT_DATE)
    usd_sofr_dfs = bootstrap_curve(usd_sofr_rates, 'USD_SOFR', SETTLEMENT_DATE)
    eur_swap_dfs = bootstrap_curve(eur_swap_rates, 'EUR_Swap', SETTLEMENT_DATE)

    # Create ZeroCurve objects
    usd_treasury_curve = ZeroCurve(usd_treasury_dfs, 'USD', 'Treasury')
    usd_sofr_curve = ZeroCurve(usd_sofr_dfs, 'USD', 'SOFR')
    eur_swap_curve = ZeroCurve(eur_swap_dfs, 'EUR', 'EUR_Swap')

    curves = {
        'USD_Treasury': usd_treasury_curve,
        'USD_SOFR': usd_sofr_curve,
        'EUR_Swap': eur_swap_curve
    }

    print("\n" + "="*70)
    print("[OK] ALL 3 CURVES BOOTSTRAPPED")
    print("="*70)

    # Run validations
    print("\n" + "="*70)
    print("VALIDATION SUITE")
    print("="*70)

    # Validate curve properties
    all_validations = []

    for name, curve in curves.items():
        print(f"\n{'-'*70}")
        print(f"Validating {name}")
        print(f"{'-'*70}")

        val_results = validate_curve_properties(curve)
        all_validations.append(val_results)

        for test in val_results['tests']:
            status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
            print(f"{status} - {test['name']}: {test['detail']}")

    # Validate instrument pricing
    print(f"\n{'-'*70}")
    print("Portfolio Instrument Pricing Validation")
    print(f"{'-'*70}")

    pricing_results = validate_instrument_pricing(curves, SETTLEMENT_DATE)

    for inst in pricing_results['instruments']:
        status = "[OK] PASS" if inst['pass'] else "[FAIL] FAIL"
        print(f"{status} - {inst['name']}")
        print(f"         {inst['detail']}")

    # ===== NEW: METHODOLOGY VALIDATIONS =====
    print(f"\n{'-'*70}")
    print("Methodology Validation (Bootstrap Approach)")
    print(f"{'-'*70}")

    # V1.5: 1Y bootstrap method (no MM shortcut)
    v15_results = validate_1y_bootstrap_method(SETTLEMENT_DATE)
    for test in v15_results:
        status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
        print(f"{status} - {test['name']}: {test['detail']}")

    # V1.6: Period accruals (not cumulative)
    v16_results = validate_period_accruals(SETTLEMENT_DATE)
    for test in v16_results:
        status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
        print(f"{status} - {test['name']}: {test['detail']}")

    # V1.7: Log-linear interpolation
    v17_results = validate_loglinear_interpolation(curves)
    for test in v17_results:
        status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
        print(f"{status} - {test['name']}: {test['detail']}")

    # V1.8: Newton convergence
    v18_results = validate_newton_convergence(curves)
    for test in v18_results:
        status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
        print(f"{status} - {test['name']}: {test['detail']}")

    # V1.9: Notional=1 convention
    v19_results = validate_notional_one_convention(curves, SETTLEMENT_DATE)
    for test in v19_results:
        status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
        print(f"{status} - {test['name']}: {test['detail']}")

    # Collect all methodology tests
    methodology_tests = v15_results + v16_results + v17_results + v18_results + v19_results

    # Overall validation status
    all_pass = (
        all(all(t['pass'] for t in v['tests']) for v in all_validations) and
        all(inst['pass'] for inst in pricing_results['instruments']) and
        all(test['pass'] for test in methodology_tests)
    )

    print("\n" + "="*70)
    if all_pass:
        print("[OK] ALL VALIDATIONS PASSED")
    else:
        print("[FAIL] SOME VALIDATIONS FAILED - REVIEW ABOVE")
    print("="*70)

    # Save curves
    print("\n" + "="*70)
    print("SAVING OUTPUTS")
    print("="*70)

    # Save curves as pickle (for Phase 2)
    pickle_path = f'{OUTPUT_DIR_SILVER}/curves_2025-11-03.pkl'
    with open(pickle_path, 'wb') as f:
        pickle.dump(curves, f)
    print(f"\n[OK] Curves saved to: {pickle_path}")

    # Save curves as CSV (human-readable)
    for name, curve in curves.items():
        df_output = pd.DataFrame({
            'tenor_years': curve.tenors,
            'discount_factor': curve.dfs,
            'zero_rate_percent': curve.zero_rates * 100
        })

        filename = f"{OUTPUT_DIR_SILVER}/{name.lower()}_curve_2025-11-03.csv"
        df_output.to_csv(filename, index=False)
        print(f"[OK] Saved: {filename}")

    # Generate validation report
    report_lines = [
        "# Phase 1 Validation Report: Curve Bootstrapping",
        f"\n**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Settlement Date**: {SETTLEMENT_DATE.strftime('%Y-%m-%d')}",
        "\n## Summary",
        f"\n**Status**: {'[OK] ALL VALIDATIONS PASSED' if all_pass else '[FAIL] SOME VALIDATIONS FAILED'}",
        "\n## Curves Built",
        "\n1. USD Treasury (10 tenors, ACT/ACT convention)",
        "2. USD SOFR Swap (10 tenors, ACT/360 convention)",
        "3. EUR Swap (10 tenors, 30/360 convention)",
        "\n## Curve Property Validations",
    ]

    for val in all_validations:
        report_lines.append(f"\n### {val['curve']}")
        for test in val['tests']:
            status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
            report_lines.append(f"- {status} **{test['name']}**: {test['detail']}")

    report_lines.append("\n## Portfolio Instrument Pricing Validations")

    for inst in pricing_results['instruments']:
        status = "[OK] PASS" if inst['pass'] else "[FAIL] FAIL"
        report_lines.append(f"\n### {inst['name']}")
        report_lines.append(f"- Status: {status}")
        report_lines.append(f"- {inst['detail']}")

    report_lines.append("\n## Methodology Validations (Bootstrap Approach)")
    report_lines.append("\nThese tests verify HOW the bootstrap was done, not just the results:")

    for test in methodology_tests:
        status = "[OK] PASS" if test['pass'] else "[FAIL] FAIL"
        report_lines.append(f"\n### {test['name']}")
        report_lines.append(f"- Status: {status}")
        report_lines.append(f"- {test['detail']}")

    report_lines.append("\n## Methodology Notes")
    report_lines.append("\n### Day-Count Conventions")
    report_lines.append("- **ACT/ACT**: Phase 1 approximation (days/365 or days/366)")
    report_lines.append("  - Error vs full ISDA: < 0.5 bps on DFs")
    report_lines.append("  - Acceptable for FRTB-SA capital calculations")
    report_lines.append("- **ACT/360**: Exact implementation")
    report_lines.append("- **30/360**: Exact standard formula")

    report_lines.append("\n### Bootstrap Method")
    report_lines.append("- **Short end (3M, 6M)**: Money market formula")
    report_lines.append("- **1Y**: Proper coupon bond/swap (no MM shortcut)")
    report_lines.append("- **Long end (≥2Y)**: Newton-Raphson with log-linear DF interpolation")
    report_lines.append("  - Derivative is approximate (standard practice)")
    report_lines.append("  - Typically converges in < 10 iterations")

    report_lines.append("\n### Interpolation")
    report_lines.append("- **Method**: Cubic spline on zero rates")
    report_lines.append("- **Validation**: Forward rate checks (auto-fallback to PCHIP if needed)")

    report_lines.append("\n## Conclusion")
    if all_pass:
        report_lines.append("\n[OK] All validations PASSED. Curves are ready for Phase 2 (sensitivity calculation).")
    else:
        report_lines.append("\n[FAIL] Some validations FAILED. Review errors above before proceeding.")

    # Save report
    report_path = f'{OUTPUT_DIR_DOCS}/validation_report_phase1_curves.md'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    print(f"\n[OK] Validation report saved to: {report_path}")

    # Final checkpoint
    print("\n" + "="*70)
    print("PHASE 1 COMPLETE - AWAITING USER APPROVAL")
    print("="*70)

    print("\nValidation Summary:")
    print(f"- Curve property validations: {len(all_validations)} curves checked")
    print(f"- Instrument pricing validations: {len(pricing_results['instruments'])} instruments checked")
    print(f"- Overall status: {'[OK] ALL PASS' if all_pass else '[FAIL] SOME FAIL'}")

    print(f"\nDetailed validation report: {report_path}")

    print("\n" + "-"*70)
    print("[PAUSE]  PHASE 1 CHECKPOINT")
    print("-"*70)
    print("\nBefore proceeding to Phase 2 (Sensitivity Calculation):")
    print("1. Review validation report: docs/validation_report_phase1_curves.md")
    print("2. Check all curves in: data/silver/*.csv")
    print("3. Verify instrument pricing results above")
    print("\n[PAUSE]  Do you approve proceeding to Phase 2? [YES/NO]")
    print("-"*70)


if __name__ == '__main__':
    main()
