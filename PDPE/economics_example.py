"""
Example: Plant Design Process Economics Calculations
Demonstrates how to use various classes from processEconomics.py
"""

from processEconomics import (
    TimeValueOfMoney, CostEstimation, EconomicAnalysis, 
    Depreciation, BreakEvenAnalysis, CashFlowAnalysis, 
    ProfitabilityMetrics, SensitivityAnalysis
)

# ========== Example 1: Time Value of Money ==========
print("Example 1: Time Value of Money Calculations")
print("=" * 50)

# Calculate future value of a present investment
P = 100000  # Initial investment ($)
i = 0.08    # Interest rate (8% per year)
n = 10      # Number of years

FV = TimeValueOfMoney.future_value_single(P, i, n)
PV = TimeValueOfMoney.present_value_single(FV, i, n)

print(f"Initial investment: ${P:,.2f}")
print(f"Interest rate: {i*100}% per year")
print(f"Number of years: {n}")
print(f"Future value: ${FV:,.2f}")
print(f"Present value (verification): ${PV:,.2f}\n")

# Calculate annuity
A = 5000  # Annual payment ($)
FV_annuity = TimeValueOfMoney.future_value_annuity(A, i, n)
PV_annuity = TimeValueOfMoney.present_value_annuity(A, i, n)

print(f"Annual payment: ${A:,.2f}")
print(f"Future value of annuity: ${FV_annuity:,.2f}")
print(f"Present value of annuity: ${PV_annuity:,.2f}\n")

# ========== Example 2: Cost Estimation ==========
print("Example 2: Cost Estimation")
print("=" * 50)

# Six-tenths rule for equipment cost scaling
C_base = 500000  # Base equipment cost ($)
Q_base = 1000    # Base capacity (kg/h)
Q_new = 2000     # New capacity (kg/h)

C_new = CostEstimation.equipment_cost_six_tenths_rule(C_base, Q_base, Q_new)

print(f"Base equipment cost: ${C_base:,.2f}")
print(f"Base capacity: {Q_base} kg/h")
print(f"New capacity: {Q_new} kg/h")
print(f"Estimated new equipment cost (six-tenths rule): ${C_new:,.2f}\n")

# Lang factor for total capital cost
C_eq = 1000000  # Total purchased equipment cost ($)
f_lang = 4.74   # Lang factor for fluid processing

C_total = CostEstimation.equipment_cost_lang_factor(C_eq, f_lang)

print(f"Purchased equipment cost: ${C_eq:,.2f}")
print(f"Lang factor: {f_lang}")
print(f"Total capital cost: ${C_total:,.2f}\n")

# Operating cost calculation
utility_rate = 0.10      # $/kWh
consumption = 5000       # kW
hours = 8760             # hours/year (24/7 operation)

utility_cost = CostEstimation.operating_cost_utilities(utility_rate, consumption, hours)

print(f"Utility rate: ${utility_rate}/kWh")
print(f"Consumption: {consumption} kW")
print(f"Operating hours: {hours} h/year")
print(f"Annual utility cost: ${utility_cost:,.2f}\n")

# ========== Example 3: Economic Analysis ==========
print("Example 3: Economic Analysis - NPV, IRR, Payback")
print("=" * 50)

# Project cash flows
initial_investment = 5000000  # Initial investment ($)
cash_flows = [
    -500000,   # Year 1: Additional investment
    1000000,   # Year 2: First year revenue
    1500000,   # Year 3
    2000000,   # Year 4
    2000000,   # Year 5
    1800000,   # Year 6
    1500000,   # Year 7
    1200000,   # Year 8
    1000000,   # Year 9
    800000     # Year 10
]
discount_rate = 0.12  # 12% discount rate

# Calculate NPV
npv = EconomicAnalysis.net_present_value(cash_flows, discount_rate, initial_investment)

print(f"Initial investment: ${initial_investment:,.2f}")
print(f"Discount rate: {discount_rate*100}%")
print(f"Net Present Value (NPV): ${npv:,.2f}")
print(f"Project is {'profitable' if npv > 0 else 'not profitable'}\n")

# Calculate IRR
irr = EconomicAnalysis.internal_rate_of_return(cash_flows, initial_investment)

print(f"Internal Rate of Return (IRR): {irr*100:.2f}%")
print(f"IRR {'exceeds' if irr > discount_rate else 'is below'} discount rate of {discount_rate*100}%\n")

# Calculate payback period
payback = EconomicAnalysis.payback_period(cash_flows, initial_investment)

print(f"Simple payback period: {payback:.2f} years\n")

# Discounted payback period
discounted_payback = EconomicAnalysis.discounted_payback_period(
    cash_flows, discount_rate, initial_investment
)

print(f"Discounted payback period: {discounted_payback:.2f} years\n")

# ========== Example 4: Depreciation ==========
print("Example 4: Depreciation Calculations")
print("=" * 50)

# Straight-line depreciation
C_initial = 1000000  # Initial cost ($)
C_salvage = 100000   # Salvage value ($)
n_life = 10          # Useful life (years)

annual_dep = CostEstimation.depreciation_straight_line(C_initial, C_salvage, n_life)

print(f"Initial cost: ${C_initial:,.2f}")
print(f"Salvage value: ${C_salvage:,.2f}")
print(f"Useful life: {n_life} years")
print(f"Annual straight-line depreciation: ${annual_dep:,.2f}\n")

# MACRS depreciation
basis = 500000  # Cost basis ($)
property_class = 7  # 7-year property class
year = 3  # Third year

macrs_rate = Depreciation.macrs_depreciation_rate(property_class, year)
macrs_amount = Depreciation.macrs_depreciation_amount(basis, property_class, year)

print(f"Cost basis: ${basis:,.2f}")
print(f"Property class: {property_class} years")
print(f"Year: {year}")
print(f"MACRS depreciation rate: {macrs_rate*100:.2f}%")
print(f"MACRS depreciation amount: ${macrs_amount:,.2f}\n")

# Double declining balance
ddb_dep = Depreciation.declining_balance(
    C_initial, C_salvage, n_life, factor=2.0, year=2
)

print(f"Double declining balance depreciation (year 2): ${ddb_dep:,.2f}\n")

# ========== Example 5: Break-Even Analysis ==========
print("Example 5: Break-Even Analysis")
print("=" * 50)

fixed_cost = 1000000     # Total fixed costs ($/year)
variable_cost = 50       # Variable cost per unit ($/unit)
selling_price = 100      # Selling price per unit ($/unit)

be_volume = BreakEvenAnalysis.break_even_volume(fixed_cost, variable_cost, selling_price)
be_revenue = BreakEvenAnalysis.break_even_revenue(fixed_cost, variable_cost, selling_price)
contribution_margin = BreakEvenAnalysis.contribution_margin(selling_price, variable_cost)

print(f"Fixed costs: ${fixed_cost:,.2f}/year")
print(f"Variable cost per unit: ${variable_cost:.2f}")
print(f"Selling price per unit: ${selling_price:.2f}")
print(f"Contribution margin per unit: ${contribution_margin:.2f}")
print(f"Break-even volume: {be_volume:,.0f} units/year")
print(f"Break-even revenue: ${be_revenue:,.2f}/year\n")

# Margin of safety
actual_sales = 25000  # Actual sales volume (units/year)
mos = BreakEvenAnalysis.margin_of_safety(actual_sales, be_volume)
mos_pct = BreakEvenAnalysis.margin_of_safety_percentage(actual_sales, be_volume)

print(f"Actual sales: {actual_sales:,} units/year")
print(f"Margin of safety: {mos:,.0f} units ({mos_pct*100:.2f}%)\n")

# ========== Example 6: Cash Flow Analysis ==========
print("Example 6: Cash Flow Analysis")
print("=" * 50)

revenue = 5000000       # Annual revenue ($)
operating_cost = 3000000  # Operating cost ($)
depreciation = 400000   # Annual depreciation ($)
tax_rate = 0.25         # Tax rate (25%)

operating_cf = CashFlowAnalysis.operating_cash_flow(
    revenue, operating_cost, depreciation, tax_rate
)

print(f"Revenue: ${revenue:,.2f}")
print(f"Operating cost: ${operating_cost:,.2f}")
print(f"Depreciation: ${depreciation:,.2f}")
print(f"Tax rate: {tax_rate*100}%")
print(f"Operating cash flow: ${operating_cf:,.2f}\n")

# Free cash flow
capex = 200000  # Capital expenditure ($)
change_wc = 50000  # Change in working capital ($)

fcf = CashFlowAnalysis.free_cash_flow(operating_cf, capex, change_wc)

print(f"Capital expenditure: ${capex:,.2f}")
print(f"Change in working capital: ${change_wc:,.2f}")
print(f"Free cash flow: ${fcf:,.2f}\n")

# ========== Example 7: Profitability Metrics ==========
print("Example 7: Profitability Metrics")
print("=" * 50)

net_profit = 1200000  # Net profit ($)
investment = 5000000  # Total investment ($)
total_assets = 8000000  # Total assets ($)
equity = 3000000      # Shareholders' equity ($)

roi = ProfitabilityMetrics.return_on_investment(net_profit, investment)
roa = ProfitabilityMetrics.return_on_assets(net_profit, total_assets)
roe = ProfitabilityMetrics.return_on_equity(net_profit, equity)

print(f"Net profit: ${net_profit:,.2f}")
print(f"Total investment: ${investment:,.2f}")
print(f"Total assets: ${total_assets:,.2f}")
print(f"Shareholders' equity: ${equity:,.2f}")
print(f"Return on Investment (ROI): {roi*100:.2f}%")
print(f"Return on Assets (ROA): {roa*100:.2f}%")
print(f"Return on Equity (ROE): {roe*100:.2f}%\n")

# Profit margins
cost_of_goods = 2000000  # Cost of goods sold ($)

gross_margin = ProfitabilityMetrics.gross_profit_margin(revenue, cost_of_goods)
net_margin = ProfitabilityMetrics.net_profit_margin(net_profit, revenue)

print(f"Revenue: ${revenue:,.2f}")
print(f"Cost of goods sold: ${cost_of_goods:,.2f}")
print(f"Gross profit margin: {gross_margin*100:.2f}%")
print(f"Net profit margin: {net_margin*100:.2f}%\n")

# ========== Example 8: Sensitivity Analysis ==========
print("Example 8: Sensitivity Analysis")
print("=" * 50)

# Base case
base_cash_flows = [1000000, 1500000, 2000000, 2000000, 1800000]
base_npv = EconomicAnalysis.net_present_value(
    base_cash_flows, discount_rate, initial_investment
)

# Varied cases (10% increase in cash flows)
varied_cash_flows = [cf * 1.1 for cf in base_cash_flows]
varied_npv = EconomicAnalysis.net_present_value(
    varied_cash_flows, discount_rate, initial_investment
)

sensitivity = SensitivityAnalysis.sensitivity_coefficient(
    base_npv, varied_npv, 0.1
)

print(f"Base NPV: ${base_npv:,.2f}")
print(f"NPV with 10% increase in cash flows: ${varied_npv:,.2f}")
print(f"Sensitivity coefficient: {sensitivity:.2f}\n")

print("=" * 50)
print("All examples completed!")
