"""
Plant Design and Process Economics Formulas for Chemical Engineering
Comprehensive collection of classes for economic analysis and cost estimation
"""

import math


class TimeValueOfMoney:
    """
    Class for time value of money calculations
    """
    
    @staticmethod
    def future_value_single(P, i, n):
        """
        Calculate future value of a single present amount.
        
        Parameters:
        P: Present value ($)
        i: Interest rate per period (decimal, e.g., 0.10 for 10%)
        n: Number of periods
        
        Returns:
        Future value ($)
        """
        return P * ((1 + i) ** n)
    
    @staticmethod
    def present_value_single(F, i, n):
        """
        Calculate present value of a single future amount.
        
        Parameters:
        F: Future value ($)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Present value ($)
        """
        return F / ((1 + i) ** n)
    
    @staticmethod
    def future_value_annuity(A, i, n):
        """
        Calculate future value of an ordinary annuity.
        
        Parameters:
        A: Annuity payment per period ($)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Future value of annuity ($)
        """
        if i == 0:
            return A * n
        return A * (((1 + i) ** n - 1) / i)
    
    @staticmethod
    def present_value_annuity(A, i, n):
        """
        Calculate present value of an ordinary annuity.
        
        Parameters:
        A: Annuity payment per period ($)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Present value of annuity ($)
        """
        if i == 0:
            return A * n
        return A * ((1 - (1 + i) ** (-n)) / i)
    
    @staticmethod
    def annuity_from_present_value(P, i, n):
        """
        Calculate annuity payment from present value.
        
        Parameters:
        P: Present value ($)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Annuity payment per period ($)
        """
        if i == 0:
            return P / n
        return P * (i * (1 + i) ** n) / ((1 + i) ** n - 1)
    
    @staticmethod
    def annuity_from_future_value(F, i, n):
        """
        Calculate annuity payment from future value.
        
        Parameters:
        F: Future value ($)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Annuity payment per period ($)
        """
        if i == 0:
            return F / n
        return F * (i / ((1 + i) ** n - 1))
    
    @staticmethod
    def effective_interest_rate(i_nominal, m):
        """
        Calculate effective interest rate from nominal rate.
        
        Parameters:
        i_nominal: Nominal interest rate per year (decimal)
        m: Number of compounding periods per year
        
        Returns:
        Effective interest rate (decimal)
        """
        return (1 + i_nominal / m) ** m - 1
    
    @staticmethod
    def continuous_compounding(P, r, t):
        """
        Calculate future value with continuous compounding.
        
        Parameters:
        P: Present value ($)
        r: Continuous interest rate per year (decimal)
        t: Time in years
        
        Returns:
        Future value ($)
        """
        return P * math.exp(r * t)
    
    @staticmethod
    def present_value_gradient_arithmetic(A, G, i, n):
        """
        Calculate present value of arithmetic gradient series.
        
        Parameters:
        A: Base annuity payment ($)
        G: Gradient amount ($)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Present value ($)
        """
        if i == 0:
            return A * n + G * n * (n - 1) / 2
        
        P_A = TimeValueOfMoney.present_value_annuity(A, i, n)
        P_G = G / i * ((1 - (1 + i) ** (-n)) / i - n * (1 + i) ** (-n))
        return P_A + P_G
    
    @staticmethod
    def present_value_gradient_geometric(A, g, i, n):
        """
        Calculate present value of geometric gradient series.
        
        Parameters:
        A: Initial payment ($)
        g: Growth rate per period (decimal)
        i: Interest rate per period (decimal)
        n: Number of periods
        
        Returns:
        Present value ($)
        """
        if i == g:
            return A * n / (1 + i)
        return A * ((1 - ((1 + g) / (1 + i)) ** n) / (i - g))


class CostEstimation:
    """
    Class for cost estimation calculations
    """
    
    @staticmethod
    def equipment_cost_six_tenths_rule(C_base, Q_base, Q_new, n=0.6):
        """
        Calculate equipment cost using six-tenths rule (power law).
        
        Parameters:
        C_base: Base cost ($)
        Q_base: Base capacity
        Q_new: New capacity
        n: Scaling exponent, default 0.6 (six-tenths rule)
        
        Returns:
        Estimated cost ($)
        """
        if Q_base <= 0 or Q_new <= 0:
            raise ValueError("Capacities must be positive")
        return C_base * ((Q_new / Q_base) ** n)
    
    @staticmethod
    def equipment_cost_lang_factor(C_eq, f_lang):
        """
        Calculate total capital cost using Lang factor.
        
        Parameters:
        C_eq: Total purchased equipment cost ($)
        f_lang: Lang factor (typically 3.1 for solid processing, 4.7 for solid-fluid, 4.74 for fluid processing)
        
        Returns:
        Total capital cost ($)
        """
        return C_eq * f_lang
    
    @staticmethod
    def installation_cost(C_eq, f_install):
        """
        Calculate installation cost.
        
        Parameters:
        C_eq: Equipment cost ($)
        f_install: Installation factor (decimal, typically 0.3-1.0)
        
        Returns:
        Installation cost ($)
        """
        return C_eq * f_install
    
    @staticmethod
    def total_module_cost(C_eq, f_module):
        """
        Calculate total module cost.
        
        Parameters:
        C_eq: Bare module cost ($)
        f_module: Module factor (typically 1.15-1.25)
        
        Returns:
        Total module cost ($)
        """
        return C_eq * f_module
    
    @staticmethod
    def grass_roots_cost(C_tm, f_gr):
        """
        Calculate grass roots cost.
        
        Parameters:
        C_tm: Total module cost ($)
        f_gr: Grass roots factor (typically 1.3-1.5)
        
        Returns:
        Grass roots cost ($)
        """
        return C_tm * f_gr
    
    @staticmethod
    def operating_cost_material(flow_rate, price, operating_hours):
        """
        Calculate annual material operating cost.
        
        Parameters:
        flow_rate: Flow rate (kg/h or similar units)
        price: Price per unit ($/unit)
        operating_hours: Operating hours per year
        
        Returns:
        Annual material cost ($)
        """
        return flow_rate * price * operating_hours
    
    @staticmethod
    def operating_cost_labor(n_operators, wage_rate, hours_per_year=8760):
        """
        Calculate annual labor operating cost.
        
        Parameters:
        n_operators: Number of operators
        wage_rate: Wage rate per hour ($/h)
        hours_per_year: Operating hours per year, default 8760 (24/7 operation)
        
        Returns:
        Annual labor cost ($)
        """
        return n_operators * wage_rate * hours_per_year
    
    @staticmethod
    def operating_cost_utilities(utility_rate, consumption_rate, operating_hours):
        """
        Calculate annual utility operating cost.
        
        Parameters:
        utility_rate: Cost per unit of utility ($/unit)
        consumption_rate: Consumption rate (units/h)
        operating_hours: Operating hours per year
        
        Returns:
        Annual utility cost ($)
        """
        return utility_rate * consumption_rate * operating_hours
    
    @staticmethod
    def maintenance_cost(C_fci, f_maintenance=0.06):
        """
        Calculate annual maintenance cost.
        
        Parameters:
        C_fci: Fixed capital investment ($)
        f_maintenance: Maintenance factor (decimal, typically 0.03-0.10, default 0.06)
        
        Returns:
        Annual maintenance cost ($)
        """
        return C_fci * f_maintenance
    
    @staticmethod
    def depreciation_straight_line(C_initial, C_salvage, n):
        """
        Calculate annual straight-line depreciation.
        
        Parameters:
        C_initial: Initial cost ($)
        C_salvage: Salvage value ($)
        n: Useful life (years)
        
        Returns:
        Annual depreciation ($)
        """
        return (C_initial - C_salvage) / n


class EconomicAnalysis:
    """
    Class for economic analysis and profitability metrics
    """
    
    @staticmethod
    def net_present_value(cash_flows, discount_rate, initial_investment):
        """
        Calculate Net Present Value (NPV).
        
        Parameters:
        cash_flows: List of cash flows for each year ($) [year 1, year 2, ..., year n]
        discount_rate: Discount rate per period (decimal)
        initial_investment: Initial investment cost ($)
        
        Returns:
        NPV ($)
        """
        pv_cash_flows = sum(cf / ((1 + discount_rate) ** (t + 1)) 
                           for t, cf in enumerate(cash_flows))
        return pv_cash_flows - initial_investment
    
    @staticmethod
    def internal_rate_of_return(cash_flows, initial_investment, guess=0.1, max_iter=100, tol=1e-6):
        """
        Calculate Internal Rate of Return (IRR) using Newton-Raphson method.
        
        Parameters:
        cash_flows: List of cash flows for each year ($)
        initial_investment: Initial investment cost ($)
        guess: Initial guess for IRR (decimal)
        max_iter: Maximum iterations
        tol: Tolerance for convergence
        
        Returns:
        IRR (decimal)
        """
        def npv_function(r):
            return sum(cf / ((1 + r) ** (t + 1)) for t, cf in enumerate(cash_flows)) - initial_investment
        
        def npv_derivative(r):
            return sum(-cf * (t + 1) / ((1 + r) ** (t + 2)) for t, cf in enumerate(cash_flows))
        
        r = guess
        for _ in range(max_iter):
            f = npv_function(r)
            if abs(f) < tol:
                return r
            df = npv_derivative(r)
            if abs(df) < tol:
                raise ValueError("Derivative too small, cannot converge")
            r_new = r - f / df
            if abs(r_new - r) < tol:
                return r_new
            r = r_new
        
        return r
    
    @staticmethod
    def payback_period(cash_flows, initial_investment):
        """
        Calculate simple payback period.
        
        Parameters:
        cash_flows: List of annual cash flows ($)
        initial_investment: Initial investment cost ($)
        
        Returns:
        Payback period (years), returns float('inf') if never pays back
        """
        cumulative = 0
        for year, cf in enumerate(cash_flows, start=1):
            cumulative += cf
            if cumulative >= initial_investment:
                # Linear interpolation for partial year
                if year == 1:
                    return initial_investment / cf if cf > 0 else float('inf')
                remaining = initial_investment - sum(cash_flows[:year-1])
                return (year - 1) + remaining / cf if cf > 0 else float('inf')
        return float('inf')
    
    @staticmethod
    def discounted_payback_period(cash_flows, discount_rate, initial_investment):
        """
        Calculate discounted payback period.
        
        Parameters:
        cash_flows: List of annual cash flows ($)
        discount_rate: Discount rate per period (decimal)
        initial_investment: Initial investment cost ($)
        
        Returns:
        Discounted payback period (years)
        """
        cumulative_pv = 0
        for year, cf in enumerate(cash_flows, start=1):
            pv_cf = cf / ((1 + discount_rate) ** year)
            cumulative_pv += pv_cf
            if cumulative_pv >= initial_investment:
                if year == 1:
                    return initial_investment / pv_cf if pv_cf > 0 else float('inf')
                remaining = initial_investment - sum(cf / ((1 + discount_rate) ** (t + 1)) 
                                                   for t, cf in enumerate(cash_flows[:year-1]))
                return (year - 1) + remaining / pv_cf if pv_cf > 0 else float('inf')
        return float('inf')
    
    @staticmethod
    def profitability_index(pv_cash_flows, initial_investment):
        """
        Calculate profitability index (benefit-cost ratio).
        
        Parameters:
        pv_cash_flows: Present value of future cash flows ($)
        initial_investment: Initial investment cost ($)
        
        Returns:
        Profitability index (dimensionless)
        """
        if initial_investment == 0:
            return float('inf')
        return pv_cash_flows / initial_investment
    
    @staticmethod
    def annual_equivalent_cost(capital_cost, annual_op_cost, i, n, salvage_value=0):
        """
        Calculate annual equivalent cost (AEC).
        
        Parameters:
        capital_cost: Initial capital cost ($)
        annual_op_cost: Annual operating cost ($)
        i: Interest rate (decimal)
        n: Project life (years)
        salvage_value: Salvage value at end of life ($)
        
        Returns:
        Annual equivalent cost ($)
        """
        if i == 0:
            capital_recovery = capital_cost / n
        else:
            capital_recovery = capital_cost * (i * (1 + i) ** n) / ((1 + i) ** n - 1)
        
        if salvage_value > 0:
            if i == 0:
                salvage_factor = salvage_value / n
            else:
                salvage_factor = salvage_value * (i / ((1 + i) ** n - 1))
            capital_recovery -= salvage_factor
        
        return capital_recovery + annual_op_cost
    
    @staticmethod
    def equivalent_annual_annuity(npv, i, n):
        """
        Calculate equivalent annual annuity from NPV.
        
        Parameters:
        npv: Net present value ($)
        i: Interest rate (decimal)
        n: Project life (years)
        
        Returns:
        Equivalent annual annuity ($)
        """
        if i == 0:
            return npv / n
        return npv * (i * (1 + i) ** n) / ((1 + i) ** n - 1)


class Depreciation:
    """
    Class for depreciation calculations
    """
    
    @staticmethod
    def macrs_depreciation_rate(property_class, year):
        """
        Get MACRS depreciation rate for given property class and year.
        
        Parameters:
        property_class: MACRS property class (3, 5, 7, 10, 15, 20, 27.5, 39 years)
        year: Year of depreciation (1-indexed)
        
        Returns:
        Depreciation rate (decimal)
        """
        # MACRS depreciation rates (percentage as decimal)
        macrs_rates = {
            3: [0.3333, 0.4445, 0.1481, 0.0741],
            5: [0.2000, 0.3200, 0.1920, 0.1152, 0.1152, 0.0576],
            7: [0.1429, 0.2449, 0.1749, 0.1249, 0.0893, 0.0892, 0.0893, 0.0446],
            10: [0.1000, 0.1800, 0.1440, 0.1152, 0.0922, 0.0737, 0.0655, 0.0655, 0.0656, 0.0655, 0.0328],
            15: [0.0500, 0.0950, 0.0855, 0.0770, 0.0693, 0.0623, 0.0590, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0590, 0.0591, 0.0295],
            20: [0.0375, 0.0722, 0.0668, 0.0618, 0.0571, 0.0528, 0.0489, 0.0452, 0.0447, 0.0447, 0.0446, 0.0447, 0.0446, 0.0447, 0.0446, 0.0447, 0.0446, 0.0447, 0.0446, 0.0447, 0.0223]
        }
        
        if property_class not in macrs_rates:
            raise ValueError(f"Unsupported MACRS property class: {property_class}")
        
        rates = macrs_rates[property_class]
        if year < 1 or year > len(rates):
            return 0.0
        return rates[year - 1]
    
    @staticmethod
    def macrs_depreciation_amount(basis, property_class, year):
        """
        Calculate MACRS depreciation amount for a specific year.
        
        Parameters:
        basis: Cost basis of property ($)
        property_class: MACRS property class (years)
        year: Year of depreciation (1-indexed)
        
        Returns:
        Depreciation amount for year ($)
        """
        rate = Depreciation.macrs_depreciation_rate(property_class, year)
        return basis * rate
    
    @staticmethod
    def declining_balance(C_initial, C_salvage, n, factor=2.0, year=None):
        """
        Calculate declining balance depreciation.
        
        Parameters:
        C_initial: Initial cost ($)
        C_salvage: Salvage value ($)
        n: Useful life (years)
        factor: Depreciation factor (1.0 for single, 2.0 for double declining balance)
        year: Year number (if None, returns list for all years)
        
        Returns:
        Depreciation for specific year or list of annual depreciation amounts
        """
        book_values = []
        deprecations = []
        current_book = C_initial
        rate = factor / n
        
        for y in range(1, n + 1):
            dep = current_book * rate
            # Ensure book value doesn't go below salvage value
            if current_book - dep < C_salvage:
                dep = max(0, current_book - C_salvage)
            
            book_values.append(current_book - dep)
            deprecations.append(dep)
            current_book = current_book - dep
        
        if year is not None:
            if year < 1 or year > n:
                return 0.0
            return deprecations[year - 1]
        return deprecations
    
    @staticmethod
    def sum_of_years_digits(C_initial, C_salvage, n, year=None):
        """
        Calculate sum-of-years-digits depreciation.
        
        Parameters:
        C_initial: Initial cost ($)
        C_salvage: Salvage value ($)
        n: Useful life (years)
        year: Year number (if None, returns list for all years)
        
        Returns:
        Depreciation for specific year or list of annual depreciation amounts
        """
        sum_years = n * (n + 1) / 2
        deprecations = []
        
        for y in range(1, n + 1):
            dep = (C_initial - C_salvage) * (n - y + 1) / sum_years
            deprecations.append(dep)
        
        if year is not None:
            if year < 1 or year > n:
                return 0.0
            return deprecations[year - 1]
        return deprecations


class BreakEvenAnalysis:
    """
    Class for break-even analysis
    """
    
    @staticmethod
    def break_even_volume(fixed_cost, variable_cost_per_unit, selling_price_per_unit):
        """
        Calculate break-even volume.
        
        Parameters:
        fixed_cost: Total fixed costs ($/year)
        variable_cost_per_unit: Variable cost per unit ($/unit)
        selling_price_per_unit: Selling price per unit ($/unit)
        
        Returns:
        Break-even volume (units/year)
        """
        contribution_margin = selling_price_per_unit - variable_cost_per_unit
        if contribution_margin <= 0:
            raise ValueError("Selling price must exceed variable cost")
        return fixed_cost / contribution_margin
    
    @staticmethod
    def break_even_revenue(fixed_cost, variable_cost_per_unit, selling_price_per_unit):
        """
        Calculate break-even revenue.
        
        Parameters:
        fixed_cost: Total fixed costs ($/year)
        variable_cost_per_unit: Variable cost per unit ($/unit)
        selling_price_per_unit: Selling price per unit ($/unit)
        
        Returns:
        Break-even revenue ($/year)
        """
        q_be = BreakEvenAnalysis.break_even_volume(
            fixed_cost, variable_cost_per_unit, selling_price_per_unit
        )
        return q_be * selling_price_per_unit
    
    @staticmethod
    def contribution_margin(selling_price, variable_cost):
        """
        Calculate contribution margin per unit.
        
        Parameters:
        selling_price: Selling price per unit ($/unit)
        variable_cost: Variable cost per unit ($/unit)
        
        Returns:
        Contribution margin per unit ($/unit)
        """
        return selling_price - variable_cost
    
    @staticmethod
    def contribution_margin_ratio(selling_price, variable_cost):
        """
        Calculate contribution margin ratio.
        
        Parameters:
        selling_price: Selling price per unit ($/unit)
        variable_cost: Variable cost per unit ($/unit)
        
        Returns:
        Contribution margin ratio (decimal)
        """
        if selling_price == 0:
            return 0.0
        return (selling_price - variable_cost) / selling_price
    
    @staticmethod
    def margin_of_safety(actual_sales, break_even_sales):
        """
        Calculate margin of safety in units.
        
        Parameters:
        actual_sales: Actual sales volume (units)
        break_even_sales: Break-even sales volume (units)
        
        Returns:
        Margin of safety (units)
        """
        return actual_sales - break_even_sales
    
    @staticmethod
    def margin_of_safety_percentage(actual_sales, break_even_sales):
        """
        Calculate margin of safety as percentage.
        
        Parameters:
        actual_sales: Actual sales volume (units)
        break_even_sales: Break-even sales volume (units)
        
        Returns:
        Margin of safety percentage (decimal)
        """
        if actual_sales == 0:
            return 0.0
        return (actual_sales - break_even_sales) / actual_sales
    
    @staticmethod
    def operating_leverage(fixed_cost, variable_cost_total, revenue):
        """
        Calculate degree of operating leverage.
        
        Parameters:
        fixed_cost: Total fixed costs ($)
        variable_cost_total: Total variable costs ($)
        revenue: Total revenue ($)
        
        Returns:
        Degree of operating leverage (dimensionless)
        """
        contribution = revenue - variable_cost_total
        if contribution == 0:
            return float('inf')
        return contribution / (contribution - fixed_cost)


class CashFlowAnalysis:
    """
    Class for cash flow analysis
    """
    
    @staticmethod
    def operating_cash_flow(revenue, operating_cost, depreciation, tax_rate):
        """
        Calculate operating cash flow (after taxes).
        
        Parameters:
        revenue: Revenue ($)
        operating_cost: Operating cost ($)
        depreciation: Depreciation ($)
        tax_rate: Tax rate (decimal, e.g., 0.25 for 25%)
        
        Returns:
        Operating cash flow ($)
        """
        ebit = revenue - operating_cost - depreciation
        taxes = ebit * tax_rate if ebit > 0 else 0
        return ebit - taxes + depreciation
    
    @staticmethod
    def free_cash_flow(operating_cf, capital_expenditure, change_working_capital=0):
        """
        Calculate free cash flow.
        
        Parameters:
        operating_cf: Operating cash flow ($)
        capital_expenditure: Capital expenditure ($)
        change_working_capital: Change in working capital ($)
        
        Returns:
        Free cash flow ($)
        """
        return operating_cf - capital_expenditure - change_working_capital
    
    @staticmethod
    def working_capital(current_assets, current_liabilities):
        """
        Calculate working capital.
        
        Parameters:
        current_assets: Current assets ($)
        current_liabilities: Current liabilities ($)
        
        Returns:
        Working capital ($)
        """
        return current_assets - current_liabilities
    
    @staticmethod
    def working_capital_requirement(revenue, days_sales, days_payable, cost_fraction):
        """
        Estimate working capital requirement.
        
        Parameters:
        revenue: Annual revenue ($)
        days_sales: Days of sales outstanding
        days_payable: Days of payables outstanding
        cost_fraction: Fraction of revenue that is cost
        
        Returns:
        Working capital requirement ($)
        """
        accounts_receivable = revenue * (days_sales / 365)
        accounts_payable = revenue * cost_fraction * (days_payable / 365)
        inventory = revenue * cost_fraction * (60 / 365)  # Assuming 60 days inventory
        
        return accounts_receivable + inventory - accounts_payable
    
    @staticmethod
    def discounted_cash_flow(cash_flow, discount_rate, period):
        """
        Calculate discounted cash flow for a specific period.
        
        Parameters:
        cash_flow: Cash flow in period ($)
        discount_rate: Discount rate per period (decimal)
        period: Period number (1-indexed)
        
        Returns:
        Discounted cash flow ($)
        """
        return cash_flow / ((1 + discount_rate) ** period)
    
    @staticmethod
    def cumulative_cash_flow(cash_flows):
        """
        Calculate cumulative cash flow over time.
        
        Parameters:
        cash_flows: List of cash flows for each period ($)
        
        Returns:
        List of cumulative cash flows ($)
        """
        cumulative = []
        running_total = 0
        for cf in cash_flows:
            running_total += cf
            cumulative.append(running_total)
        return cumulative


class ProfitabilityMetrics:
    """
    Class for various profitability metrics
    """
    
    @staticmethod
    def return_on_investment(net_profit, investment):
        """
        Calculate Return on Investment (ROI).
        
        Parameters:
        net_profit: Net profit ($)
        investment: Investment amount ($)
        
        Returns:
        ROI (decimal)
        """
        if investment == 0:
            return float('inf') if net_profit > 0 else float('-inf')
        return net_profit / investment
    
    @staticmethod
    def return_on_assets(net_profit, total_assets):
        """
        Calculate Return on Assets (ROA).
        
        Parameters:
        net_profit: Net profit ($)
        total_assets: Total assets ($)
        
        Returns:
        ROA (decimal)
        """
        if total_assets == 0:
            return 0.0
        return net_profit / total_assets
    
    @staticmethod
    def return_on_equity(net_profit, equity):
        """
        Calculate Return on Equity (ROE).
        
        Parameters:
        net_profit: Net profit ($)
        equity: Shareholders' equity ($)
        
        Returns:
        ROE (decimal)
        """
        if equity == 0:
            return float('inf') if net_profit > 0 else float('-inf')
        return net_profit / equity
    
    @staticmethod
    def gross_profit_margin(revenue, cost_of_goods_sold):
        """
        Calculate gross profit margin.
        
        Parameters:
        revenue: Revenue ($)
        cost_of_goods_sold: Cost of goods sold ($)
        
        Returns:
        Gross profit margin (decimal)
        """
        if revenue == 0:
            return 0.0
        return (revenue - cost_of_goods_sold) / revenue
    
    @staticmethod
    def net_profit_margin(net_profit, revenue):
        """
        Calculate net profit margin.
        
        Parameters:
        net_profit: Net profit ($)
        revenue: Revenue ($)
        
        Returns:
        Net profit margin (decimal)
        """
        if revenue == 0:
            return 0.0
        return net_profit / revenue
    
    @staticmethod
    def earnings_before_interest_taxes(revenue, operating_expenses, depreciation):
        """
        Calculate Earnings Before Interest and Taxes (EBIT).
        
        Parameters:
        revenue: Revenue ($)
        operating_expenses: Operating expenses ($)
        depreciation: Depreciation ($)
        
        Returns:
        EBIT ($)
        """
        return revenue - operating_expenses - depreciation
    
    @staticmethod
    def net_income(ebit, interest_expense, tax_rate):
        """
        Calculate net income.
        
        Parameters:
        ebit: Earnings before interest and taxes ($)
        interest_expense: Interest expense ($)
        tax_rate: Tax rate (decimal)
        
        Returns:
        Net income ($)
        """
        ebt = ebit - interest_expense
        return ebt * (1 - tax_rate) if ebt > 0 else ebt


class SensitivityAnalysis:
    """
    Class for sensitivity analysis calculations
    """
    
    @staticmethod
    def sensitivity_coefficient(npv_base, npv_varied, parameter_change):
        """
        Calculate sensitivity coefficient.
        
        Parameters:
        npv_base: Base case NPV ($)
        npv_varied: NPV with varied parameter ($)
        parameter_change: Change in parameter (decimal, e.g., 0.1 for 10% increase)
        
        Returns:
        Sensitivity coefficient (dimensionless)
        """
        if parameter_change == 0 or npv_base == 0:
            return 0.0
        return ((npv_varied - npv_base) / npv_base) / parameter_change
    
    @staticmethod
    def tornado_diagram_rank(sensitivity_coefficients):
        """
        Rank parameters by sensitivity (absolute value).
        
        Parameters:
        sensitivity_coefficients: Dictionary of parameter: sensitivity_coefficient
        
        Returns:
        Sorted list of tuples (parameter, coefficient) by absolute value
        """
        return sorted(sensitivity_coefficients.items(), 
                     key=lambda x: abs(x[1]), reverse=True)
    
    @staticmethod
    def scenario_analysis(base_npv, optimistic_npv, pessimistic_npv, prob_optimistic, prob_pessimistic):
        """
        Calculate expected NPV from scenario analysis.
        
        Parameters:
        base_npv: Base case NPV ($)
        optimistic_npv: Optimistic scenario NPV ($)
        pessimistic_npv: Pessimistic scenario NPV ($)
        prob_optimistic: Probability of optimistic scenario (decimal)
        prob_pessimistic: Probability of pessimistic scenario (decimal)
        
        Returns:
        Expected NPV ($)
        """
        prob_base = 1 - prob_optimistic - prob_pessimistic
        return (prob_base * base_npv + 
                prob_optimistic * optimistic_npv + 
                prob_pessimistic * pessimistic_npv)
