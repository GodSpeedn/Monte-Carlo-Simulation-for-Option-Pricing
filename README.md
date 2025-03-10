# Monte-Carlo-Simulation-for-Option-Pricing

A Python-based Monte Carlo simulation for pricing European call options using Geometric Brownian Motion (GBM). This project demonstrates computational finance techniques, including variance reduction with antithetic variates, statistical analysis, and validation against the Black-Scholes model.

## Features
- **GBM Simulation**: Simulates stock price paths using Geometric Brownian Motion.
- **Option Pricing**: Prices European call options with Monte Carlo methods.
- **Variance Reduction**: Implements antithetic variates to improve accuracy and efficiency.
- **Statistical Analysis**: Computes standard error and 95% confidence interval for price estimates.
- **Visualization**: Plots sample price paths with strike price reference.
- **Validation**: Compares results to the analytical Black-Scholes formula.

## Requirements
- Python 3.x
- Libraries: `numpy`, `matplotlib`, `scipy`  
  Install with: `pip install numpy matplotlib scipy`

## Usage
1. Clone the repo: `git clone <repo-url>`
2. Run the script: `python monte_carlo_pricer.py`

### Example Output
Monte Carlo Price: $8.05, Std Error: 0.0400, 95% CI: [7.97, 8.13], Time: 0.1234s
Antithetic Monte Carlo Price: $8.03, Time: 0.2345s
Black-Scholes Price: $8.02

A plot of 5 sample stock price paths will also display.

## Code Structure
- `MonteCarloPricer` Class:
  - `__init__`: Initializes parameters (e.g., stock price, strike, volatility).
  - `simulate_gbm`: Simulates stock price paths.
  - `simulate_gbm_antithetic`: Simulates with antithetic variates.
  - `price_option`: Computes option price with stats.
  - `plot_paths`: Visualizes sample paths.
- `black_scholes_call`: Analytical Black-Scholes pricing function.
- Main block: Executes simulations, prints results, and plots paths.

## Parameters
- `S0 = 100`: Initial stock price
- `K = 105`: Strike price
- `T = 1.0`: Time to maturity (years)
- `r = 0.05`: Risk-free rate (5%)
- `sigma = 0.2`: Volatility (20%)
- `n_simulations = 10000`: Number of simulated paths
- `n_steps = 252`: Time steps (trading days in a year)

## Results
The Monte Carlo price (~$8.05) and antithetic version (~$8.03) closely match the Black-Scholes price ($8.02), with a tight confidence interval demonstrating accuracy.

## Future Enhancements
- Vectorized GBM simulation for speed.
- Support for American options with early exercise.
- Parameter sensitivity analysis (e.g., price vs. volatility).

## Author
Narayan - 3rd-year CS student aiming for a quant graduate role. March/2025
