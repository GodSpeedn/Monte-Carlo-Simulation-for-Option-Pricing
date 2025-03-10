import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import time

# Monte Carlo Pricer Class
class MonteCarloPricer:
    def __init__(self, S0, K, T, r, sigma, n_simulations, n_steps):
        self.S0 = S0              # Initial stock price
        self.K = K                # Strike price
        self.T = T                # Time to maturity (years)
        self.r = r                # Risk-free rate
        self.sigma = sigma        # Volatility
        self.n_simulations = n_simulations  # Number of simulations
        self.n_steps = n_steps    # Number of time steps
        self.dt = T / n_steps     # Time step size

    def simulate_gbm(self):
        """Simulate stock price paths using Geometric Brownian Motion."""
        S = np.zeros((self.n_simulations, self.n_steps + 1))
        S[:, 0] = self.S0
        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        for t in range(1, self.n_steps + 1):
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt +
                                         self.sigma * np.sqrt(self.dt) * Z[:, t-1])
        return S

    def simulate_gbm_antithetic(self):
        """Simulate paths with antithetic variates for variance reduction."""
        S = np.zeros((self.n_simulations, self.n_steps + 1))
        S_anti = np.zeros((self.n_simulations, self.n_steps + 1))
        S[:, 0] = self.S0
        S_anti[:, 0] = self.S0
        Z = np.random.normal(0, 1, (self.n_simulations, self.n_steps))
        for t in range(1, self.n_steps + 1):
            S[:, t] = S[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt +
                                         self.sigma * np.sqrt(self.dt) * Z[:, t-1])
            S_anti[:, t] = S_anti[:, t-1] * np.exp((self.r - 0.5 * self.sigma**2) * self.dt -
                                                   self.sigma * np.sqrt(self.dt) * Z[:, t-1])
        ST = S[:, -1]
        ST_anti = S_anti[:, -1]
        payoffs = 0.5 * (np.maximum(ST - self.K, 0) + np.maximum(ST_anti - self.K, 0))
        return np.exp(-self.r * self.T) * np.mean(payoffs)

    def price_option(self):
        """Price the option with basic Monte Carlo and return stats."""
        paths = self.simulate_gbm()
        payoffs = np.maximum(paths[:, -1] - self.K, 0)
        discounted_payoffs = np.exp(-self.r * self.T) * payoffs
        price = np.mean(discounted_payoffs)
        std_error = np.std(discounted_payoffs) / np.sqrt(self.n_simulations)
        ci_lower = price - 1.96 * std_error  # 95% confidence interval
        ci_upper = price + 1.96 * std_error
        return price, std_error, (ci_lower, ci_upper)

    def plot_paths(self):
        """Plot sample price paths."""
        paths = self.simulate_gbm()
        plt.figure(figsize=(10, 6))
        time_axis = np.linspace(0, self.T, self.n_steps + 1)
        for i in range(min(5, self.n_simulations)):
            plt.plot(time_axis, paths[i], label=f'Path {i+1}', alpha=0.7)
        plt.axhline(y=self.K, color='k', linestyle='--', label=f'Strike = ${self.K}')
        plt.title('Simulated Stock Price Paths (GBM)', fontsize=14)
        plt.xlabel('Time (Years)', fontsize=12)
        plt.ylabel('Stock Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.show()

# Black-Scholes Analytical Solution
def black_scholes_call(S0, K, T, r, sigma):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# Main Execution
if __name__ == "__main__":
    # Parameters
    S0 = 100      # Initial stock price
    K = 105       # Strike price
    T = 1.0       # Time to maturity (1 year)
    r = 0.05      # Risk-free rate
    sigma = 0.2   # Volatility
    n_simulations = 10000  # Number of simulations
    n_steps = 252  # Number of time steps

    # Initialize pricer
    pricer = MonteCarloPricer(S0, K, T, r, sigma, n_simulations, n_steps)

    # Basic Monte Carlo
    start = time.time()
    price, std_err, ci = pricer.price_option()
    end = time.time()
    print(f"Monte Carlo Price: ${price:.2f}, Std Error: {std_err:.4f}, "
          f"95% CI: [{ci[0]:.2f}, {ci[1]:.2f}], Time: {end - start:.4f}s")

    # Antithetic Monte Carlo
    start = time.time()
    anti_price = pricer.simulate_gbm_antithetic()
    end = time.time()
    print(f"Antithetic Monte Carlo Price: ${anti_price:.2f}, Time: {end - start:.4f}s")

    # Black-Scholes
    bs_price = black_scholes_call(S0, K, T, r, sigma)
    print(f"Black-Scholes Price: ${bs_price:.2f}")

    # Plot paths
    pricer.plot_paths()