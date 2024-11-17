# Black Scholes Monte Carlo Simulation
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as impvol

# Simulation dependent parameters
S0 = 100.0
T = 1.0
r = 0.03
steps = 252 # trading of time steps 
sims = 1000

# Black Scholes dependent parameters
sigma = 0.5

def BS_model_sims(S0, r, sigma, T, steps, sims):
    """
    

    Parameters
    ----------
    S0 : float
        Initial Stock Price.
    mu : float
        Expected rate of return.
    sigma : float
        Volatility of the underlying process.
    T : float
        Time of simulation.
    steps : int
        Number of time steps.
    sims : int
        Number of simulations.

    Returns
    -------
    S : ndarray
        Asset price over time.

    """
    
    dt = T / steps
    
    # Brownian Motion of the price process
    Bt = np.random.normal(0, 1, (steps, sims))
    
    S = np.full(shape=(steps+1, sims), fill_value=S0)
    
    for i in range(1, steps+1):
        S[i] = S[i-1] * np.exp((r - (sigma**2/2))*dt + sigma * np.sqrt(dt)*Bt[i-1, :])
        
    return S
    
time = np.linspace(0, T, steps + 1)

S_t = BS_model_sims(S0, r, sigma, T, steps, sims)

plt.figure(figsize=(10,6))
plt.plot(time, S_t)
plt.title('Underlying Paths Under Black Scholes Model')
plt.xlabel('Time')
plt.ylabel(r'Price $S_T$')
plt.show()

plt.figure(figsize=(10,6))
sns.kdeplot(S_t[-1], color="blue")
plt.title(r'The Price Density Under Black Scholes Model')
plt.xlim([20, 180])
plt.xlabel(r'$S_t$')
plt.ylabel("Density")
plt.legend()
plt.show()



