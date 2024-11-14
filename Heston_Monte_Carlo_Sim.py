import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from py_vollib_vectorized import vectorized_implied_volatility as impvol

# Parameters 
# simulation dependent
S0 = 100.0
T = 1.0
r = 0.03
steps = 252 # trading of time steps 
sims = 1000

# Heston dependent parameters
kappa = 2
theta = 0.20 ** 2
v0 = 0.25 ** 2
rho = - 0.65
sigma = 0.5

def heston_model_sim(S0, v0, rho, kappa, theta, sigma, T, steps, sims):
    """
    

    Parameters
    ----------
    S0 : float
        Initial Price.
    v0 : float
        Initial variance.
    rho : float
        Correlation between asset returns and variance.
    kappa : float
        rate of mean reversion in variance process.
    theta : float
        long-term mean of variance process.
    sigma : float
        volatility of the variance process.
    T : int
        Time of simulation.
    steps : int
        Number of time steps.
    sims : int
        Number of simulations.

    Returns
    -------
    S : arr
         Asset prices over time.
    v : arr
         Variance over time
    """
    
    dt = T / steps
    
    # Parameters for generating Brownian Motion Correlation
    mu = np.array([0, 0])
    cov = np.array([[1, rho],
                    [rho, 1]])
    
    # Brownian Motions of both the price and variance process
    Bt = np.random.multivariate_normal(mu, cov, (steps, sims))
    # Brownian Motion of the price process
    BtS = np.squeeze(Bt[:, :, 0])
    # Brownian Motion of the variance process
    Btv = np.squeeze(Bt[:, :, 1])
    
    S = np.full(shape=(steps+1, sims), fill_value=S0)
    v = np.full(shape=(steps+1, sims), fill_value=v0)
        
    '''
    plt.figure(figsize=(10,6))
    plt.plot(BtS[:, 0].cumsum(axis=0), label=r"$B_t^S$", linestyle='-')
    plt.plot(Btv[:, 0].cumsum(axis=0), label=r"$B_t^\nu$", linestyle='-')
    plt.title(rf"Correlated Brownian Motion, $\rho$={rho}")
    plt.xlabel("Time Steps")
    plt.ylabel("Brownian Motion")
    plt.legend()
    plt.show()
    '''
    
    for i in range(1, steps+1):
        S[i] = (S[i-1] + r*S[i-1]*dt + 
                               np.sqrt(v[i-1] * dt) * S[i-1] * BtS[i-1, :])
        v[i] = np.maximum(v[i-1] + kappa*(theta - v[i-1])*dt + 
                          sigma*np.sqrt(v[i-1]*dt)*Btv[i-1, :], 0)
         
    return S, v


#### Monte Carlo Simulation Plots

time = np.linspace(0, T, steps + 1)

S_t, v_t = heston_model_sim(S0, v0, rho, kappa, 
                            theta, sigma, T, steps, sims)

plt.figure(figsize=(10,6))
plt.plot(time, S_t)
plt.title('Underlying Paths Under Heston Model')
plt.xlabel('Time')
plt.ylabel(r'Price $S_T$')
plt.show()

plt.figure(figsize=(10,6))
plt.plot(time, v_t)
plt.title('Volatility Paths Under Heston Model')
plt.xlabel('Time')
plt.ylabel(r'Volatility $\nu_T$')
plt.show()


#### Density Plots

S1_t, v1_t = heston_model_sim(S0, v0, 0.95, kappa, 
                            theta, sigma, T, steps, sims)

S2_t, v2_t = heston_model_sim(S0, v0, -0.95, kappa, 
                            theta, sigma, T, steps, sims)

plt.figure(figsize=(10,6))
sns.kdeplot(S1_t[-1], color="blue", label=r"$\rho = 0.95$")
sns.kdeplot(S2_t[-1], color="red", label=r"$\rho = -0.95$")
plt.title(r'The Price Density Under Heston Model')
plt.xlim([20, 180])
plt.xlabel(r'$S_t$')
plt.ylabel("Density")
plt.legend()
plt.show()

#### Volatility Smile Plot

K = np.arange(40, 200, 2)
S_T = S_t[-1, :] # Maturity prices
C = np.array([np.mean(np.exp(-r*T)*np.maximum(S_T - k, 0)) for k in K]) # Call prices
P = np.array([np.mean(np.exp(-r*T)*np.maximum(k - S_T, 0)) for k in K]) # Put prices

call_impvol = impvol(C, S0, K, T, r, flag='c', q=0, return_as='numpy') 
put_impvol = impvol(P, S0, K, T, r, flag='p', q=0, return_as='numpy')

plt.figure(figsize=(10,6))
plt.plot(K, call_impvol, label='Calls')
plt.plot(K, put_impvol, label='Puts')
plt.title(r"Implied Volatility")
plt.ylabel("Implied Volatility")
plt.xlabel('Strike Price')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.legend()
plt.show()