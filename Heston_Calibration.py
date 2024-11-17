
# Numerical Solution for Heston Model
# Monte Carlo Simulations

import os 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import plotly.graph_objects as go
import webbrowser


from scipy.integrate import quad
from scipy.optimize import minimize

import yfinance as yf
from nelson_siegel_svensson import NelsonSiegelSvenssonCurve
from nelson_siegel_svensson.calibrate import calibrate_ns_ols

S0 = 100
K = 100
v0 = 0.1
r = 0.03
kappa = 1.5768
theta = 0.0398
sigma = 0.3
lambd = 0.575
rho = -0.5711
tau = 1.0


# Heston Characteristic Funtion
def heston_characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r):
    
    a = kappa*theta
    b = kappa+lambd
    rspi = rho*sigma*phi*1j #1j is the complex number i
    
    d = np.sqrt((rho * sigma * phi * 1j - b) ** 2 + 
                (phi*1j + phi**2) * sigma ** 2)
    
    g = (b - rspi + d) / (b - rspi - d)
    
    # characteristic function by components
    exp1 = np.exp(r * phi * 1j * tau)
    term2 = (S0 ** (phi*1j) * ((1 - g*np.exp(d*tau)) / (1 - g)) ** 
             (-2 * a / sigma ** 2))
    exp2 = np.exp((a * tau / sigma ** 2) * (b - rspi + d) + 
                  (v0 / sigma**2) * (b - rspi + d) * 
                  ((1 - np.exp(d*tau)) / (1 - g*np.exp(d*tau))))
    
    return exp1 * term2 * exp2


def integrand(phi, K, *args):
    #print(K)
    numerator_term_1 = np.exp(r * tau) * heston_characteristic_function(phi - 1j, S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator_term_2 = K * heston_characteristic_function(phi, S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
    numerator = numerator_term_1 - numerator_term_2
    denominator = 1j * phi * K ** (1j * phi)

    return numerator / denominator

# C(K, S_0, nu_0, tau)
def heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    """Using Scipy's Integration Quad function"""
    #print(K)
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r) 
       
    integral, err = np.real(quad(lambda phi: integrand(phi, K, *args), 0, 100))
    # integral, err = np.real(quad(integrand, 0, 100, args=args))
    return (S0 - K*np.exp(-r*tau))/ 2 + integral/np.pi

def heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r):
    ''' This was created mainly due issues that I faced during the calibration step using scipy quad'''
    args = (S0, v0, kappa, theta, sigma, rho, lambd, tau, r)
   
    P, phimax, steps = 0, 100, 10000
    dphi = phimax / steps
    
    # rectangular integration
    for i in range(1, steps):
        #phi = dphi * (2*i + 1) / 2 # midpoint to calculate height
        #P += dphi * integrand(phi, K, *args)
        phi = dphi * (2*i + 1)/2 # midpoint to calculate height
        numerator = np.exp(r*tau)*heston_characteristic_function(phi-1j,*args) - K * heston_characteristic_function(phi,*args)
        denominator = 1j*phi*K**(1j*phi)

        P += dphi * numerator/denominator
        
    return np.real((S0 - K * np.exp(-r * tau)) / 2 + P/np.pi)
    
def calibrate_heston_model(ImpVolSurfaceLong, S0):
    
    r = ImpVolSurfaceLong['rate'].to_numpy(dtype=float)
    K = ImpVolSurfaceLong['strike'].to_numpy(dtype=float)
    tau = ImpVolSurfaceLong['maturity'].to_numpy(dtype=float)
    P = ImpVolSurfaceLong['price'].to_numpy(dtype=float)
    
    # x is Theta in our paper
    
    params = {
        "v0": {"x0": 0.3, "lbub": [1e-4, 1.0]},
        "kappa": {"x0": 1, "lbub": [0.1, 10]},
        "theta": {"x0": 0.05, "lbub": [1e-4, 0.8]},
        "sigma": {"x0": 0.3, "lbub": [0.1, 2]},
        "rho": {"x0": -0.8, "lbub": [-1, 0.5]},
        "lambd": {"x0": 0.05, "lbub": [-2, 2]}
        }
    
    x0 = [param["x0"] for param in params.values()]
    bounds = [param["lbub"] for param in params.values()]
    
    # Objective / Loss Function
    def SqErr(x):
        v0, kappa, theta, sigma, rho, lambd = x
        
        predicted_prices = np.array([
            heston_price_rec(S0, K_i, v0, kappa, theta, sigma, rho, lambd, tau_i, r_i)
            for K_i, tau_i, r_i in zip(K, tau, r)
        ])
        
        pen = np.sum( [(x_i-x0_i)**2 for x_i, x0_i in zip(x, x0)] ) # 0
        
        return np.sum((P - predicted_prices) ** 2/len(P)) + pen

    # Optimization using SLSQP
    result = minimize(SqErr, x0, tol=1e-3, method="SLSQP", bounds=bounds, options={"maxiter": 1000, "disp": True})

    return result

    
#print(heston_price(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r))

# Yield Rates Data.
# https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics?data=yield%27

yield_maturities = np.array([1/12, 2/12, 3/12, 4/12, 6/12, 1, 2, 3, 5, 7, 10, 20, 30])
yields = np.array([4.70, 4.68, 4.61, 4.53, 4.43, 4.36, 4.34, 4.30, 4.32, 4.38, 4.43, 4.69, 4.58]).astype(float)/100

# Nelson Siegel Svensson model using ordinary least squares approach
# Nelson Siegel Svensson calibration model
curve, status = calibrate_ns_ols(yield_maturities, yields)

## Collecting the Option Strike Prices
ticker = yf.Ticker("NVDA")
S0 = ticker.history(period="1d")['Close'].iloc[0]

market_prices = {}

# Collecting all available expiration dates
for expiration in ticker.options:
    
    option_chain = ticker.option_chain(expiration)
    
    calls = option_chain.calls
    
    market_prices[expiration] = {
        'strike': calls['strike'].tolist(),
        'price': ((calls['bid'] + calls['ask']) / 2).tolist()
    }
    
all_strikes = [v['strike'] for _, v in market_prices.items()]

# common strikes across all maturities
common_strikes = set.intersection(*map(set, all_strikes))
common_strikes = sorted(common_strikes)

prices = []
maturities = []

for date, v in market_prices.items():
    time_to_maturity = (dt.strptime(date, '%Y-%m-%d') - dt.today()).days / 362.25
    maturities.append(time_to_maturity)
    
    # Filtering prices for common strikes
    price = [v['price'][i] for i, x in enumerate(v['strike']) if x in common_strikes]
    prices.append(price)
    
price_arr = np.array(prices, dtype=object)

ImpVolSurface = pd.DataFrame(price_arr, index=maturities, columns=common_strikes)
ImpVolSurface = ImpVolSurface.loc[
    (ImpVolSurface.index > 0.04) & (ImpVolSurface.index < 1),
    (ImpVolSurface.columns > 99) & (ImpVolSurface.columns < 151)
]
# print(ImpVolSurface)

ImpVolSurfaceLong = ImpVolSurface.melt(ignore_index=False).reset_index() # long-format
ImpVolSurfaceLong.columns = ['maturity', 'strike', 'price']
ImpVolSurfaceLong['rate'] = ImpVolSurfaceLong['maturity'].apply(curve) # The risk-free rate for each maturity using the calibrated yield curve
# print(ImpVolSurfaceLong)

# Calibration step
result = calibrate_heston_model(ImpVolSurfaceLong, S0)
print(result)

# From the calibration results, we will use these parameters to estimate option prices
v0, kappa, theta, sigma, rho, lambd = [param for param in result.x]
print(v0, kappa, theta, sigma, rho, lambd)

r = ImpVolSurfaceLong['rate'].to_numpy(dtype=float)
K = ImpVolSurfaceLong['strike'].to_numpy(dtype=float)
tau = ImpVolSurfaceLong['maturity'].to_numpy(dtype=float)

# Heston price estimates
heston_prices = heston_price_rec(S0, K, v0, kappa, theta, sigma, rho, lambd, tau, r)

ImpVolSurfaceLong['heston_price'] = heston_prices



# Plot of the Market Prices to the Heston Model Prices
# Market Prices
fig = go.Figure(data=[
    go.Mesh3d(
        x=ImpVolSurfaceLong['maturity'], 
        y=ImpVolSurfaceLong['strike'], 
        z=ImpVolSurfaceLong['price'], 
        color='grey', 
        opacity=0.55
    )
])

# Calibrated Heston Prices
fig.add_scatter3d(
    x=ImpVolSurfaceLong['maturity'], 
    y=ImpVolSurfaceLong['strike'], 
    z=ImpVolSurfaceLong['heston_price'], 
    mode='markers', 
    marker=dict(size=3, color='black')
)

fig.update_layout(
    title_text='Comparison of Calibrated Heston Prices Scatter and Market Prices Mesh',
    scene=dict(
        xaxis_title='Maturity (Years)',
        yaxis_title='Strike (Pts)',
        zaxis_title='Option Price Imdex (Pts)'
    ),
    height=800,
    width=800
)

fig.write_html("market_vs_heston.html")
fig.show(renderer="browser")


# Plot 2: Colour map of the residuals


# Residuals between market prices and Heston prices
ImpVolSurfaceLong_sorted = ImpVolSurfaceLong.sort_values(by=['maturity', 'strike']).reset_index(drop=True)
maturities = ImpVolSurfaceLong_sorted['maturity'].values
strikes = ImpVolSurfaceLong_sorted['strike'].values
prices = ImpVolSurfaceLong_sorted['price'].values
residuals = np.abs(ImpVolSurfaceLong_sorted['price'] - ImpVolSurfaceLong_sorted['heston_price']).values

unique_maturities = np.unique(maturities)
unique_strikes = np.unique(strikes)

# The plot 
M, S = np.meshgrid(unique_maturities, unique_strikes, indexing='ij')
Z_prices = np.full(M.shape, np.nan)
Z_residuals = np.full(M.shape, np.nan)
for i in range(len(maturities)):
    maturity_index = np.where(unique_maturities == maturities[i])[0][0]
    strike_index = np.where(unique_strikes == strikes[i])[0][0]
    Z_prices[maturity_index, strike_index] = prices[i]
    Z_residuals[maturity_index, strike_index] = residuals[i]

fig = go.Figure(data=[
    go.Surface(
        x=M,
        y=S,
        z=Z_prices,
        surfacecolor=Z_residuals,
        colorscale='Viridis',
        colorbar=dict(title="Residuals"),
        cmin=np.nanmin(Z_residuals),
        cmax=np.nanmax(Z_residuals)
    )
])

fig.update_layout(
    title_text='Market Prices Colored by Residuals with Calibrated Heston Model',
    scene=dict(
        xaxis=dict(
            title='TIME (Years)',
            range=[unique_maturities.min(), unique_maturities.max()]
        ),
        yaxis=dict(
            title='STRIKES (Pts)',
            range=[unique_strikes.min(), unique_strikes.max()]
        ),
        zaxis=dict(
            title='INDEX OPTION PRICE (Pts)',
            range=[np.nanmin(Z_prices), np.nanmax(Z_prices)]
        )
    ),
    height=800,
    width=800
)


fig.write_html("market_vs_heston_colormap_surface.html")
fig.show(renderer="browser")