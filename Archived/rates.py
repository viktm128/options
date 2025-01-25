"""Describe stochastic rates under different rate models."""
import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt


class EmpiricalTerm:
    """Collect market data and build smooth yield and forward rates curve."""
    def __init__(self, file):
        """Construct empircal term structure from market data."""
        self.file = file
        self.time_max = None
        self.yield_curve = self.empirical_yield_curve() 
        self.forward_curve = self.forward_finite_differences() 


    def empirical_yield_curve(self):
        """Use a cubic spline model to build treasury spot rate curve."""
        sc = pd.read_csv(self.file)

        self.time_max = sc['Years'].iloc[len(sc['Years']) - 1]
        # Currently, adding overnight rate as "1 day zero coupon bond"
        return CubicSpline(sc['Years'], sc['Fitted Zero Coupon Yield'] / 100)

    def forward_finite_differences(self, delta = 0.01, n_steps = 100):
        """Using emprical yield curve, fit an instantaneous forward curve."""
        times = np.linspace(0, self.time_max, n_steps + 1)
        infc = np.empty(n_steps + 1)
        for i in range(n_steps + 1):
            if i == 0:
                infc[i] = -((np.log(self.yield_curve(times[i] + delta)) - np.log(self.yield_curve(times[i]))) / delta)
            elif i == n_steps:
                infc[i] = -((np.log(self.yield_curve(times[i])) - np.log(self.yield_curve(times[i] - delta))) / delta)
            else:
                infc[i] = -((np.log(self.yield_curve(times[i] + delta / 2)) - np.log(self.yield_curve(times[i] - delta / 2))) / delta)

        return CubicSpline(times, infc)

    def p(self, T):
        """Return p*(0, T)."""
        return self.yield_curve(T)

    def f(self, T):
        """Return f*(0, T)."""
        return self.forward_curve(T)


class HullWhiteRate():
    """Model rates which follow dr_t = (theta_t - ar_t)dt + sdW_t."""

    def __init__(self):
        """Instantiate unfitted discrete Vasicek model."""
        self.a = None
        self.sigma = None
        self.term = EmpiricalTerm('data/zero_coupon_annual_yields_11-22-2024.csv')

    def fit_params(self):
        """Choose values for a, theta, s by fiting bond options data."""

        # Fit an a value using historical data?
        # ltr_file = "data/daily-long-term-treasury-rates.csv"
        # self.a = pd.read_csv(ltr_file)['LT Real Average (10> Yrs)'].mean()
        # TODO: this is really the wrong parameter for a, it is no longer the long term rate under Hull-White
        # Professor suggested use this as a starting point
        self.a = 0.045


        # Back out sigma value from bond option data.
        # TODO: need caplet data for this 
        self.sigma = 0.008

        # Use current yield curve to derive theta_t forall t
    
    def g(self, T):
        """Helper function for Hull-White calibration."""
        return (self.sigma ** 2) / (2 * self.a ** 2) * (1 - np.exp(-self.a * T)) ** 2

    def gprime(self, T):
        """Helper derivative for Hull-White calibration."""
        return self.sigma ** 2 / self.a ** 2 * (1 - np.exp(-self.a * T)) * (self.a * np.exp(-self.a * T))

    def theta(self, T, delta = 0.01):
        """Combine model with empirical data."""
        return (self.term.forward_curve(T + delta) - self.term.forward_curve(T - delta)) / (2 * delta) + self.gprime(T) + self.a * (self.term.forward_curve(T) + self.g(T))




    def predict_path(self, r0, delta, N):
        """Return the rate over the next N time steps starting from r0."""
        norms = np.random.normal(0, np.sqrt(delta), size=N)
        rate_path = np.empty(N + 1)
        rate_path[0] = r0
        for j in range(1, N):
            rate_path[j + 1] = rate_path[j] + (self.theta(delta * j) - self.a * rate_path[j]) * delta + self.sigma * norms[j - 1]

        return rate_path

    def ZCB(self, T):
        """Price a zero coupon bond with maturity T given in years."""
        pass



class Vasicek():
    """Model rates which follow dr_t = (b - ar_t)dt + s dW_t."""

    def __init__(self):
        """Instantiate unfitted discrete Vasicek model."""
        self.a = None
        self.b = None
        self.r0 = None
        self.sigma = None

    def fit_params(self):
        """Choose values for a, b, s."""
        pass

    def predict_path(self, delta, N):
        """Return the rate over the next N time steps starting from r0."""
        pass

    def ZCB(self, T):
        """Price a zero coupon bond with maturity T given in years."""
        pass


class ConstantRate():
    """Model rates as a fixed parameter compounded daily."""

    def __init__(self):
        """Instantiate discrete unfitted constant rate model."""
        self.r = None

    def fit_params(self, r_in):
        """Pass value of r however it is generated."""
        self.r = r_in

    def predict_path(self, r0, delta, N):
        """Return the rate over the next N time steps starting from r0."""
        return np.ones(N) * (self.r / 365)

    def ZCB(self, T):
        """Price a zero coupon bond with maturity T given in years."""
        return np.power(1 + self.r / 365,  -365 * T)



if __name__ == "__main__":
    HW = HullWhiteRate()
    HW.fit_params()
    rate_path = HW.predict_path(0.04, 0.01, 1000)
    t = np.linspace(0, 10, 1001)

    fig, ax = plt.subplots()
    ax.plot(t, rate_path)
    plt.show()

