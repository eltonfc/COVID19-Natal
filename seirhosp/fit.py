import functools

import numpy as np
from scipy.stats import poisson
from scipy import optimize

from . import model

class SEIRHopsOptimizer():
    def __init__(self, StoE, period=30):
        self.StoE = StoE
        self.period = period

    def prepare_model(self, **kwargs):
        self.ModelClass = functools.partial(model.SEIRHosp, **kwargs)

    def load_data(self, datafile):
        self.data = np.loadtxt(datafile)
        self.times = self.data[:,0]
        self.infectious = self.data[:,1]

    def _discretize(self, series, column=1):
        """
        Obtain a discrete timeseries, with integer times and values.

        Parameters
        ----------
        data : ndarray
            The data to be discretized, time series in columns: column 0 = timestamp.
        column : int, optional
            Column from `data` to be discretized. (default=1)

        Returns
        -------
        discretized : array of ints
            One dimensional array with the discretized values in order.
        """
        discretized = np.rint(np.interp(self.times, series[:,0], series[:,column]))
        return discretized

    def negative_log_likelihood(self, StoE):
        """
        Calculate the negative log likelihood of given data for the parameters.

        Parameters
        ----------
        data : ndarray
            The data for which the likelihood will be calculated.
        StoE : float
            Pribability of transmission per unit time. Parameter to be fitted.

        Returns
        -------
        nloglike : float
            Sum of negative likelihoods.
        """
        StoE = StoE[0]
        print(f"Preparing StoE={StoE}")
        system = self.ModelClass(StoE=StoE)
        print(f"Running StoE={StoE}...")
        system.run(T=self.period, print_interval=1000)
        column = 3

        model_results = self._discretize(system.tseries, column)
        # TODO: take an average of n runs.
        nloglike = -np.sum(poisson.logpmf(k=self.infectious, mu=model_results))
        print(StoE, -nloglike, model_results)
        return nloglike

    def optimize_StoE(self):
        initial_simplex = np.clip(np.array([[self.StoE * 0.5], [self.StoE*0.75]]),
                0, 1.0)
        self.optimizer = optimize.minimize(self.negative_log_likelihood,
                self.StoE, method='Nelder-Mead')
                #options={'initial_simplex': initial_simplex,
                #    'disp':True})
        self.StoE_opt = self.optimizer.x
        return self.StoE_opt
