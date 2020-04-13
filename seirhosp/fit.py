import functools

import numpy as np
from scipy.stats import poisson
from scipy import optimize

from . import model

class SEIRHopsOptimizer():
    def __init__(self, StoE, period=30, nruns=1):
        self.StoE = StoE
        self.period = period
        self.nruns = nruns

    def prepare_model(self, **kwargs):
        self.ModelClass = functools.partial(model.SEIRHosp, **kwargs)
        testmodel = self.ModelClass(StoE=self.StoE)
        print("StoE:",testmodel.StoE[0],
                "EtoI:",  testmodel.EtoI[0],
                "ItoR:", testmodel.ItoR[0],
                "ItoH:",  testmodel.ItoH[0],
                "ItoD:",  testmodel.ItoD[0],
                "HtoR:",  testmodel.HtoR[0],
                "HtoC:",  testmodel.HtoC[0],
                "HtoD:",  testmodel.HtoD[0],
                "CtoR:",  testmodel.CtoR[0],
                "CtoD:",  testmodel.CtoD[0])

    def load_data(self, datafile):
        self.data = np.loadtxt(datafile)
        self.times = self.data[:,0]
        self.infectious = self.data[:,1]
        self.dead = self.data[:,2]

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
        # StoE = StoE[0]
        column = 7

        model_results = []

        for run in range(self.nruns):
            print(f"Preparing StoE={StoE}, run {run} of {self.nruns}")
            system = self.ModelClass(StoE=StoE)
            system.run(T=self.period, print_interval=1000)

            model_results.append(self._discretize(system.tseries, column))
            # TODO: take an average of n runs.

        model_results = np.rint(np.average(model_results, axis=0))
        model_results[model_results==0] = 1
        nloglike = -np.sum(poisson.logpmf(k=self.dead[17:], mu=model_results[17:]))
        print(StoE, -nloglike, model_results)
        return nloglike

    def optimize_StoE(self):
        initial_simplex = np.clip(np.array([[self.StoE * 0.75], [self.StoE*1.25]]),
                 0, 1.0)
        self.optimizer = optimize.minimize(self.negative_log_likelihood,
                self.StoE, method='Nelder-Mead',
                options={'initial_simplex': initial_simplex})

        self.StoE_opt = self.optimizer.x
        return self.StoE_opt
