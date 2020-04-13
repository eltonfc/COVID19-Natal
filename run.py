#!/usr/bin/env python

import numpy as np
from seirhosp import fit, model, launch

def invexp(x):
    return 1.0 - np.exp(-1.0/x)

def fit_beta(popsize=1000000, StoE=0.5187, dtEtoI=4,
             dtItoR=14, dtItoH=7, dtItoD=30,
             dtHtoC=14, dtHtoR=7, dtHtoD=28,
             dtCtoR=14, dtCtoD=14,
             initE=0, initI=1, initH=0, initC=0, initD=0,
             contacts=None, p_global=0, ages=None, comorbidity=None,
             algorithm='gillespie'):

    np.set_printoptions(linewidth=200)
    optimizer = fit.SEIRHopsOptimizer(StoE, period=30, nruns=3)
    optimizer.load_data('data/RN.dat')

    optimizer.prepare_model(popsize=popsize, EtoI=invexp(dtEtoI),
            ItoR=invexp(dtItoR), ItoH=invexp(dtItoH), ItoD=invexp(dtItoD),
            HtoR=invexp(dtHtoR), HtoC=invexp(dtHtoC), HtoD=invexp(dtHtoD),
            CtoR=invexp(dtCtoR), CtoD=invexp(dtCtoD))

    StoE_opt = optimizer.optimize_StoE()
    print(f"Final optimized StoE: {StoE_opt}")

    print("Initializing Final System:")
    for k in range(5):
        system = model.SEIRHosp(popsize=popsize, StoE=StoE_opt,
                EtoI=invexp(dtEtoI),
                ItoR=invexp(dtItoR), ItoH=invexp(dtItoH), ItoD=invexp(dtItoD),
                HtoR=invexp(dtHtoR), HtoC=invexp(dtHtoC), HtoD=invexp(dtHtoD),
                CtoR=invexp(dtCtoR), CtoD=invexp(dtCtoD))
        system.run(T=360)
        launch.export_csv(system, f'optimized_{k}.dat')

if __name__ == '__main__':
    fit_beta()

