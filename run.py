#!/usr/bin/env python

import numpy as np
from seirhosp import fit, model, launch


def fit_beta(popsize=10000, StoE=0.5, EtoI=0.2,
             ItoR=0.1, ItoH=0.0714, ItoD=0.01,
             HtoC=0.1428, HtoR=0.1, HtoD=0.05,
             CtoR=0.1428, CtoD=0.1,
             initE=0, initI=1, initH=0, initC=0, initD=0,
             contacts=None, p_global=0, ages=None, comorbidity=None,
             algorithm='gillespie'):

    optimizer = fit.SEIRHopsOptimizer(StoE, period=30)
    optimizer.load_data('data/RN.dat')

    optimizer.prepare_model(popsize=popsize, EtoI=EtoI, ItoR=ItoR, HtoC=HtoC, HtoR=HtoR,
                            HtoD=HtoD, CtoR=CtoR, CtoD=CtoD)

    StoE_opt = optimizer.optimize_StoE()
    print(f"Final optimized StoE: {StoE_opt}")

    print("Initializing Final System:")
    system = model.SEIRHosp(popsize=popsize, StoE=StoE_opt,
                            EtoI=EtoI, ItoR=ItoR, HtoC=HtoC, HtoR=HtoR,
                            HtoD=HtoD, CtoR=CtoR, CtoD=CtoD)
    system.run(T=360)

    launch.export_csv(system, 'optimized.dat')

if __name__ == '__main__':
    fit_beta()

