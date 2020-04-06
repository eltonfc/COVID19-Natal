#!/usr/bin/env python

import numpy as np
from seirhosp import launch, model

system = launch.prepare_from_dt(popsize=100000, initI=10, ncontacts=0, StoE=0.25, beta_HtoC=0.2, beta_ItoH=0.2, dtEtoI=4, dtItoH=8, dtItoR=14, dtHtoC=3, dtHtoR=14, dtHtoD=28, dtCtoD=7, dtCtoR=14)

system.run(T=5, print_interval=100)

launch.export_csv(system, "SEIRS.dat")


