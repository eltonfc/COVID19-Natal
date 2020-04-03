import model

def prepare_from_dt(popsize, ncontacts, StoE, dtEtoI=None, dtItoH=None,
                    dtItoR=None, dtHtoC=None, dtHtoR=None, dtHtoD=None,
                    initE=0, initI=1, initH=0, initC=0, initD=0, ages=None,
                    comorbidity=None):
    """
    Prepare a `SEIRHosp` object based on state durations.

    Parameters
    ----------
    popsize : int
        Population size
    ncontacts : int
        Average number of contacts per node in the contact graph.
    StoE : float
        Probability of transmission per unit time.
    dtEtoI : float, optional
        Average time for a transition from Exposed to Infectious to occour.
    dtItoH : float, optional
        Average time for a transition from Infectous to Hospitalized to occour.
    dtItoR : float, optional
        Average time for a transition from Infectous to Recovered to occour.
    dtItoD : float, optional
        Average time for a transition from Infectous to Deceased to occour.
    dtHtoC : float, optional
        Average time for a transition from Hospitalized to intensiveCare to occour.
    dtHtoR : float, optional
        Average time for a transition from Hospitalized to Recovered to occour.
    dtHtoD : float, optional
        Average time for a transition from Hospitalized to Deceased to occour.
    initE : int, optional
        Initial Exposed population
    initI : int, optional
        Initial Infecctious population
    initH : int, optional
        Initial Hospitalized population
    initC : int, optional
        Initial population in intensiveCare
    initR : int, optional
        Initial Recovered population
    ages : dict, optional
        Prevalence of each age group in the population.
        Format: { '0-5': x, '5-10': y ...} if x, y, ... are floats, they will
        be considered probabilities for each age bracket. If x, y, ... are
        ints, they are considered absolute values.
    comorbidity : dicts, optional
        Comorbidity rate per age group.. Must be in the same format as `ages`.
    """

    # The probability of
    EtoI = 1 - np.exp(-1.0/dtEtoI) if dtEtoI is not None else None
    ItoH = 1 - np.exp(-1.0/dtItoH) if dtItoH is not None else None
    ItoR = 1 - np.exp(-1.0/dtItoR) if dtItoR is not None else None
    HtoC = 1 - np.exp(-1.0/dtHtoC) if dtHtoC is not None else None
    HtoR = 1 - np.exp(-1.0/dtHtoR) if dtHtoR is not None else None
    HtoD = 1 - np.exp(-1.0/dtHtoD) if dtHtoD is not None else None

    return model.SEIRHosp(popsize, ncontacts, StoE, EtoI, ItoR, ItoH, ItoD,
            HtoC, HtoR, HtoD, CtoR, CtoD, initE, initI, initH, initC, initD,
            ages, comorbidity)
