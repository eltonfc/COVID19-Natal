import numpy as np
import networkx as nx


class SEIRHosp():
    """
    Enchanced SEIR model considering hospitalizations.

    Parameters
    ----------
    popsize : int
        Population size
    ncontacts : int
        Average number of contacts per node in the contact graph.
    StoE : float
        Probability of transmission per unit time.
    EtoI : float
        Probability of transition from Exposed to Infectious per unit time.
    ItoR : float
        Probability of transition from Infectious to Recovered per unit time.
    ItoH : float, optional
        Probability of transition from Infectious to Hospitalized per unit time.
    ItoD : float, optional
        Probability of transition from Infectious to Deceased per unit time.
    HtoC : float, optional
        Probability of transition from Hospitalized to intensiveCare per unit time.
    HtoR : float, optional
        Probability of transition from Hospitalized to Recovered per unit time.
    HtoD : float, optional
        Probability of transition from Hospitalized to Deceased per unit time.
    CtoR : float, optional
        Probability of transition from intensiveCare to Recovered per unit time.
    CtoD : float, optional
        Probability of transition from intensiveCare to Deceased per unit time.
    dtEtoI : float, optional
        Minimum time before a transition from Exposed to Infectious can occour.
    dtItoH : float, optional
        Minimum time before a transition from Infectous to Hospitalized can occour.
    dtItoR : float, optional
        Minimum time before a transition from Infectous to Recovered can occour.
    dtHtoC : float, optional
        Minimum time before a transition from Hospitalized to intensiveCare can occour.
    dtHtoR : float, optional
        Minimum time before a transition from Hospitalized to Recovered can occour.
    dtHtoD : float, optional
        Minimum time before a transition from Hospitalized to Deceased can occour.
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

    def __init__(self, popsize, ncontacts, StoE, EtoI, ItoR, ItoH=None, ItoD=None,
                 HtoC=None, HtoR=None, HtoD=None, CtoR=None, CtoD=None, dtEtoI=None,
                 dtItoH=None, dtItoR=None, dtHtoC=None, dtHtoR=None, dtHtoD=None,
                 initE=0, initI=1, initH=0, initC=0, initD=0, ages=None,
                 comorbidity=None):
        pass

    def initialize_population(self, initE, initI, initH, init_C, initD):
        pass

    def generate_graph(self):
        pass

    def generate_quarantine_graph(self):
        pass

