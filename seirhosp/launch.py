from . import model
import numpy as np
import networkx as nx

def prepare_from_dt(popsize, StoE,
                    beta_ItoH=None, beta_HtoC=None,
                    dtEtoI=None, dtItoH=None, dtItoD=None,
                    dtItoR=None, dtHtoC=None, dtHtoR=None, dtHtoD=None,
                    dtCtoD=None, dtCtoR=None,
                    initE=0, initI=1, initH=0, initC=0, initD=0,
                    ncontacts=None, p_global=0, ages=None,
                    comorbidity=None):
    """
    Prepare a `SEIRHosp` object based on state durations.

    Parameters
    ----------
    popsize : int
        Population size
    StoE : float
        Probability of transmission per unit time.
    ItoH : float
        Probability of _ever_ going to the hospital
    HtoC : float
        Probability of _ever_ going to intensive care from the hospital
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
    dtCtoR : float, optional
        Average time for a transition from intensiveCare to Recovered to occour.
    dtCtoD : float, optional
        Average time for a transition from intensiveCare to Deceased to occour.
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
    ncontacts : int, optional
        Average number of contacts per node in the contact graph.
    p_global : float, optional
        Probability that an individual makes contact with someoe outside their
        contact network. default = 0
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
    ItoD = 1 - np.exp(-1.0/dtItoD) if dtItoD is not None else None
    HtoC = 1 - np.exp(-1.0/dtHtoC) if dtHtoC is not None else None
    HtoR = 1 - np.exp(-1.0/dtHtoR) if dtHtoR is not None else None
    HtoD = 1 - np.exp(-1.0/dtHtoD) if dtHtoD is not None else None
    CtoR = 1 - np.exp(-1.0/dtCtoR) if dtHtoR is not None else None
    CtoD = 1 - np.exp(-1.0/dtCtoD) if dtHtoD is not None else None

    if beta_ItoH is not None:
        ItoH *= beta_ItoH
    if beta_HtoC is not None:
        HtoC *= beta_HtoC

    if ncontacts is not None:
        contacts = custom_exponenital_graph(popsize, ncontacts)
    else:
        contacts = None
    return model.SEIRHosp(popsize, StoE, EtoI, ItoR, ItoH, ItoD,
            HtoC, HtoR, HtoD, CtoR, CtoD, initE, initI, initH, initC, initD,
            contacts, p_global, ages, comorbidity)

def custom_exponenital_graph(popsize, ncontacts, scale=100):
    """
    Obtain a preferential attachment graph without minimum contacts.

    Algorithm by Ryan McGee in package `seirsplus`.
    """
    print("Generating contact graph...")
    graph = nx.barabasi_albert_graph(popsize, ncontacts)

    #Randomly delete nodes
    print("Excluding edges from graph...")
    for node in graph:
        neighbors = list(graph[node].keys())
        num_to_keep = int(min(np.random.exponential(scale=scale, size=1),
                              len(neighbors)))
        neighbors_to_keep = np.random.choice(neighbors,
                                             size=num_to_keep,
                                             replace=False)
        for neighbor in neighbors:
            if neighbor not in neighbors_to_keep:
                graph.remove_edge(node, neighbor)

        return graph


def export_csv(system, filename):
    np.savetxt(filename, system.tseries,
               header="t\t\t\t\tS\t\t\t\tE\t\t\t\tI\t\t\t\tH\t\t\t\tC\t\t\t\tR\t\t\t\tD",
               delimiter='\t')

