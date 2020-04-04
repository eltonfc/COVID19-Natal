"""
Extended SEIR model taking into account hospitaliztion constraints.

Integration is done using the Gillepsie Algorithm.
"""
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
        Transition probability from Exposed to Infectious per unit time
    ItoR : float
        Transition probability from Infectious to Recovered per unit time
    ItoH : float, optional
        Transition probability from Infectious to Hospitalized per unit time
    ItoD : float, optional
        Transition probability from Infectious to Deceased per unit time
    HtoC : float, optional
        Transition probability from Hospitalized to intensiveCare per unit time
    HtoR : float, optional
        Transition probability from Hospitalized to Recovered per unit time
    HtoD : float, optional
        Transition probability from Hospitalized to Deceased per unit time
    CtoR : float, optional
        Transition probability from intensiveCare to Recovered per unit time
    CtoD : float, optional
        Transition probability from intensiveCare to Deceased per unit time
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

    def __init__(self, popsize, ncontacts, StoE, EtoI, ItoR, ItoH=None,
                 ItoD=None, HtoC=None, HtoR=None, HtoD=None, CtoR=None,
                 CtoD=None, initE=0, initI=1, initH=0, initC=0, initD=0,
                 ages=None, comorbidity=None):
        """ Initialize population state, time series and transition rates."""
        # State enumeration
        self.nstates = 7
        self.S = 0
        self.E = 1
        self.I = 2
        self.H = 3
        self.C = 4
        self.R = 5
        self.D = 6

        self.popsize = popsize
        # Initialize rates as column arrays with `popsize` lines to ease
        # propensity calculations
        self.StoE = np.full(shape=(popsize, 1), fill_value=StoE)
        self.EtoI = np.full(shape=(popsize, 1), fill_value=EtoI)
        self.ItoR = np.full(shape=(popsize, 1), fill_value=ItoR)
        self.ItoH = np.full(shape=(popsize, 1),
                            fill_value=ItoH if ItoH is not None else 0)
        self.ItoD = np.full(shape=(popsize, 1),
                            fill_value=ItoD if ItoD is not None else 0)
        self.HtoC = np.full(shape=(popsize, 1),
                            fill_value=HtoC if HtoC is not None else 0)
        self.HtoR = np.full(shape=(popsize, 1),
                            fill_value=HtoR if HtoR is not None else 0)
        self.HtoD = np.full(shape=(popsize, 1),
                            fill_value=HtoD if HtoD is not None else 0)
        self.CtoR = np.full(shape=(popsize, 1),
                            fill_value=CtoR if CtoR is not None else 0)
        self.CtoD = np.full(shape=(popsize, 1),
                            fill_value=CtoD if CtoD is not None else 0)

        # Initialize population
        self.initialize_population(initS, initE, initI, initH, init_C, initD)
        # TODO: initialize contact graphs

        self.initialize_time_series()

    def initialize_population(self, initE, initI, initH, init_C, initD):
        """
        Populate the population array with initial states.

        Optionally, the contact network graph is generated here.
        """
        initS = self.popsize - (initE + initI + initH + init_C + initD)
        self.pop = numpy.array([self.S] * initS + [self.E] * initE
                               + [self.I] * initI + [self.H] * initH
                               + [self.C] * initC
                               + [self.D] * initD).reshape(self.popsize, 1)
        # TODO: initialize age groups
        np.random.shuffle(self.pop)

    def initialize_time_series(self):
        """Initialize the tiome series arrays.

        The time series is stored as a 2-dimensional `ndarray` where each
        column stores a variable and each row is a timestep.

        For readability, there's an enumeration to point to the column indices.
        """
        # TODO: implement as Pandas dataframe
        # TODO: store data split by agegroup.

        # Column enumeration
        # NOTE: must match the State enumeration in `__init__`
        self.t_col = 0
        self.S_col = 1
        self.E_col = 2
        self.I_col = 3
        self.H_col = 4
        self.C_col = 5
        self.R_col = 6
        self.D_col = 7

        self.tseries = np.full(shape(self.popsize, self.nstates + 1),
                               fill_value=0)

        # Initialization values.
        self.t = 0      # Time in days, float
        self.t_idx = 0  # Current timestep index in self.tseries.
        self.t_max = None
        for state in range(self.numstates):
            self.tseries[self.t_idx, state + 1] = np.count_nonzero(
                np.pop == state)

    def extend_time_series(self):
        """
        Increase the time series array size.

        We do this periodically to avoid `insert`ing every iteration, which can
        be costly.
        """
        self.tseries = np.pad(array=tseries,
                              pad_width=((0, self.popsize), (0, 0)),
                              mode='constant', constant_values=0)

    def update_time_series(self):
        """
        Write current state counts to the timeseries. Extend if necessary.

        Instead of `pad`ding every iteration, we periodically `pad` the time
        series by `popsize` entries by calling `extend_time_series` to save
        time.
        """
        if(self.t_idx >= len(self.tseries) - 1):
            # The next iteration won't fit in the time series.
            self.extend_time_series()

        for state in range(self.numstates):
            self.tseries[self.t_idx, state + 1] = np.count_nonzero(
                np.pop == state)

    def calc_propensities(self):
        """
        Calculate the propensities for each transition for each individual.

        This is where we implement our differential equations.

        Returns
        -------
        propensities: array_like
            1-D Flattened propensities array.
        transitions: list of tuples
            Transitions in the order as they appear in `propensities`. Each one
            is a tuple in the form `(from, to)`, where `from` and `to` are
            state enums.
        """
        # TODO
        pass

    def execute_transition(self, transition_node, transition_type):
        """
        Execute the transition, taking constraints into account.

        This is done simply by flipping the state in the population array. If
        there are constraints, like limited number of hospital beds, we treat
        them here.
        """
        if(self.pop[transition_node] != transitions[transition_type][0]):
            raise RuntimeError(
                f"At step {self.t_idx}, node {transition_node} state is"
                + f" {self.pop[transition_node]} but it is scheduled for"
                + f" a {transitions[transition_type]} transition.")

        # TODO: Implement hospital and ICU beds constraints:
        # if np.count_nonzero(self.pop == self.C) > self.num_icu_beds...
        self.pop[transition_node] = transitions[transition_type][1]

    def run_iteration(self):
        """
        Execute the next reaction.

        This method executes one reaction in the Gillespie Algorithm.;

        Returns
        -------
            False if all propensities are 0, the system is static.
            True otherwise
        """
        propensities, transition_types = self.calc_propensities()

        if (propensities.sum() <= 0.0):
            # Terminate because no more transitions are possible.
            self.tseries = self.tseries[:self.t_idx + 1]
            return False

        # Now for the Gillespie Algorithm:
        r1, r2 = numpy.random.random(2)

        alpha = propensities.sum()      # The propensity of _any_ transition
        cumsum = propensities.cumsum()  # Transition roulette wheel!

        tau = (1 / alpha) * np.log(1.0 / r1)    # Time until the next iteration
        self.t += tau
        selt.t_idx += 1

        # Spin the roulette!
        transition_idx = np.searchsorted(cumsum, r2 * alpha)
        transition_node = transition_idx % self.pop
        transition_type = transition_types[int(transition_idx / self.pop)]

        self.execute_transition(transition_node, transition_type)

        self.update_time_series()

        return True

    def run(self, T, print_interval=10, verbose=False):
        """Run or extend the simulation for T days."""
        self.t_max = T if self.t_max is None else self.t_max + T
        running = True
        while running:
            running = self.run_iteration()
