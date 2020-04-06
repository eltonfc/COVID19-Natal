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

    def __init__(self, popsize, ncontacts, StoE, EtoI,
                 ItoR, ItoH=None, ItoD=None,
                 HtoC=None, HtoR=None, HtoD=None,
                 CtoR=None, CtoD=None,
                 initE=0, initI=1, initH=0, initC=0, initD=0,
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

        # Flag which state's population (or transition probability changed) in
        # each step to avoid calculating costly propensities
        self.changed_states = [ True for k in range(self.nstates)]

        self.transitions = [(self.S, self.E),
                       (self.E, self.I),
                       (self.I, self.R),
                       (self.I, self.H),
                       (self.I, self.C),
                       (self.H, self.C),
                       (self.H, self.R),
                       (self.H, self.D),
                       (self.C, self.R),
                       (self.C, self.D)]

        # Initialize rates as arrays with `popsize` elements to ease
        # propensity calculations
        # TODO: implement as 2d array, the same as propensities.
        # TODO: Transition enum
        self.StoE = np.full(shape=popsize, fill_value=StoE)
        self.EtoI = np.full(shape=popsize, fill_value=EtoI)
        self.ItoR = np.full(shape=popsize, fill_value=ItoR)
        self.ItoH = np.full(shape=popsize,
                            fill_value=ItoH if ItoH is not None else 0)
        self.ItoD = np.full(shape=popsize,
                            fill_value=ItoD if ItoD is not None else 0)
        self.HtoC = np.full(shape=popsize,
                            fill_value=HtoC if HtoC is not None else 0)
        self.HtoR = np.full(shape=popsize,
                            fill_value=HtoR if HtoR is not None else 0)
        self.HtoD = np.full(shape=popsize,
                            fill_value=HtoD if HtoD is not None else 0)
        self.CtoR = np.full(shape=popsize,
                            fill_value=CtoR if CtoR is not None else 0)
        self.CtoD = np.full(shape=popsize,
                            fill_value=CtoD if CtoD is not None else 0)

        # Propensities are stored as an array with one line per transition with
        # `self.popsize` elements each.
        self.propensities = np.zeros((len(self.transitions), self.popsize))
        # Initialize population
        self.initialize_population(initE, initI, initH, initC, initD)
        # TODO: initialize contact graphs

        self.initialize_time_series()

    def initialize_population(self, initE, initI, initH, initC, initD):
        """
        Populate the population array with initial states.

        Optionally, the contact network graph is generated here.
        """
        initS = self.popsize - (initE + initI + initH + initC + initD)
        self.pop = np.array([self.S] * initS + [self.E] * initE
                            + [self.I] * initI + [self.H] * initH
                            + [self.C] * initC
                            + [self.D] * initD)
        # TODO: initialize age groups
        np.random.shuffle(self.pop)

    def initialize_time_series(self):
        """Initialize the time series arrays.

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
        self.tseries = np.full(shape=(self.popsize, self.nstates + 1),
                               fill_value=0.0)
        # Initialization values.
        self.t = 0.0      # Time in days, float
        self.t_idx = 0  # Current timestep index in self.tseries.
        self.t_max = 0.0
        self.update_time_series()

    def extend_time_series(self):
        """
        Increase the time series array size.

        We do this periodically to avoid `insert`ing every iteration, which can
        be costly.
        """
        self.tseries = np.pad(array=self.tseries,
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

        self.tseries[self.t_idx, 0] = self.t
        for state in range(self.nstates):
            self.tseries[self.t_idx, state + 1] = np.count_nonzero(
                self.pop == state)

    def calc_propensities(self):
        """
        Calculate the propensities for each transition for each individual.

        This is where we implement our differential equations.
        """

        # TODO: make work with a network
        if self.changed_states[self.S] or self.changed_states[self.E]:
            # propensities_StoE
            self.propensities[0] = (self.StoE
                    * (self.tseries[self.t_idx][self.I_col] / self.popsize)
                    * (self.pop == self.S))
        if self.changed_states[self.E] or self.changed_states[self.I]:
            # propensities_EtoI
            self.propensities[1] = self.EtoI * (self.pop == self.E)
        if self.changed_states[self.I] or self.changed_states[self.R]:
            # propensities_ItoR
            self.propensities[2] = self.ItoR * (self.pop == self.I)
        if self.changed_states[self.I] or self.changed_states[self.H]:
            # propensities_ItoH
            self.propensities[3] = self.ItoH * (self.pop == self.I)
        if self.changed_states[self.I] or self.changed_states[self.D]:
            # propensities_ItoD
            self.propensities[4] = self.ItoD * (self.pop == self.I)
        if self.changed_states[self.H] or self.changed_states[self.C]:
            # propensities_HtoC
            self.propensities[5] = self.HtoC * (self.pop == self.H)
        if self.changed_states[self.H] or self.changed_states[self.R]:
            # propensities_HtoR
            self.propensities[6] = self.HtoR * (self.pop == self.H)
        if self.changed_states[self.H] or self.changed_states[self.D]:
            # propensities_HtoD
            self.propensities[7] = self.HtoD * (self.pop == self.H)
        if self.changed_states[self.C] or self.changed_states[self.R]:
            # propensities_CtoR
            self.propensities[8] = self.CtoR * (self.pop == self.C)
        if self.changed_states[self.C] or self.changed_states[self.D]:
            # propensities_CtoD
            self.propensities[9] = self.CtoD * (self.pop == self.C)

        # Flag which state's population (or transition probability changed) in
        # each step to avoid recalculating costly propensities
        self.changed_states = [ False for k in range(self.nstates)]

    def execute_transition(self, transition_node, transition_type):
        """
        Execute the transition, taking constraints into account.

        This is done simply by flipping the state in the population array. If
        there are constraints, like limited number of hospital beds, we treat
        them here.
        """
        if(self.pop[transition_node] != transition_type[0]):
            raise RuntimeError(
                f"At step {self.t_idx}, node {transition_node} state is"
                + f" {self.pop[transition_node]} but it is scheduled for"
                + f" a {transition_type} transition.")

        # TODO: Implement hospital and ICU beds constraints:
        # if np.count_nonzero(self.pop == self.C) > self.num_icu_beds...
        self.pop[transition_node] = transition_type[1]
        self.changed_states[transition_type[0]] = True
        self.changed_states[transition_type[1]] = True

    def run_iteration(self):
        """
        Execute the next reaction.

        This method executes one reaction in the Gillespie Algorithm.;

        Returns
        -------
            False if all propensities are 0, the system is static.
            True otherwise
        """
        self.calc_propensities()

        if (self.propensities.sum() <= 0.0):
            # Terminate because no more transitions are possible.
            self.tseries = self.tseries[:self.t_idx + 1]
            print(f"Null propensities. Terminating.")
            return False

        # Now for the Gillespie Algorithm:
        r1, r2 = np.random.random(2)

        cumsum = self.propensities.cumsum()  # Transition roulette wheel!
        alpha = cumsum[-1]
        # The propensity of _any_ transition is the sum of all propensities,
        # which is the last value of the cummulative sum.

        tau = (1.0 / alpha) * np.log(1.0 / r1)    # Time until the next iteration
        self.t += tau
        self.t_idx += 1

        # Spin the roulette!
        transition_idx = np.searchsorted(cumsum, r2 * alpha)
        transition_node = transition_idx % self.popsize
        transition_type = self.transitions[int(transition_idx / self.popsize)]

        self.execute_transition(transition_node, transition_type)

        self.update_time_series()

        return True

    def run(self, T, print_interval=1, verbose=False):
        """Run or extend the simulation for T days."""
        self.t_max = T if self.t_max is None else self.t_max + T
        running = True
        while running and self.t < self.t_max:
            running = self.run_iteration()
            if (self.t_idx % print_interval == 0):
                print(f"{self.t_idx}: {self.tseries[self.t_idx]}")
