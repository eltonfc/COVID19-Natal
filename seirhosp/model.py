"""
Extended SEIR model taking into account hospitaliztion constraints.

Integration is done using the Gillepsie Algorithm.
"""
import numpy as np
import networkx as nx
from scipy import sparse


class SEIRHosp():
    """
    Enchanced SEIR model considering hospitalizations.

    Parameters
    ----------
    popsize : int
        Population size
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
    contacts : ndarray or networkx graph, optional
        Social contact graph/matrix. Must be a networkx.Graph object with
        `popsize` nodes or 2D matrix with `popsize` rows/columns.
    p_global : float, optional
        Probability that an individual makes contact with someone outside their
        contact network. default = 0
    ages : dict, optional
        Prevalence of each age group in the population.
        Format: { '0-5': x, '5-10': y ...} if x, y, ... are floats, they will
        be considered probabilities for each age bracket. If x, y, ... are
        ints, they are considered absolute values and must sum to `popsize`.
    comorbidity : dicts, optional
        Comorbidity rate per age group.. Must be in the same format as `ages`.
    algorithm : string, optional
        String to integrate the model equations:
            - `gillespie`: *Default* Use the Gillespie algorithm, a dynamic
              Monte Carlo algorithm that estimates the time interval `dt`
              between state transitions and ityerates per trasition.
            - `discrete`:  Iterate with timestep dt=1 (day) and attempt random
              transitions for all individuals, with probabilities as in the
              SEIR equations.
    """

    def __init__(self, popsize, StoE, EtoI,
                 ItoR, ItoH=None, ItoD=None,
                 HtoC=None, HtoR=None, HtoD=None,
                 CtoR=None, CtoD=None,
                 initE=0, initI=1, initH=0, initC=0, initD=0,
                 contacts=None, p_global=0, ages=None, comorbidity=None,
                 algorithm='gillespie'):
        """ Initialize population state, time series and transition rates."""
        # State enumeration, fixed here to avoid hashing later on.
        # This enumeration doubles as time series column enumeration.
        self.nstates = 7
        self.t_enum = 0
        self.S = 1
        self.E = 2
        self.I = 3
        self.H = 4
        self.C = 5
        self.R = 6
        self.D = 7

        self.popsize = popsize

        # Flag which state's population (or transition probability changed) in
        # each step to avoid calculating costly propensities
        self.changed_states = [ True for k in range(self.nstates + 1)]

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
        self.StoE = np.full(shape=self.popsize, fill_value=StoE)
        self.EtoI = np.full(shape=self.popsize, fill_value=EtoI)
        self.ItoR = np.full(shape=self.popsize, fill_value=ItoR)
        self.ItoH = np.full(shape=self.popsize,
                            fill_value=ItoH if ItoH is not None else 0)
        self.ItoD = np.full(shape=self.popsize,
                            fill_value=ItoD if ItoD is not None else 0)
        self.HtoC = np.full(shape=self.popsize,
                            fill_value=HtoC if HtoC is not None else 0)
        self.HtoR = np.full(shape=self.popsize,
                            fill_value=HtoR if HtoR is not None else 0)
        self.HtoD = np.full(shape=self.popsize,
                            fill_value=HtoD if HtoD is not None else 0)
        self.CtoR = np.full(shape=self.popsize,
                            fill_value=CtoR if CtoR is not None else 0)
        self.CtoD = np.full(shape=self.popsize,
                            fill_value=CtoD if CtoD is not None else 0)

        # Propensities are stored as an array with one line per transition with
        # `self.popsize` elements each.
        self.propensities = np.zeros((len(self.transitions), self.popsize))
        # Initialize population
        self.initialize_population(initE, initI, initH, initC, initD)

        if contacts is not None:
            self.update_adjacency_matrix(contacts)
            self.p_global = np.full(shape=self.popsize, fill_value=p_global)
        else:
            self.adjacency = None

        self.initialize_time_series()
        if algorithm == 'gillespie':
            self.run_iteration = self.run_gillespie_iteration
        elif algorithm == 'discrete':
            raise(NotImplementedError, "Discrete algorithm not yet implemented.")
        else:
            raise(ValueError, f"{algorithm} algorithm not recognized")

    def initialize_population(self, initE, initI, initH, initC, initD):
        """
        Populate the population array with initial states.

        zOptionally, the contact network graph is generated here.
        """
        initS = self.popsize - (initE + initI + initH + initC + initD)
        self.pop = np.array([self.S] * initS + [self.E] * initE
                            + [self.I] * initI + [self.H] * initH
                            + [self.C] * initC
                            + [self.D] * initD)
        # TODO: initialize age groups
        np.random.shuffle(self.pop)

    def update_adjacency_matrix(self, contacts):
        """
        Expose the ajacency matrix from the contact graph as a sparse matrix.

        Parameters
        ----------
        contacts : ndarray or networkx.Graph
            Contact graph as a `popsize` x `popsize` array or as a
            `networkx.Graph` object with `popsize`nodes.
        """
        raise(NotImplementedError, "Dealing with contact networks is currently "
              + "buggy. Will be fixed in next releases.")
        self.contacts = contacts
        if type(contacts) == np.ndarray:
            if contacts.shape != (self.popsize, self.popsize):
                raise(ValueError, "Parameter contacts must have shape "
                      + f"({self.popsize},{self.popsize}) if it\'s entered as "
                      + f"a numpy array. Got {contacts.shape} instead.")
            self.adjacency = sparse.csr_matrix(contacts)
        elif type(contacts) == nx.classes.graph.Graph:
            if contacts.number_of_nodes() != self.popsize:
                raise(ValueError, f"Parameter contacts must have {popsize} "
                      + f"nodes, got {contacts.number_of_nodes()} instead.")
            self.adjacency = nx.adj_matrix(contacts)
            # Number of contacts per individual. Much quicker to do this way,
            # as serisplus does, than contacts.degree()
            self.num_contacts = np.asarray(self.adjacency.sum(axis=0))[0]
        else:
            raise(TypeError, "Parameter contats bust be a 2d numpy array or a "
                  + " network Graph object.")

    def initialize_time_series(self):
        """Initialize the time series arrays.

        The time series is stored as a 2-dimensional `ndarray` where each
        column stores a variable and each row is a timestep.

        For readability, there's an enumeration to point to the column indices.
        """
        self.tseries = np.full(shape=(self.popsize, self.nstates + 1),
                               fill_value=0.0)
        # TODO: store data split by agegroup.
        # Initialization values.
        self.t = 0.0        # Time in days, float
        self.t_idx = 0      # Current timestep index in self.tseries.
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

        self.tseries[self.t_idx, self.t_enum] = self.t
        for state in range(1, self.nstates + 1):
            # States start from 1, so we adapt range().
            self.tseries[self.t_idx, state] = np.count_nonzero(
                self.pop == state)

    def calc_propensities(self):
        """
        Calculate the propensities for each transition for each individual.

        This is where we implement our differential equations.
        """

        if self.changed_states[self.S] or self.changed_states[self.E] or self.changed_states[self.I]:
            if self.adjacency is None:
                # Only global transmission
                    # propensities_StoE
                    self.propensities[0] = (self.StoE
                            * (self.tseries[self.t_idx][self.I] / self.popsize)
                            * (self.pop == self.S))
            else:
                # Taking the contact network into account
                # TODO: Fix wrong individuals being transitioned.
                StoE_contacts = np.asarray(sparse.csr_matrix.dot(
                    self.adjacency, self.pop == self.I), dtype='float64')
                self.propensities[0] = (self.p_global * self.StoE
                        * (self.tseries[self.t_idx][self.I] / self.popsize)
                        + (1 - self.p_global)
                        * np.divide((self.StoE * StoE_contacts),
                                     self.num_contacts,
                                     out=np.zeros_like(self.num_contacts,
                                                       dtype='float64'),
                                     where=self.num_contacts!=0)
                        * (self.pop == self.S))

        if self.changed_states[self.E] or self.changed_states[self.I]:
            # propensities_EtoI
            self.propensities[1] = self.EtoI * (self.pop == self.E)
        if self.changed_states[self.I] or self.changed_states[self.R]:
            # propensities_ItoR
            self.propensities[2] = self.ItoR * (self.pop == self.I)
        if self.changed_states[self.I] or self.changed_states[self.H]:
            self.propensities[3] = self.ItoH * (self.pop == self.I)
        if self.changed_states[self.I] or self.changed_states[self.D]:
            self.propensities[4] = self.ItoD * (self.pop == self.I)
        if self.changed_states[self.H] or self.changed_states[self.C]:
            self.propensities[5] = self.HtoC * (self.pop == self.H)
        if self.changed_states[self.H] or self.changed_states[self.R]:
            self.propensities[6] = self.HtoR * (self.pop == self.H)
        if self.changed_states[self.H] or self.changed_states[self.D]:
            self.propensities[7] = self.HtoD * (self.pop == self.H)
        if self.changed_states[self.C] or self.changed_states[self.R]:
            self.propensities[8] = self.CtoR * (self.pop == self.C)
        if self.changed_states[self.C] or self.changed_states[self.D]:
            self.propensities[9] = self.CtoD * (self.pop == self.C)

        # Flag which state's population (or transition probability changed) in
        # each step to avoid recalculating costly propensities
        self.changed_states = [ False for k in range(self.nstates + 1)]

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

    def run_gillespie_iteration(self):
        """
        Determine and execute the next reaction.

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
        selected_idx = np.searchsorted(cumsum, r2 * alpha)
        transition_node = selected_idx % self.popsize
        transition_type = self.transitions[int(selected_idx / self.popsize)]

        self.execute_transition(transition_node, transition_type)

        self.update_time_series()

        return True

    def run(self, T, print_interval=250, verbose=False):
        """Run or extend the simulation for T days."""
        self.t_max = T if self.t_max is None else self.t_max + T
        running = True
        while running and self.t < self.t_max:
            running = self.run_iteration()
            if (self.t_idx % print_interval == 0):
                print(f"{self.t_idx}: {self.tseries[self.t_idx]}")
        else:
            self.tseries = self.tseries[:self.t_idx + 1]
