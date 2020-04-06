from open_spiel.python.egt import dynamics as egt_dyn
import numpy as np
#NOTE: the learning rate is set outside of these dynamics

"""Cross-learning dynamics = replicator dynamics
    Based on equations (9) on page 12 in
    https://jair.org/index.php/jair/article/view/10952"""
def replicator(state, fitness):
    return egt_dyn.replicator(state, fitness)


"""Based on equations on p4 in
https://www.researchgate.net/publication/221454203_Frequency_adjusted_multiagent_Q-learning"""
#Boltzmann Q-learning is already defined in open_spiel.python.egt.dynamics
def boltzmann_qlearning(state, fitness, temperature=0.01):
    return egt_dyn.boltzmannq(state, fitness, temperature)

def boltzmann_faqlearning(state, fitness, temperature=0.001):
    exploitation = (1. / temperature) * replicator(state, fitness)

    # exploration_q = (np.log(state) - state.dot(np.log(state).transpose()))
    # explor = [state[1]*np.log(state[1]/state[0]), state[0]*np.log(state[0]/state[1])]
    explor_2 = np.flip(state) * np.log(np.flip(state)/state)        #only valid for 2x2 games
    return exploitation - state * explor_2

class MultiPopulationDynamics(egt_dyn.MultiPopulationDynamics):
    def __init__(self, payoff_tensor, dynamics):
        super(MultiPopulationDynamics, self).__init__(payoff_tensor, dynamics)

    def __call__(self, state, time=None):
        state = np.array(state)
        n = self.payoff_tensor.shape[0]     # number of players
        ks = self.payoff_tensor.shape[1:]   # number of strategies for each player
        assert state.shape[0] == sum(ks)

        states = np.split(state, np.cumsum(ks)[:-1])
        dstates = [None] * n
        for p in range(n):
            fitness = self.fitness_calc(n, p, ks, states)
            dstates[p] = self.dynamics[p](states[p], fitness)
        return np.concatenate(dstates)


    def fitness_calc(self, n, p, ks, states):
        """
        :param n: number of players
        :param p: player to calc fitness from
        :param ks: number of strategies for each player
        :param states: states from all n players
        :return: fitness of player p
        """
        # move i-th population to front
        fitness = np.moveaxis(self.payoff_tensor[p], p, 0)
        # marginalize out all other populations
        for p_ in set(range(n)) - {p}:
            fitness = np.tensordot(states[p_], fitness, axes=[0, 1])
        return fitness


#for a lenient agent/population the fitness function changes
#instead of having (for player P1) for each Action(P1): fitness(P1) = PayoffMatrix(P1) * ActionProb(P2)
#we have: equation (7) on page 6 of http://www.flowermountains.nl/pub/Bloembergen2010lfaq.pdf
# fitness(P1) = sum_j ( PayoffMatrix(P1)[j] * ActionProb(P2)[j] * [sum_k:Aik≤Aij(ActionProb(P2)[k])**k - sum_k:Aik<Aij(ActionProb(P2)[k])**k] / sum_k:Aik=Aij(ActionProb(P2)[k])
class LenientMultiPopulationDynamics(MultiPopulationDynamics):
    def __init__(self, payoff_tensor, dynamics, k=1):
        super(LenientMultiPopulationDynamics, self).__init__(payoff_tensor, dynamics)
        self.k = k

    def fitness_calc(self, n, p, ks, states):
        payoff = np.moveaxis(self.payoff_tensor[p], p, 0)
        fitness = np.zeros(shape=ks[p])
        # for each action
        for i in range(ks[p]):
            # for all other players
            p_ = abs(1 - p)  # FOR 2 PLAYERS ONLY
            u_i = 0
            # for each action each other player can take:
            for j in range(ks[p_]):
                strictworse_actions = 0     # k:Aik<Aij
                worse_actions = 0           # k:Aik≤Aij
                equal_actions = 0           # k:Aik==Aij
                #iterate over all actions the other player could have taken
                for k in range(ks[p_]):
                    if payoff[i][k] < payoff[i][j]:
                        strictworse_actions += (states[p_][k])
                        worse_actions += (states[p_][k])
                    elif payoff[i][k] == payoff[i][j]:
                        worse_actions += (states[p_][k])
                        equal_actions += (states[p_][k])
                ## u_i = sum_j (PayoffMatrix(P1)[j]        * ActionProb(P2)[j] * [sum_k:Aik≤Aij(ActionProb(P2)[k])**k - sum_k:Aik<Aij(ActionProb(P2)[k])**k] / sum_k:Aik=Aij(ActionProb(P2)[k])
                u_i += payoff[i][j] * states[p_][j] * ( worse_actions**self.k - strictworse_actions**self.k) / equal_actions
            fitness[i] = u_i
        return fitness