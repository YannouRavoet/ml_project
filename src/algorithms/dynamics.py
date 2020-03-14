from open_spiel.python.egt.dynamics import replicator, boltzmannq, MultiPopulationDynamics
import numpy as np
#NOTE: the learning rate is set outside of these dynamics


"""Based on equations (9) on page 12 in
    https://jair.org/index.php/jair/article/view/10952"""
#Cross-learning dynamics  = replicator dynamics
def cross_learning(state, fitness):
    return replicator(state, fitness)

"""Based on equations on p4 in
https://www.researchgate.net/publication/221454203_Frequency_adjusted_multiagent_Q-learning"""
#Boltzmann Q-learning is already defined in open_spiel.python.egt.dynamics
def boltzmann_qlearning(state, fitness, temperature=1):
    return boltzmannq(state, fitness, temperature)

def boltzmann_faqlearning(state, fitness, temperature=1):
    exploitation = (1. / temperature) * replicator(state, fitness)
    exploration = state.dot(np.log(state / state))
    return exploitation - state * exploration

#for a lenient agent/population the fitness function changes
#instead of having (for player P1) for each Action(P1): fitness(P1) = PayoffMatrix(P1) * ActionProb(P2)
#we have: equation (7) on page 6 of http://www.flowermountains.nl/pub/Bloembergen2010lfaq.pdf
# fitness(P1) = sum_j ( PayoffMatrix(P1)[j] * ActionProb(P2)[j] * [sum_k:Aik≤Aij(ActionProb(P2)[k])**k - sum_k:Aik<Aij(ActionProb(P2)[k])**k] / sum_k:Aik=Aij(ActionProb(P2)[k])
class LenientMultiPopulationDynamics(MultiPopulationDynamics):
    def __init__(self, payoff_tensor, dynamics, k=5):
        super(LenientMultiPopulationDynamics, self).__init__(payoff_tensor, dynamics)
        self.k = k

    #We follow the notation of the paper referenced above
    def __call__(self, state, time=None):
        state = np.array(state)
        n = self.payoff_tensor.shape[0]     # number of players
        ks = self.payoff_tensor.shape[1:]   # number of strategies for each player
        assert state.shape[0] == sum(ks)

        states = np.split(state, np.cumsum(ks)[:-1])
        dstates = [None] * n
        #for each player: calculate change in strategy
        for p in range(n):
            fitness = np.zeros(shape=ks[p])
            #for each action
            for i in range(ks[p]):
                # for all other players
                u_i = 0
                for p_ in set(range(n)) - {p}:
                    # for each action each other player can take
                    for j in range(ks[p_]):
                        strictworse_actions = []    # k:Aik<Aij
                        worse_actions = []          # k:Aik≤Aij
                        equal_actions = []          # k:Aik==Aij
                        for k in range(ks[p_]):
                            if self.payoff_tensor[p][i][k] < self.payoff_tensor[p][i][j]:
                                strictworse_actions.append(states[p_][k])
                                worse_actions.append(states[p_][k])
                            elif self.payoff_tensor[p][i][k] == self.payoff_tensor[p][i][j]:
                                worse_actions.append(states[p_][k])
                                equal_actions.append(states[p_][k])
                        #u_i = sum_j (PayoffMatrix(P1)[j] * ActionProb(P2)[j] * [sum_k:Aik≤Aij(ActionProb(P2)[k])**k - sum_k:Aik<Aij(ActionProb(P2)[k])**k] / sum_k:Aik=Aij(ActionProb(P2)[k]
                        u_i += self.payoff_tensor[p][i][j]*states[p_][j] * (np.sum(worse_actions)**self.k - np.sum(strictworse_actions)**self.k) / np.sum(equal_actions)
                fitness[i] = u_i
            dstates[p] = self.dynamics[p](states[p], fitness)
        return np.concatenate(dstates)