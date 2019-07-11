import json
import random

class Policy:
    def __init__(self, fname=None):
        if fname is None:
            self.distribution = {}
        else:
            with open(fname, 'r') as f:
                self.distribution = json.load(f)

    def _get_actions(self, nonterminal, is_growing, depth):
        actions = []
        probs = []

        if is_growing:
            action_set = self.distribution[nonterminal]['growing']
        else:
            action_set = self.distribution[nonterminal]['not_growing']

        for action, prob in action_set.items():
            actions.append(action)
            probs.append(prob[depth])

        return actions, probs

    def _rebuild_distribution(self, actions, probs, choices):
        adj_actions, cum_probs = [], []
        prob_i = 0
        for action, prob in zip(actions, probs):
            if action in choices:
                adj_actions.append(action)
                prob_i += prob
                cum_probs.append(prob_i)
        # Normalize
        tot = float(cum_probs[-1])
        cum_probs = [ x / tot for x in cum_probs]

        return adj_actions, cum_probs

    def choose(self, nonterminal, is_growing, depth, choices, eps=0.0):
        if eps > 0 and random.random() < eps:
            return random.choice(choices)
        actions, probs = self._get_actions(nonterminal,is_growing,depth)

        if len(actions) < 1:
            actions, probs = self._get_actions(nonterminal,not is_growing,depth)

        adj_actions, adj_probs = self._rebuild_distribution(actions, probs, choices)

        x, ret = random.random(), adj_actions[0]
        # print(adj_actions, adj_probs, x)
        for c, p in zip(adj_actions, adj_probs):
            if x < p:
                return c

        return adj_actions[0]


if __name__ == "__main__":
    policy1 = Policy(fname='grammar_files/b_policy.json')
    print(policy1.distribution)
    print(policy1._get_actions('backbone',True, 2))
    print(policy1.choose('backbone',True,1,['backbone_a','backbone_c']))