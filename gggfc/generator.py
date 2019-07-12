import os
from random import choice, random
from queue import PriorityQueue

from .grammar import Grammar
from .policy import Policy
from .graph import Graph


class GeneratorHistory:
    def __init__(self):
        self.step_count = 0
        self.lh_counts = {}
        self.production_counts = {}

        self.sequence = {
            "step" : [], 
            "lh" : [],
            "rh" : [],
            "depth" : [],
            "grow" : []
        }

    def _add_sequence(self, lh, production_name, depth, is_grow):
        self.sequence["step"].append(self.step_count)
        self.step_count += 1

        self.sequence["lh"].append(lh)
        self.sequence["rh"].append(production_name)
        self.sequence["depth"].append(depth)
        self.sequence["grow"].append(is_grow)

    def add(self, lh, production_name, depth, is_grow):
        lh_count = self.lh_counts.get(lh,0)
        self.lh_counts[lh] = lh_count + 1

        lh_prod_counts = self.production_counts.get(lh,{})
        name_count = lh_prod_counts.get(production_name,0)
        lh_prod_counts[production_name] = name_count+1
        self.production_counts[lh] = lh_prod_counts

        self._add_sequence(lh, production_name, depth, is_grow)

    def print_history(self):
        print(self.lh_counts)
        print(self.production_counts)
        print(self.sequence)

    def write_history(self, save_dir, fname):
        save_file = os.path.join(save_dir, fname + '.csv')
        with open(save_file, 'w') as f:
            f.write('step, lh, rh, depth, grow\n')
            for i in range(self.step_count):
                f.write('%d, %s, %s, %d, %d\n' % 
                    (i, 
                    self.sequence['lh'][i], 
                    self.sequence['rh'][i],
                    self.sequence['depth'][i],
                    1 if self.sequence['grow'][i] else 0)
                ) 

class Generator:
    def __init__(self, grammar, policy, max_depth=5, grow_n = 25):
        self.grammar = grammar
        self.policy = policy
        # self.stack = []
        self.max_depth = max_depth
        self.N = float(grow_n)
        self.history = None

    def _production_names(self, productions):
        return [ production.prefix for production in productions ]

    def _select_production(self, productions, production_name):
        for production in productions:
            if production.prefix == production_name:
                return production
        return None 

    def get_production(self, op_type, depth=-1):
        grow = (  self.history.step_count < self.N ) \
            and depth < self.max_depth
        productions, real_grow = self.grammar.get_productions(op_type, grow)
        production_name =  self.policy.choose(op_type, real_grow, depth, self._production_names(productions))
        # production = choice(productions)
        self.history.add(op_type, production_name, depth+1, grow)
        ret_production = self._select_production(productions, production_name)
        return ret_production, ret_production.nonterms

    def generate(self, save_dir):
        self.history = GeneratorHistory()
        graph, nonterms = self.get_production('root')
        queue = PriorityQueue()
        for item in nonterms:
            queue.put((0, item))
        # self.stack.extend(nonterms)

        lh_counts = {}
        production_counts={}

        while not queue.empty():
            depth, next_node = queue.get()
            next_op = graph.get_node_op(next_node)
            rh, new_nonterms = self.get_production(next_op, depth)
            # next_prefix = graph.get_node_prefix(next_node)
            new_prefix = graph.insert_graph(next_node,rh)

            new_nonterms = [ new_prefix + '/' + x for x in new_nonterms]
            for item in new_nonterms:
                queue.put((depth+1, item))

        #self.history.print_history()
        self.history.write_history(save_dir, graph.func_name)

        return graph

if __name__ == "__main__":
    grammar = Grammar()
    grammar.build()
    policy = Policy(fname='grammar_files/b_policy.json')
    gen = Generator(grammar,policy, max_depth=10, grow_n = 20)
    new_graph = gen.generate()
    function_name, code_string = new_graph.convert_to_keras_builder()
    with open("models/"+function_name+".py","w") as f:
        f.write(code_string)
    new_graph.write_dot()
    print(len(new_graph.nodes))