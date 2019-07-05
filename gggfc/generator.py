from grammar import Grammar
from graph import Graph
from random import choice, random
from queue import PriorityQueue

class GeneratorHistory:
    def __init__(self):
        self.step_count = 0
        self.lh_counts = {}
        self.production_counts = {}

        self.sequence = {
            "step" : [], 
            "lh" : [],
            "rh" : [],
            "depth" : []
        }

    def _add_sequence(self, lh, production_name, depth):
        self.sequence["step"].append(self.step_count)
        self.step_count += 1

        self.sequence["lh"].append(lh)
        self.sequence["rh"].append(production_name)
        self.sequence["depth"].append(depth)

    def add(self, lh, production_name, depth):
        lh_count = self.lh_counts.get(lh,0)
        self.lh_counts[lh] = lh_count + 1

        lh_prod_counts = self.production_counts.get(lh,{})
        name_count = lh_prod_counts.get(production_name,0)
        lh_prod_counts[production_name] = name_count+1
        self.production_counts[lh] = lh_prod_counts

        self._add_sequence(lh, production_name, depth)

    def print_history(self):
        print(self.lh_counts)
        print(self.production_counts)
        print(self.sequence)

class Generator:
    def __init__(self, grammar, p_grow=0.7, max_depth=10, sr_budget=(2,4)):
        self.grammar = grammar
        # self.stack = []
        self.p_grow = p_grow
        self.max_depth = max_depth
        self.sr_budget = sr_budget
        self.history = None

    def get_production(self, op_type, depth=-1):
        grow = ( random() < self.p_grow) and depth < self.max_depth
        productions = self.grammar.get_productions(op_type, grow)
        production = choice(productions)
        self.history.add(op_type, production.prefix, depth+1)
        return production, production.nonterms

    def generate(self):
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

        self.history.print_history()

        return graph

if __name__ == "__main__":
    grammar = Grammar()
    grammar.build()
    gen = Generator(grammar,p_grow=0.7,max_depth=6)
    new_graph = gen.generate()
    function_name, code_string = new_graph.convert_to_keras_builder()
    with open("models/"+function_name+".py","w") as f:
        f.write(code_string)
    new_graph.write_dot()
    print(len(new_graph.nodes))
    
    