from grammar import Grammar
from graph import Graph
from random import choice

class Generator:
    def __init__(self, grammar):
        self.grammar = grammar
        self.stack = []

    def get_production(self, op_type):
        productions = self.grammar.get_productions(op_type)
        production = choice(productions)
        return production, production.nonterms

    def generate(self):
        graph, nonterms = self.get_production('root')
        self.stack.extend(nonterms)

        while len(self.stack) > 0:
            next_node = self.stack.pop()
            next_op = graph.get_node_op(next_node)
            rh, new_nonterms = self.get_production(next_op)
            # next_prefix = graph.get_node_prefix(next_node)
            new_prefix = graph.insert_graph(next_node,rh)

            new_nonterms = [ new_prefix + '/' + x for x in new_nonterms]
            self.stack.extend(new_nonterms)

        return graph

if __name__ == "__main__":
    grammar = Grammar()
    grammar.build()
    gen = Generator(grammar)
    new_graph = gen.generate()
    new_graph.convert_to_keras_builder()
    new_graph.print_dot()
    