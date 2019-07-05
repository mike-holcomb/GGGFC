import glob, os
import networkx as nx
from graph import Graph, Node


class Grammar:
    def __init__(self):
        self.productions = {}
        self.productions['root'] = []

        # self.terminal_only = {}
        # self.nonterminal = {}

        self.growing = {}
        self.finishing = {}

        self.body_sr = []
        self.body_nosr = []

    def convert_to_graph(self, graphx, prefix):
        new_graph = Graph(prefix)
        name_map = {}
        is_terminal = True
        nonterms = []

        has_input, has_output = False, False
        for node_name, node in graphx.nodes(data=True):
            if node.get('input','no') == 'yes':
                new_name = new_graph.add_input(node['op_type'])
                has_input = True
                if node.get('output','no') == 'yes':
                    new_graph.sink = new_name
                    has_output = True
            elif node.get('output','no') == 'yes':
                new_name = new_graph.add_output(node['op_type'])
                has_output = True
            else:
                new_name = new_graph.add_node(node['op_type'])

            name_map[node_name] = new_name

            if node.get('shape','circle') == 'box':
                nonterms.append(node_name)
                is_terminal = False

        assert has_input, "Missing input for prefix %s" % prefix
        assert has_output, "Missing output for prefix %s" % prefix

        sort_nodes = nx.algorithms.dag.topological_sort(graphx)
        for sn in sort_nodes:
            if sn in nonterms:
                new_graph.add_nonterm(name_map[sn])

        for node_name, node in graphx.nodes(data=True):
            new_name = name_map[node_name]
            for inbound_old in graphx.predecessors(node_name):
                new_inbound = name_map[inbound_old]
                new_graph.nodes[new_name].add_inbound(new_inbound)
                new_graph.nodes[new_inbound].add_outbound(new_name)

        return new_graph, is_terminal

    def build(self, grammar_dir="grammar_files"):
        for file in glob.glob(os.path.join(grammar_dir,"*.dot")):
            prod_name, _  = os.path.splitext(os.path.basename(file))
            lh = prod_name.split('_')[0]
            new_gx = nx.DiGraph(nx.drawing.nx_agraph.read_dot(file))
            # print("*** " + prod_name + ' ' + lh + " ***")
            self.add_production(prod_name,lh, new_gx)
            # for node_name, shape in new_gx.nodes(data='shape',default='circle'):
            #     print(node_name + ' ' + str(shape== "box"))


    def add_production(self, prod_name, lh, rh):
        # print(prod_name + " " + lh)
        lh_list = self.productions.get(lh,[])
        # print(lh_list)
        g, is_terminal = self.convert_to_graph(rh, prod_name)
        self.productions[lh] = lh_list + [ g ]

        if is_terminal or len(g.nodes) < 2:
            lh_list = self.finishing.get(lh,[])
            self.finishing[lh] = lh_list + [ g ]
        else:
            lh_list = self.growing.get(lh,[])
            self.growing[lh] = lh_list + [ g ]


    def get_productions(self,op_type, grow=False):
        ret = None

        try:
            self.growing[op_type]
            can_grow = True
        except KeyError as error:
            can_grow = False

        if grow and can_grow:
            ret = self.growing[op_type]
        else:
            try:
                ret = self.finishing[op_type]
            except KeyError as error:
                ret = self.growing[op_type]

        return ret

    def print_grammar(self):
        print("Num total: " + str(len(self.productions)))
        print("Num finishing: " + str(len(self.finishing)))
        print("Num growing: " + str(len(self.growing)))
        for k, v in self.productions.items():
            print("Term: " + k)
            for rh in v:
                rh.print_dot()

if __name__ == "__main__":
    grammar = Grammar()
    grammar.build()
    grammar.print_grammar()
