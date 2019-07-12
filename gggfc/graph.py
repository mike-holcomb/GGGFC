import time
import csv
import os

class OpsMap:
    def __init__(self, fname = 'grammar_files/ops.keras'):
        self.lookup = {}
        self.code = []
        self.num_inputs = []
        self.channel_mult = []

        with open(fname) as tsvfile:
            reader = csv.DictReader(tsvfile, dialect='excel-tab')
            for i, row in enumerate(reader):
                self.lookup[row['name']] = i
                self.code.append(row['code'])
                self.num_inputs.append(int(row['num_inputs']))
                self.channel_mult.append(int(row['channel_mult']))

    def get_ops_info(self, op_name):
        op_id = self.lookup[op_name]
        return self.num_inputs[op_id], self.channel_mult[op_id], self.code[op_id]

    def print_opsmap(self):
        print(self.lookup)
        print(self.code)
        print(self.num_inputs)
        print(self.channel_mult)

class Node:
    def __init__(self, name, op_type, attrs={}, is_input=False, is_output=False):
        self.name = name
        self.op_type = op_type
        self.attrs = attrs
        self.inbound = set()
        self.outbound = set()
        self.is_input = is_input
        self.is_output = is_output

    def get(self, attr):
        return self.attrs.get(attr,None)

    def get_prefix(self):
        prefixes = self.name.split('/')
        prefixes.pop()
        return '/'.join(prefixes)

    def add_inbound(self, source):
        if self.is_input:
            raise ValueError("Input node cannot have input")

        self.inbound.add(source)

    def add_outbound(self, sink):
        if self.is_output:
            raise ValueError("Output node cannout have output")

        self.outbound.add(sink)

    def remove_inbound(self, source):
        if self.is_input:
            raise ValueError("Input node does not have input")

        self.inbound.remove(source)

    def remove_outbound(self, sink):
        if self.is_output:
            raise ValueError("Output node does not have output")
        self.outbound.remove(sink)

    @staticmethod
    def _swap_prefix(old_name, new_prefix):
        old_names = old_name.split('/')
        # old_names[-2] = new_prefix
        return new_prefix + '/' + old_names[-1]
    
    def copy_node(self, new_prefix):
        new_name = Node._swap_prefix(self.name, new_prefix)
        new_node = Node(
            name=new_name,
            op_type=self.op_type,
            attrs=self.attrs,
            is_input=self.is_input,
            is_output=self.is_output
        )

        for inbound in self.inbound:
            new_node.add_inbound(Node._swap_prefix(inbound, new_prefix))

        for outbound in self.outbound:
            new_node.add_outbound(Node._swap_prefix(outbound, new_prefix))

        return new_name, new_node

    # def replace_inbound(self, replacement):
    #     if (not self.is_input):
    #         for input_ in self.inbound:
    #             input_.add_outbound(replacement)
    #             input_.remove_outbound(self)
    #             replacement.add_inbound(input_)

    # def replace_outbound(self, replacement):
    #     if (not self.is_output):
    #         for output_ in self.outbound:
    #             output_.add_inbound(replacement)
    #             output_.remove_inbound(self)
    #             replacement.add_outbound(output_)

    def print_edges(self):
        for ins in self.inbound:
            print("\t\"%s\" -> \"%s\"" % (ins, self.name))
            

    def write_edges(self):
        lines = []
        for ins in self.inbound:
            lines.append("\t\"%s\" -> \"%s\"" % (ins, self.name))
        return lines


class Graph:
    def __init__(self, prefix):
        self.prefix = prefix
        self.nodes = {}
        self.source = None
        self.sink = None
        self.ids = {}
        self.nonterms = []
        self.curr = 0
        self.func_name = "build_model_%s" % time.strftime("%Y%m%d_%H%M%S")

    def add_nonterm(self, node_name):
        self.nonterms.append(node_name)

    def _build_name(self, op_type):
        op_id = self.ids.get(op_type,0)
        name = "%s/%s_%d" % (self.prefix, op_type, op_id)
        self.ids[op_type] = op_id + 1
        return name

    def add_input(self, op_type, attrs={}):
        name = self._build_name(op_type)

        if name in self.nodes.keys():
            raise ValueError("Cannot have duplicate names: %s" % name)
        new_node = Node(name, op_type, attrs, is_input=True)
        self.nodes[name] = new_node
        self.source = name

        return name

    def add_output(self, op_type, attrs={}):
        name = self._build_name(op_type)

        if name in self.nodes.keys():
            raise ValueError("Cannot have duplicate names: %s" % name)
        new_node = Node(name, op_type, attrs, is_output=True)
        self.nodes[name] = new_node
        self.sink = name

        return name

    def add_node(self, op_type, attrs={}):
        name = self._build_name(op_type)

        if name in self.nodes.keys():
            raise ValueError("Cannot have duplicate names: %s" % name)
        new_node = Node(name, op_type, attrs)
        self.nodes[name] = new_node

        return name

    def add_edge(self, source_name, sink_name):
        source_node = self.nodes[source_name]
        sink_node = self.nodes[sink_name]

        source_node.add_outbound(sink_name)
        sink_node.add_inbound(source_name)
        
    def _replace_inbound(self, original_node, replace_name):
        if original_node.is_input:
            self.source = replace_name
            return

        replace_node = self.nodes[replace_name]
        replace_node.is_input = False

        for inbound_name in original_node.inbound:
            inbound_node = self.nodes[inbound_name]
            inbound_node.outbound.discard(original_node.name)
            inbound_node.outbound.add(replace_name)
            replace_node.inbound.add(inbound_name)
            # print(original_node.name + ' ' + replace_name + ' ' + inbound_name)

            
    def _replace_outbound(self, original_node, replace_name):
        if original_node.is_output:
            self.sink = replace_name
            return

        # print('Trying to replace ' + str(replace_name) + ' ' + original_node.name)
        replace_node = self.nodes[replace_name]
        replace_node.is_output = False

        for outbound_name in original_node.outbound:
            outbound_node = self.nodes[outbound_name]
            outbound_node.inbound.discard(original_node.name)
            outbound_node.inbound.add(replace_name)
            replace_node.outbound.add(outbound_name)
        # print(original_node.name + '..' + replace_name + '..')

    def replace_node(self, original_name, input_replace, outbound_replace):
        original_node = self.nodes[original_name]
        self._replace_inbound(original_node, input_replace)
        self._replace_outbound(original_node, outbound_replace)
        if original_node.is_input:
            self.source = input_replace
        if original_node.is_output:
           self.sink = outbound_replace

        if original_name in self.nonterms:
            self.nonterms.remove(original_name)

        del self.nodes[original_name]


    def insert_graph(self, original_name, new_graph):
        original_node_ = self.nodes[original_name]
        parent_prefix = original_node_.get_prefix()
        child_prefix = new_graph.prefix
        new_prefix = parent_prefix + '_' + str(self.curr) 
        new_graph_copy = new_graph.copy_graph(new_prefix + '/' + child_prefix)
        self.curr += 1
        for node_name, node in new_graph_copy.nodes.items():
            self.nodes[node_name] = node

        self._replace_inbound(original_node_, new_graph_copy.source)
        self._replace_outbound(original_node_, new_graph_copy.sink)

        del self.nodes[original_name]

        return new_prefix

        # print(new_graph_copy.source + ' ' + new_graph_copy.sink)
        # if original_node.is_input:
        #     self.source = input_replace
        # if original_node.is_output:
        #    self.sink = outbound_replace

        # if original_name in self.nonterms:
        #     self.nonterms.remove(original_name)

        # del self.nodes[original_name]

    def print_dot(self):
        print("digraph %s {" % self.prefix)

        sorted_nodes = self._sort_nodes()
        for node_name in sorted_nodes:
            print("\t\"%s\"" % node_name) 
        print("")

        for node_name in sorted_nodes:
            node = self.nodes[node_name]
            if (not node.is_input):
                node.print_edges()
        print("")    
        print("}")
        # print("Source: %s, sink: %s" % (self.source, self.sink))

    def write_dot(self, save_dir, fname="graph"):
        save_file = os.path.join(save_dir, fname + '.dot')

        dot_lines = []
        dot_lines.append("digraph %s {" % self.prefix)

        sorted_nodes = self._sort_nodes()
        for node_name in sorted_nodes:
            dot_lines.append("\t\"%s\"" % node_name) 
        dot_lines.append("")

        for node_name in sorted_nodes:
            node = self.nodes[node_name]
            if (not node.is_input):
                edge_lines = node.write_edges()
                dot_lines.extend(edge_lines)
        dot_lines.append("")    
        dot_lines.append("}")

        with open(save_file,"w") as f:
            f.write("\n".join(dot_lines))


    def copy_graph(self, new_prefix):
        new_graph_ = Graph(new_prefix)
        # new_graph.source = self.source
        # new_graph.sink = self.sink

        for n in self.nodes.values():
            new_name, new_node = n.copy_node(new_prefix)
            new_graph_.nodes[new_name] = new_node
            if n.name == self.sink:
                new_graph_.sink = new_name
            if n.name == self.source:
                new_graph_.source = new_name

        return new_graph_

    def get_node_op(self, node_name):
        return self.nodes[node_name].op_type

    def get_node_prefix(self, node_name):
        return self.nodes[node_name].get_prefix()

    def _get_sorted_nodes(self):
        """
        https://stackoverflow.com/questions/47192626/deceptively-simple-implementation-of-topological-sorting-in-python
        """
        seen = set()
        path = []
        q = [ self.sink ]

        while q:
            v = q.pop()
            if v not in seen:
                seen.add(v)
                path.insert(0,v)
                # path.append(v)
                q.extend(list(self.nodes[v].inbound))

        # print(path)
        return path

    def _sort_visit(self, v, visited, stack):
        visited[v] = True
        # print("Visited nodes: " + str(visited))

        for i in self.nodes[v].inbound:
            if not visited[i]:
                self._sort_visit(i, visited, stack)

        stack.append(v)

    def _sort_nodes(self):
        visited_list = [False] * len(self.nodes)
        visited = dict(zip(self.nodes.keys(),visited_list))
        stack = []
        self._sort_visit(self.sink, visited, stack)

        # print("Sort nodes stack: " +  str(stack))
        return stack


    def _build_function_head(self):
        return ["def " + self.func_name +"(num_channels):"], self.func_name

    def convert_to_keras_builder(self, ops_map_file):
        layer_names_map = {}
        layer_nums = {}

        output_names = {}
        output_num = 0

        # BUILD OUTPUT/LAYER NAME MAPPING
        sorted_nodes = self._sort_nodes()
        for node_name in sorted_nodes:
            node = self.nodes[node_name]
            this_layer_num = layer_nums.get(node.op_type,0)
            layer_names_map[node.name] = node.op_type + '_' + str(this_layer_num)
            layer_nums[node.op_type] = this_layer_num + 1

            output_names[node.name] = 'y' + str(output_num)
            output_num += 1

        # BEGIN CODE GENERATION
        func_code, function_name = self._build_function_head()
        
        opsmap = OpsMap(fname=ops_map_file)
        # opsmap.print_opsmap()

        X = 1 # Channel number multiplier
        for node_name in sorted_nodes:
            node_ = self.nodes[node_name]
            output_var = output_names[node_name]
            layer_name = layer_names_map[node_name]
            op_type = node_.op_type

            num_inputs, channel_mult, code = opsmap.get_ops_info(op_type)

            X *= channel_mult

            if num_inputs == 0:
                code_line = "    %s = %s" % (output_var, code)
            elif num_inputs == 1:
                input_list = list(node_.inbound)
                input_var = output_names[input_list[0]]
                mod_code = code.replace("?",str(X))
                code_line = "    %s = %s(%s)" % ( output_var, mod_code, input_var )
            # elif num_inputs == 2:
            #     input_list = list(node_.inbound)
            #     input_var1, input_var2 = output_names[input_list[0]], output_names[input_list[1]]
            #     mod_code = code.replace("?",str(X))
            #     code_line = "    %s = %s([%s, %s])" % ( output_var, mod_code, input_var1, input_var2 )

            else:
                input_list = list(node_.inbound)
                mod_code = code.replace("?",str(X))
                code_line = "    %s = %s([" % ( output_var, mod_code)
                input_vars = [ output_names[x] for x in input_list]
                input_var_code = ", ".join(input_vars)
                code_line = code_line + input_var_code + "])"

            # print(code_line)
            func_code.append(code_line)

        in_var, out_var = output_names[self.source], output_names[self.sink]

        func_code.append("    return Model(inputs=%s, outputs=%s)" % (in_var, out_var))
        code_string = "\n".join(func_code)
        #print(code_string)
        return function_name, code_string

        # locals()["myfunction"]()

if __name__ == "__main__":
    g = Graph('graph1')
    g.add_input('input')
    g.add_output('output')
    model_node = g.add_node('model')

    g.add_edge('graph1/input_0','graph1/model_0')
    g.add_edge('graph1/model_0','graph1/output_0')

    g2 = g.copy_graph('graph2')
    
    g.print_dot()
    print("")
    g2.print_dot()
