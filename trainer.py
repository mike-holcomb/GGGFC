from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
from absl import logging

import os

from gggfc.grammar import Grammar
from gggfc.policy import  Policy
from gggfc.generator import Generator

FLAGS = flags.FLAGS

flags.DEFINE_integer('n_trials', 10, 'Number of graphs to try')
flags.DEFINE_integer('epochs_per_trial',10,'Number of epochs to train per trial')
flags.DEFINE_integer('n_grow', 10, 'Number of grow steps in generator')
flags.DEFINE_integer('max_depth', 5, 'Maximum depth of graph generator')
flags.DEFINE_string('grammar_path','gggfc/grammar_files','Location of grammar files')
flags.DEFINE_string('ops_map_file','gggfc/grammar_files/ops.keras','Location of operator definitions')
flags.DEFINE_string('policy_file','gggfc/grammar_files/b_policy.json','Location of generation policy file')
flags.DEFINE_string('model_path','models','Path to store generated models')
flags.DEFINE_string('graph_path','graphs','Path to store generated model graphs in dot')
flags.DEFINE_string('sequences_path','sequences','Path to store model generation sequences')
flags.DEFINE_string('history_path','history','Path to store model training histories')


def _build_output_dirs():
    output_dirs = [FLAGS.model_path, FLAGS.sequences_path, FLAGS.history_path, FLAGS.graph_path]

    for output_dir in output_dirs:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

def _save_model_function(function_name, code_string):
    this_model_file = os.path.join(FLAGS.model_path, function_name + ".py")
    with open(this_model_file,"w") as f:
            f.write(code_string)

def main(argv):
    del argv  # Unused.
    
    # Set up output directories
    _build_output_dirs()

    # Build generator
    grammar = Grammar()
    grammar.build(FLAGS.grammar_path)
    policy = Policy(FLAGS.policy_file)
    gen = Generator(grammar,policy, max_depth=FLAGS.max_depth, grow_n = FLAGS.n_grow)

    # Generate model files
    new_graph = gen.generate(save_dir=FLAGS.sequences_path)
    function_name, code_string = new_graph.convert_to_keras_builder(ops_map_file=FLAGS.ops_map_file)
    _save_model_function(function_name, code_string)

    new_graph.write_dot(FLAGS.graph_path,function_name)

if __name__ == '__main__':
  app.run(main)
