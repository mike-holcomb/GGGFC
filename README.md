# Graph Generator from Grammar For Computation

## Design Overview:
The system is designed to generate novel convolutional neural network designs for classifying images from the CIFAR-10 dataset based on a type of context-free graph grammar.  We generate graphs according to a probabilistic CFG generation algorithm.  The probablities are learned parameters from a genetic algorithm.  Specificall, several hundred graphs are trained and the production counts of the highest performing graphs as maximum likelihood estimates for the probabilities of the best graph generator (top 50% of validation accuracy) for the next, improved population of graphs.

### Grammar
The grammar is a context-free graph grammar in that all rules have only one non-terminal vertex on the left hand side. This grammar can parse four key convolutional neural network architectures (VGG-19, Inception v3, ResNet v1, DenseNet). 

Since convolutional neural network classifiers are directed acyclic graphs (DAG) with one source node (input images) and one output node (classifications) we make the simplifying assumption that all non-terminals have one input and one output as well as all redexes (right hand side of productions) also have one input and one output.  As such, R-applications of these rules in the forward generation phase is a straightforward process of connecting the old input edge to the input node of the redex, likewise for the output.

One key issue in ensuring that the grammar produces valid convolutional graphs was ensuring that the dimensions of inputs and outputs are compatible.  For the most part, this is handled at graph generation time using a handful of conventions discussed in "Generation - Conventions."  Operations that modify spatial or channel dimensions are handled by different operation names within the grammar.

At a high-level, the root node produces a typical model with an tail, backbone, and a head.  The tail is the input, while the head is the global pooler of features as well as the mapping of features to output classifications.  The backbone contains the primary feature extraction machinery of the model. The backbone node can be composed of several body and spatial reduction blocks (discussed further in "Generation").  Body nodes are in turn composed of various convolution blocks, potentially with residual connections, as well as feature concatenation blocks.

#### Context-Sensitive vs Context-Free
This grammar could also be summarized with fewer rules using a context-sensitive grammar.  Namely, the five productions in this grammar intended to parse DenseNet/Inception-style feature concatenations can be replaced with two rules that add a concatention as well as a rule that adds edges.  The two rules could also be used to derive more graph possibilities than could be described by the five production rules.  However, the context-sensitive rules would add significant complexity to our generation and policy learning processes. 

## Generation
The generation algorithm is dramatically simplified but using a probabilistic context-free grammar.  Essentially, generation algorithm is as follows, assuming we have a grammar Gamma, alphabet Sigma, and policy Pi (mapping from productions to probabilities where all probabilities for a given nonterminal sum to 1):

1. Initialize a graph G with the root node
2. Add to a queue a list of non-terminal nodes in the graph
3. While the queue is not empty:
    1. Pop a non-terminal, u, node off of the queue.
    2. Select a production, p, from Gamma according to the probability distribution from Pi for the non-terminal u.
    3. Replace node u in the graph G with redex of the production p.
    4. Add the non-terminal nodes within the redex of the production p to the queue.

#### Conventions
When translating operations to Python code:

* We assume that the first convolutional operation has 8, 16, or 32 output filters (a base filter count); the choice is explained more in depth in "Constraints."
* Operations that reduce the spatial dimensions of the input:
    * are contained only in "BlockSR" blocks, 
    * always cut the width and height in half,
    * are only in the redexes of "Backbone" nodes.
* "Backbone" nodes are used to limit the number of "BlockSR" blocks with the goal of limiting the number of spatial reductions in the overall network.
* Operations that increase the number of channels:
    * are only in the redexes of "Body" blocks,
    * and always do so either by a factor of 2 or 4.
    
### Constraints
To ensure that the generated graphs are tractable, we enforce three primary parameterized constraints during the generation process.

#### 1. Growth Steps
Productions naturally by their definition can either grow the graph or terminate the graph.  Specifically, productions that contain more than one non-terminal node in their redex will grow the graph.  Therefore, the grammar is segregated into productions that grow the graph as well as those that terminate the graph and we set a budget or ceiling for the number of "growth" productions that can be used.

#### 2. Maximum Depth
Envisioning the parse tree of the generated graph, productions are R-applied at various depths (or distance from the root).  In order to limit the growth of the graph, we also limit the maximum depth at which a growth rule can be applied.

#### 3. Parameters
Once a graph is generated, we finally check whether the total number of parameters is within an acceptable range.  This ensures that models are able to fit in memory or not too small to perform well.  Multiple base filter channel counts are tested in order to maximize the number of generated graphs that are trained.

## Training
We leverage the Keras/Tensorflow neural network framework and a selected portion of the CIFAR-10 dataset to train our graphs.

We use a genetic algorithm to refine the production rule probabilities, or policy, over the course of several refined generations (populations).  At a high level, our algorithm is as follows:

1. Initialize our policy Pi to be the uniform distribution
2. For each generation (poplulation):
    1. Generate N graphs according to grammar Gamma with policy Pi
    2. Train these N graphs to convergence
    3. Select N/2 graphs with the highest validation accuracy
    4. Calculate new production rule probabilities using the counts of the productions used in the top performers of the prior step
    5. Assign these probabilities to Pi
    
For the initial training phase, we select 24,000 images from the training set of CIFAR 10.  These 24,000 images are further subdivided into a training as well as evaluation dataset.  The training and evaluation set is limited in order to expedite the training phase.

In the final evaluation phase, the highest performing graphs generated in the final population will be trained on the full CIFAR-10 training set and evaluated on the true test set for final evaluation of the algorithm.
