import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import seaborn as sns
from scipy.spatial import distance


'''
This file is for some basic functions and classes that will be used in the

'''


'''
Graph Generating Functions
'''

class SimpleGraph(nx.Graph):
  '''
  Generate a random normal hidden variable network, where each node is given
  a fitness randomly from normal distribution, and probability of connection
  is higher for less more separate fitness
  
  self:
    node_count: counts total number of nodes to automatically label next node
    node_edges_evaluated: modified by add_fitness_edges to track what edges have
      been evaluated. Then if new nodes are added, and add_fitness_edges run again
      eval_func will only run for nodes that have not been evaluated
  '''
  def __init__(self, incoming_graph_data=None, **attr):
    super().__init__(incoming_graph_data, **attr)
    self.node_count = 0
    self.node_edges_evaluated = 0
    
  def add_node(self, **attr):
    super().add_node(self.node_count, **attr)
    self.node_count += 1



class SimpleSpatialGraph(nx.Graph):
  '''
  Generate a spatial graph such that every node requires an n-dimensional tuple
  coordinate
  params:
    dimension: spatial dimension of the graph, default is 2
    size: size of each dimension of the graph. Graph is assumed to start from 
      [0,0,...], default is [1,1]
     
  '''
  def __init__(self, incoming_graph_data=None, dimension=2, size=[1,1], **attr):
    super().__init__(incoming_graph_data, **attr)
    if(len(size) != dimension):
      raise Exception('Size of the graph does not match its dimension')
      
    self.dimension = dimension
    self.node_count = 0
    self.node_edges_evaluated = 0
    self.size = size
    
  def add_node(self, coordinate=None, **attr):
    '''
    Add a node optionally with a specified coordinate. If no coordinate is given,
    the node will be added with uniform random coordinates in the size of the graph
    params:
      coordinate: optional coordinate for the node
    '''
    
    if(coordinate is None):
      #no coordinate passed, generate one uniformly at random
      coordinate = []
      for i in range(self.dimension):
        coordinate.append(np.random.uniform(0, self.size[i]))
    else:
      #check that the coordinate passed matched the size of the graph
      if(type(coordinate) != list):
        raise Exception('Missing list coordinate')
      if(len(coordinate) != self.dimension):
        raise Exception('Coordinate does not have the same dimensions as the graph')
    
    super().add_node(self.node_count, coordinate=coordinate, **attr)
    self.node_count += 1



class SpatialGraph(nx.Graph):
  '''
  Generate a spatial graph such that every node requires an n-dimensional tuple
  coordinate
  Same as SimpleSpatialGraph but requires assignment of node labels
  params:
    dimension: spatial dimension of the graph, default is 2
    size: size of each dimension of the graph. Graph is assumed to start from 
      [0,0,...], default is [1,1]
     
  '''
  def __init__(self, incoming_graph_data=None, dimension=2, size=[1,1], **attr):
    super().__init__(incoming_graph_data, **attr)
    if(len(size) != dimension):
      raise Exception('Size of the graph does not match its dimension')
      
    self.dimension = dimension
    self.node_count = 0
    self.node_edges_evaluated = 0
    self.size = size
    
  def add_node(self, n, coordinate=None, **attr):
    '''
    Add a node optionally with a specified coordinate. If no coordinate is given,
    the node will be added with uniform random coordinates in the size of the graph
    params:
      coordinate: optional coordinate for the node
    '''
    
    if(coordinate is None):
      #no coordinate passed, generate one uniformly at random
      coordinate = []
      for i in range(self.dimension):
        coordinate.append(np.random.uniform(0, self.size[i]))
    else:
      #check that the coordinate passed matched the size of the graph
      if(type(coordinate) != list):
        raise Exception('Missing list coordinate')
      if(len(coordinate) != self.dimension):
        raise Exception('Coordinate does not have the same dimensions as the graph')
    
    super().add_node(n, coordinate=coordinate, **attr)
    self.node_count += 1




def add_normal_nodes(G, n):
  '''
  Add a normal of nodes with normally distributed fitness function to given graph
  params:
    G: passed in SIMPLE networkx graph to modify
    n: number of nodes to add
  '''
  for i in range(n):
    G.add_node(fitness=np.random.normal())


def add_exponential_nodes(G, n, beta=1.0):
  '''
  Add nodes with exponentially distributed fitness function to given graph
  params:
    G: passed in SIMPLE networkx graph to modify
    n: number of nodes to add
  '''
  for i in range(n):
    G.add_node(fitness=np.random.exponential(beta))  


def add_fitness_edges(G, eval_func):
  '''
  Add edges to graph based on passesd evaluation function.
  For every possible pair of nodes, the eval_func will be run to determine
  whether an edge should be added. Assumes that a SimpleGraph is being used (i.e.
  nodes labelled from 0 to n)
  params:
    G: passed in SIMPLE networkx graph to modify
    eval_func: function that determines if edge should be made
      eval_func should take in two fitness values and return True or False
  '''
  for i in range(G.node_count):
    node_start = max((G.node_edges_evaluated+1, i+1))
        
    for j in range(node_start, G.node_count):
      add_fitness_edge(G, eval_func, i, j)

  G.node_edges_evaluated = i



def add_fitness_edge(G, eval_func, i, j):
  '''
  Simply add an edge between two nodes in a graph if the eval_func determines
  it. Will be based on fitness of the two nodes
  params:
    G: passed in networkx graph to modify
    eval_func: function that determines if edges should be made, takes in 
      two fitness values and returns True or False
    i, j: node labels
  '''
  fitness_i = G.nodes[i]['fitness']
  fitness_j = G.nodes[j]['fitness']
  if(eval_func(fitness_i, fitness_j)):
    G.add_edge(i, j)
    


def add_eval_edges(G, eval_func):
  '''
  More general version of add_fitness_edges, passing in the entire node to
  eval_func
  '''
  for i in range(G.node_count):
    node_start = max((G.node_edges_evaluated+1, i+1))
        
    for j in range(node_start, G.node_count):
      add_eval_edge(G, eval_func, i, j)

  G.node_edges_evaluated = i


def add_eval_edge(G, eval_func, i, j):
  '''
  Analogous to add_fitness_edge for add_eval_edges
  params:
    i, j: node labels (not actual dictionary)
  '''
  if(eval_func(G.nodes[i], G.nodes[j])):
    G.add_edge(i, j)

  





'''
Graphing Functions
'''

def graph_degree_distribution(G, alpha=None):
  '''
  Create histogram/distplot of degree distribution of given graph
  params:
    G: passed in networkx graph
    alpha: draw a power line to attempt to fit 
  return:
    degrees: array containing degree sequence of nodes (non-ordered)
  '''
  degrees = []
  for i in list(G.degree):
    degree = i[1]
    degrees.append(degree)
  
  sns.distplot(degrees)
  
  if(alpha):
    x = np.linspace(0, max(degrees))
    y = np.power(x, -1 * alpha)
    plt.plot(x, y)

  return degrees
  


def graph_degree_distribution_subplot(G, sub1, sub2, sub3, alpha=None):
  '''
  Create histogram/distplot of degree distribution of given graph
  This version gives us the ability to specify plt.subplot() parameters to create multiple graphs in the plot
  params:
    G: passed in networkx graph
    alpha: draw a power line to attempt to fit 
    sub1, sub2, sub3: positional arguments matching those for
        plt.subplot(sub1, sub2, sub3)
  return:
    degrees: array containing degree sequence of nodes (non-ordered)
  '''
  degrees = []
  for i in list(G.degree):
    degree = i[1]
    degrees.append(degree)
  
  plt.subplot(sub1, sub2, sub3)
  sns.distplot(degrees)
   
  if(alpha):
    x = np.linspace(0, max(degrees))
    y = np.power(x, -1 * alpha)
    plt.plot(x, y)

  return degrees


  
def graph_power_degree_distribution(G, alpha=None):
  '''
  Create histogram/distplot of degree distribution of given graph with
  bins of power of 2 size and logarithmic scales
  params:
    G: passed in networkx graph
    alpha: draw a power line to attempt to fit
  return:
    (degrees, counts, bins)
    degrees: array containing degree sequence of nodes (non-ordered)
    counts: number of nodes of degree in a given bin
    bins: bin boundaries
  '''
  degrees = []
  for i in list(G.degree):
    degree = i[1]
    degrees.append(degree)
    
  bins = power_of_2_bins(degrees)
  sns.distplot(degrees, bins)
  plt.yscale('log')
  plt.xscale('log')
  
  counts, bins = np.histogram(degrees, bins)
  return (degrees, counts, bins)


  
def graph_fitness_distribution(G):
  '''
  Create graph of fitnesses by smoothed histogram
  params:
    G: passed in networkx graph
  '''
  fitnesses = []
  for i in list(G.nodes):
    fitness = G.nodes[i]['fitness']
    fitnesses.append(fitness)
    
  sns.distplot(fitnesses)



def graph_spatial_network_plot(G, graph_edges=False, highlight_edges=False):
  '''
  Create a plot showing what a 2D spatial graph looks like
  params:
    G: passed in SimpleSpatialGraph
    graph_edges: graph edges as well
    highlight_edges: Optionally parameter to draw in specific edges highlighted, will 
        also highlight the latest node
  '''
  X = []
  Y = []
  degrees = []
  
  for i in list(G.nodes):
    coordinate = G.nodes[i]['coordinate']
    X.append(coordinate[0])
    Y.append(coordinate[1])
    degrees.append(G.degree[i])
  
  if(graph_edges):
    for i in G.edges:
      coordinate_i = G.nodes[i[0]]['coordinate']
      coordinate_j = G.nodes[i[1]]['coordinate']
      x = (coordinate_i[0], coordinate_j[0])
      y = (coordinate_i[1], coordinate_j[1])
      plt.plot(x, y, alpha=0.2, color='grey')

  if(highlight_edges):
    for i in highlight_edges:
      coordinate_i = G.nodes[i[0]]['coordinate']
      coordinate_j = G.nodes[i[1]]['coordinate']
      x = (coordinate_i[0], coordinate_j[0])
      y = (coordinate_i[1], coordinate_j[1])
      plt.plot(x, y, alpha=0.2, color='red')


  plt.scatter(X, Y, c=degrees, cmap='viridis')
  plt.colorbar()

  if(highlight_edges):
    node = G.nodes[len(G.nodes) - 1]
    x = [node['coordinate'][0]]
    y = [node['coordinate'][1]]
    plt.scatter(x, y, c='red')

  plt.xlim([0, G.size[0]])
  plt.ylim([0, G.size[1]])
    






'''
Helper Functions
'''

def power_of_2_bins(array):
  '''
  Create bins of power of 2 width for power-law histogram
  params:
    array: array of numbers to his)togram
  return:
    bin edges in format for plt.histogram or sns.distplot
  '''
  M = max(array)
  bins = [1]
  edge = 1
  while(edge < M):
    edge = edge * 2
    bins.append(edge)
  
  return bins
  


def average_degree(G):
    '''
    Calculate average node degree <k> of the given SimpleGraph
    '''
    total_degree = 0
    degrees = G.degree()
    for i in range(G.node_count):
        total_degree += degrees[i]
    return total_degree / G.node_count