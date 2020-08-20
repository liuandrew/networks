'''
This code was used in the production of "Spatial strength centrality and the 
effect of spatial embeddings on network architecture", Andrew Liu and Mason A. 
Porter 2020, Physical Review E.
'''

'''
This file is contains basic functions used in the production and analysis
of spatial networks
We use the Python package NetworkX as the basis for creating networks
You may also need to install the seaborn package for some of the plotting functions

Both of these packages can be installed if you have pip via
    $ pip install networkx seaborn
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import collections
import seaborn as sns
from scipy.spatial import distance


'''
======================
Graph Generating Class
======================
This is the basic graph object we are working with throughout the paper
'''


class SimpleSpatialGraph(nx.Graph):
    '''
    Generate a spatial graph such that every node requires an n-dimensional tuple
    coordinate
    params:
        dimension: spatial dimension of the graph, default is 2
        size: size of each dimension of the graph. Graph is assumed to start from 
            [0,0,...], default is [1,1]
            i.e. size=[1, 1] will produce a graph embedded in [0, 1] x [0, 1] 
    
    Example:
        g = SimpleSpatialGraph() #this creates our empty graph
        g.add_node() #this adds a node with position given uniformly at random
                                 #with the label 0. Subsequent nodes are labelled 1, 2, etc.
    '''
    def __init__(self, incoming_graph_data=None, dimension=2, size=[1,1], **attr):
        super().__init__(incoming_graph_data, **attr)
        if(len(size) != dimension):
            raise Exception('Size of the graph does not match its dimension')
            
        self.dimension = dimension
        self.node_count = 0
        self.node_edges_evaluated = 0
        self.size = size
        
    def add_node(self, label=False, coordinate=None, **attr):
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
        if(not label):
            super().add_node(self.node_count, coordinate=coordinate, **attr)
        else:
            super().add_node(label, coordinate=coordinate, **attr)
        self.node_count += 1







'''
=====================
Node adding functions
=====================
These functions are used to add nodes to a SimpleSpatialGraph with specific fitness
values
'''

def add_normal_nodes(G, n):
    '''
    Add a number n of nodes with normally distributed fitness function to given graph G
    params:
        G: passed in SimpleSpatialGraph to modify
        n: number of nodes to add
    '''
    for i in range(n):
        G.add_node(fitness=np.random.normal())




    





'''
==================
Graphing Functions
==================
Functions that will draw plots of some graph characteristics
If you do not wish to use the seaborn package, you can comment these out
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


        






'''
================
Helper Functions
================
These functions help to calculate some values of the graph
'''

def average_degree(G):
    '''
    Calculate average node degree <k> of the given SimpleGraph
    '''
    total_degree = 0
    degrees = G.degree()
    for i in range(G.node_count):
        total_degree += degrees[i]
    return total_degree / G.node_count





def pbc_distances(G, pbc=True):
    '''
    Calculate the distances between nodes, can return periodic boundary conditioned
    distances or simple Euclidean distances

    params:
        coordinates, coordinates_2: two sets of coordinates to use in calculation
        pbc: whether or not to use periodic boundary conditions
    return:
        distances - distances given in a n x n numpy array, where distances[i][j] is the pbc
            distance between points i and j
    '''
    coordinates = []
    for i in G.nodes:
        coordinates.append(G.nodes[i]['coordinate'])
    coordinates = np.array(coordinates)
    dimensions = G.size
    
    num_dim = coordinates.shape[1]
    num_coords = coordinates.shape[0]
    delta_blocks = []
    
    for d in range(num_dim):
        coordinate_block = coordinates.T[d]
        dimension = dimensions[d]
        
        delta = np.abs(coordinate_block.reshape(num_coords, 1) - coordinate_block)
        delta = np.triu(delta)
        if(pbc):
            for i in range(delta.shape[0]):
                for j in range(delta.shape[1]):
                    if(delta[i][j] > 0.5 * dimension):
                        delta[i][j] = dimension - delta[i][j]
                                
        delta_blocks.append(delta ** 2)
    
    distances = np.sqrt(np.sum(delta_blocks, axis=0))
    return distances




def pbc_distances_single(G, node=False, pbc=True):
    '''
    Calculate the distances from a given node to all others in the graph
    Used by SPA model
    params:
        G: the graph
        node: node label to calculate distances from, if none use last node
        pbc: whether periodic boundary conditions should be used
    return:
        distances - distances given in a n x n numpy array, where distances[i][j] is the pbc
            distance between points i and j
    '''
    coordinates = []
    dimensions = G.size
    for i in range(G.node_count):
        coordinates.append(G.nodes[i]['coordinate'])    
    coordinates = np.array(coordinates)
    
    num_dim = coordinates.shape[1]
    num_coords = coordinates.shape[0]
    delta_blocks = []
    
    for d in range(num_dim):
        dimension = dimensions[d]
        coordinate_block = coordinates.T[d]
        if(node is False):
            node = G.node_count - 1
        delta = np.abs(coordinate_block - G.nodes[node]['coordinate'][d])
        if(pbc):
            delta = np.where(delta > 0.5 * dimension, dimension - delta, delta)
        
        delta_blocks.append(delta ** 2)
    
    distances = np.sqrt(np.sum(delta_blocks, axis=0))
    return distances