import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import collections
import seaborn as sns
from scipy.spatial import distance
import heapq
import os
import gc
import pickle
import math
from sklearn.linear_model import LinearRegression

from base_network_extensions import *

'''
!!
Evaluate fitness edges with vectorized functions
!!
'''

def pbc_distances(G, pbc=True):
    '''
    Calculated the periodic boundary condition distances given coordinates and dimensions of space
    Coordinates need to be passed in an (n, d) numpy array 
    params:
        coordinates, coordinates_2: two sets of coordinates to use in calculation
    return:
        distances - distances given in a n x n numpy array, where distances[i][j] is the pbc
            distance between points i and j
    '''
    coordinates = []
    for i in G.node:
        coordinates.append(G.node[i]['coordinate'])
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

    

def add_vectorized_eval_edges(G, dimensions=[1, 1], threshold=0.05, beta=-1.5, keep_edge_distances=False):
    '''
    Add edges between each pair of nodes in a simple spatial graph with normally distributed fitness values
    Assumes that number of nodes is a multiple of 500
    
    params:
        threshold: Heavistep threshold for edge to form
        beta: Power factor for distance
        c: scalar for distance (NOTE: removed this parameter as it is absorbed by threshold)
        keep_edge_distances: boolean that will store edge distances as edge attributes. Increases memory usage for
            the graph, but necessary for closeness centrality calculation
    '''
    coordinates = []
    fitnesses = []
    for i in G.node:
        coordinates.append(G.node[i]['coordinate'])
        fitnesses.append(G.node[i]['fitness'])
    
    coordinates = np.array(coordinates)
    fitnesses = np.array(fitnesses)
    
    #not breaking up into blocks
    fitness_evals = np.abs(fitnesses.reshape(G.node_count, 1) - fitnesses)
    #distances = pbc_distances(coordinates, coordinates, dimensions)
    distances = pbc_distances(G)
    distances = np.where(distances > 1, distances ** beta, 1)
    
    scores = fitness_evals * distances
    
    for i in range(G.node_count):
        for j in range(i+1, G.node_count):
            if(scores[i][j] > threshold):
                G.add_edge(i, j)
                G[i][j]['distance'] = distances[i][j]


'''
!!
Fitness Evaluation Functions
!!
'''

def normal_difference_heavistep(fitness_i, fitness_j):
  '''
  Expect 2 normally distributed fitnesses
  Check if the absolute difference (distance) between the fitnesses exceeds
  a given threshold
  params:
    fitness_i, fitness_j: fitness values of the two nodes
  TWEAK:
    threshold: value that fitness absolute difference must exceed 
  '''
  threshold = 3.5
  if(abs(fitness_i - fitness_j) > threshold):
    return True
  else:
    return False



def normal_difference_probability(fitness_i, fitness_j):
  '''
  Expect 2 normally distributed fitnesses
  Give absolute difference 
  '''
  pass
  
  

def distance_heavistep(node_i, node_j):
  '''
  Expect 2 node dictionaries that include 'coordinate' keys
  Check if they are close enough (euclidean distance) to form an edge
  params:
    node_i, node_j: dictionary entry of the two nodes (given by G.nodes[i])
  TWEAK:
    threshold: distance that nodes have to be within to form edge
  '''
  threshold = 0.3
  coordinate_i = node_i['coordinate']
  coordinate_j = node_j['coordinate']
  if(distance.euclidean(coordinate_i, coordinate_j) < threshold):
    return True
  else:
    return False
  

def normal_difference_distance_heavistep(node_i, node_j):
  '''
  Expect 2 node dictionaries with both 'coordinate' and 'fitness' keys
  Check if the absolute difference of fitness times a multiplier of 
  distance (euclidean) exceeds a given threshold
  
  |fitness_i - fitness_j| * (dist(i, j))^-a > theta 
  
  params:
    node_i, node_j: dictionary entry of the two nodes (given by G.nodes[i])
  TWEAK:
    threshold: value that fitness times distance modifier must exceed
    alpha: exponent of distance used (will be negative)
  '''
  threshold = 1.5
  alpha = 2
  fitness_i = node_i['fitness']
  fitness_j = node_j['fitness']
  coordinate_i = node_i['coordinate']
  coordinate_j = node_j['coordinate']
  dist = distance.euclidean(coordinate_i, coordinate_j)
  total = (dist ** alpha) * abs(fitness_i - fitness_j)
  if(total > threshold):
    return True
  else:
    return False


def gnp_random(node_i, node_j):
    '''
    Uniformly randomly add edge with probability p
    TWEAK:
        p: probability of adding edge
    '''
    p = 0.05
    if(np.random.uniform() < p):
        return True
    else:
        return False
    
def fully_connected(node_i, node_j):
    '''
    Return True for all edges
    '''
    return True


'''
MODEL GENERATING FUNCTION
This code completes the whole graph generation from initiation to edge evaluation
It is used throughout the rest of the code
'''

'''
Load target_thresholds
'''
# target_thresholds = pickle.load(open("target_thresholds.pickle", 'rb'))

def graph_model(n=500, beta=-1.0, size=[100, 100], threshold=False, keep_edge_distances=False):
    '''
    The most important thing to tweak is the threshold
    If no threshold is specified, we will look for the threshold found for n=500 that gives
        mean degree 20
    '''
    if(not threshold):
        threshold = target_thresholds[beta]
    
    g = SimpleSpatialGraph(size=size)
    add_normal_nodes(g, n)

    add_vectorized_eval_edges(g, dimensions=size, threshold=threshold, beta=beta, keep_edge_distances=keep_edge_distances)
    return g



'''
!!
Graph the spatial network and plot centralities
!!
'''

def graph_spatial_network_plot_valued(G, values=False, graph_edges=False, hide_ticks=True, subplot=False,
  highlight_edges=False, highlight_recent=False, color_points=False, color_bar=False, bounds=False, pbc=True,
  edge_style=None, ss_title=False, alpha=0.2):
  '''
  Create a plot showing what a 2D spatial graph looks like, coloring by passed values and sized by degree
  params:
    G: passed in SimpleSpatialGraph
    values: values according to the labels of the nodes
    graph_edges: graph edges as well (this function takes into account periodic boundary conditions)
    highlight_edges: Optionally parameter to draw in specific edges highlighted, will 
        also highlight the latest node
    color_points: array of color labels if coloring points specifically
    bounds: given by list of lists - each row with two entries
    pbc: whether to graph as pbc
  '''
  X = []
  Y = []
  degrees = []
  scatter_values = []

  if(subplot):
    plt.subplot(subplot[0], subplot[1], subplot[2])
  else:
    plt.figure(figsize=(16, 16))
  for i in list(G.nodes):
    coordinate = G.nodes[i]['coordinate']
    X.append(coordinate[0])
    Y.append(coordinate[1])
    degrees.append(G.degree[i])
    if(values):
      scatter_values.append(values[i])
  
  if(ss_title):
    _, spatial_strength = report_spatial_strength_centrality(G, pbc=pbc, graph=False,
      ret=True, normalized=4)
    plt.title(r'S = ' + str(round(spatial_strength, 3)))
  if(graph_edges):
    if(pbc):
        for i in G.edges:
          coordinate_i = G.nodes[i[0]]['coordinate']
          coordinate_j = G.nodes[i[1]]['coordinate']
          points = False
          if(abs(coordinate_i[0] - coordinate_j[0]) > G.size[0] / 2 and
              abs(coordinate_i[1] - coordinate_j[1]) > G.size[0] / 2):
            max_x = max(coordinate_i[0], coordinate_j[0])
            max_y = max(coordinate_i[1], coordinate_j[1])
            min_x = min(coordinate_i[0], coordinate_j[0])
            min_y = min(coordinate_i[1], coordinate_j[1])
            x1 = (max_x, max_x + 50)
            x2 = (min_x, min_x - 50)
            y1 = (max_y, max_y + 50)
            y2 = (min_y, min_y - 50)
            points = (x1, x2, y1, y2)
          elif(abs(coordinate_i[0] - coordinate_j[0]) > G.size[0] / 2):
            coordinates = sorted([[coordinate_i[0], coordinate_i[1]], 
              [coordinate_j[0], coordinate_j[1]]])
            x1 = (coordinates[1][0], coordinates[1][0] + 50)
            x2 = (coordinates[0][0], coordinates[0][0] - 50)
            y1 = (coordinates[1][1], coordinates[0][1])
            y2 = (coordinates[0][1], coordinates[1][1])
            points = (x1, x2, y1, y2)
          elif(abs(coordinate_i[1] - coordinate_j[1]) > G.size[0] / 2):
            coordinates = sorted([[coordinate_i[1], coordinate_i[0]], 
              [coordinate_j[1], coordinate_j[0]]])
            y1 = (coordinates[1][0], coordinates[1][0] + 50)
            y2 = (coordinates[0][0], coordinates[0][0] - 50)
            x1 = (coordinates[1][1], coordinates[0][1])
            x2 = (coordinates[0][1], coordinates[1][1])
            points = (x1, x2, y1, y2)

          if(points):
            plt.plot(points[0], points[2], alpha=alpha, color='grey')
            plt.plot(points[1], points[3], alpha=alpha, color='grey')
          else:
            x = (coordinate_i[0], coordinate_j[0])
            y = (coordinate_i[1], coordinate_j[1])
            plt.plot(x, y, alpha=alpha, color='grey')
    else:
        for i in G.edges:
          coordinate_i = G.nodes[i[0]]['coordinate']
          coordinate_j = G.nodes[i[1]]['coordinate']
          x = (coordinate_i[0], coordinate_j[0])
          y = (coordinate_i[1], coordinate_j[1])
          plt.plot(x, y, alpha=alpha, color='grey')

  if(highlight_edges):
    for i in highlight_edges:
      coordinate_i = G.nodes[i[0]]['coordinate']
      coordinate_j = G.nodes[i[1]]['coordinate']
      x = (coordinate_i[0], coordinate_j[0])
      y = (coordinate_i[1], coordinate_j[1])
      plt.plot(x, y, alpha=alpha, color='red')

  if(color_points):
    plt.scatter(X, Y, c=color_points, s=degrees)
  elif(color_bar):
    plt.scatter(X, Y, c=scatter_values, s=degrees, cmap='viridis')
    plt.colorbar()
  else:
    plt.scatter(X, Y, s=degrees)
    
  if(highlight_recent):
    node = G.nodes[len(G.nodes) - 1]
    x = [node['coordinate'][0]]
    y = [node['coordinate'][1]]
    plt.scatter(x, y, c='red')
  if(bounds):
    plt.xlim([bounds[0][0], bounds[0][1]])
    plt.ylim([bounds[1][0], bounds[1][1]])
  else:
    plt.xlim([0, G.size[0]])
    plt.ylim([0, G.size[1]])

  if(hide_ticks):
    plt.xticks([])
    plt.yticks([])
    

def plot_centrality_correlation(G, centrality1=False, centrality2=False):
    '''
    Plot the correlations between centralities - will plot a grid of correlations,
    each plot will be ordered by the value on the x-axis
    Returns the calculated centralities dictionary
    '''
    centralities = {
        'Degree': G.degree
    }
    sorting_dict = {}
    print('Calculating betweenness')
    centralities['Betweenness'] = nx.betweenness_centrality(g)
    print('Calculating closeness')
    centralities['Closeness'] = nx.closeness_centrality(g)
    print('Gathering fitnesses')
    fitnesses = {}
    for i in range(G.node_count):
        fitnesses[i] = G.node[i]['fitness']
    centralities['Fitness'] = fitnesses
    
    print('Plotting graphs')
    plt.figure(figsize=(16, 16))
    #plt.tight_layout()
    
    for i in range(G.node_count):
        sorting_dict[i] = {}
        for key in centralities:
            sorting_dict[i][key] = centralities[key][i]
    
    for i, key1 in enumerate(centralities):
        #key1 will be the y-axis
        for j, key2 in enumerate(centralities):
            plt.subplot(len(centralities), len(centralities), i * len(centralities) + j + 1)
            #plot histogram along diagonal
            if(i == j):
                hist_values = []
                for k in range(G.node_count):
                    hist_values.append(centralities[key1][k])
                sns.distplot(hist_values)
                plt.xlabel(key1)
                plt.ylabel('Proportion')
            else:
                sorted_keys = sorted(sorting_dict, key=lambda x: (sorting_dict[x][key2], sorting_dict[x][key1]))
                x = []
                y = []
                for k in range(G.node_count):
                    x.append(centralities[key2][sorted_keys[k]])
                    y.append(centralities[key1][sorted_keys[k]])
                plt.scatter(x, y, alpha=0.4)
                plt.xlabel(key2)
                plt.ylabel(key1)
    
    plt.tight_layout()
    return centralities


'''
!!
Calculate Barycenter
!!
'''
def get_coordinates(g):
    '''
    return all coordinates for g
    '''
    coordinates = []
    for i in range(len(g.size)):
        coordinates.append([])
        
    for i in range(g.node_count):
        for j in range(len(g.size)):
            coordinates[j].append(g.node[i]['coordinate'][j])
    
    return coordinates
    
    
    
def regular_barycenter(g, weighted=False):
    '''
    calculate regular barycenter without pbc
    params:
        weighted: whether to weight nodes based on their degrees
    '''
    coordinates = get_coordinates(g)
    barycenter = []
    for i in range(len(g.size)):
        x = coordinates[i]
        if(weighted):
            degrees = np.array(list(g.degree)).T[1]
            x_com = np.sum(x * degrees) / np.sum(degrees)
        else:
            x_com = np.sum(x) / g.node_count
        barycenter.append(x_com)
    return barycenter
    
    
def pbc_barycenter(g, weighted=False):
    '''
    calculate barycenter for nodes in a pbc space
    based on circular calculation on wikipedia
    params:
        weighted: whether to weight nodes based on their degrees
    '''
    coordinates = get_coordinates(g)
    barycenter = []
    for i in range(len(g.size)):
        x_max = g.size[i]
        x = coordinates[i]
        theta = np.array(x) * 2 * np.pi / x_max
        zeta = np.cos(theta)
        xi = np.sin(theta)
        
        zeta_bar = 0
        xi_bar = 0
        
        if(weighted):
            degrees = np.array(list(g.degree)).T[1]
            zeta_bar = np.sum(zeta * degrees) / np.sum(degrees)
            xi_bar = np.sum(xi * degrees) / np.sum(degrees)
        else:
            zeta_bar = np.sum(zeta) / g.node_count
            xi_bar = np.sum(xi) / g.node_count
        
        theta_bar = np.arctan2(-zeta_bar, -xi_bar) + np.pi
        x_com = x_max * theta_bar / (2  * np.pi)
        barycenter.append(x_com)
    return barycenter



def pbc_barycenter_centrality(g, weighted=False):
    '''
    calculate distance of each point to barycenter using pbc
    '''
    coordinates = get_coordinates(g)
    barycenter = pbc_barycenter(g, weighted)
    delta = np.abs(np.array(coordinates).T - barycenter)
    delta = delta.T
    for i, dimension in enumerate(g.size):
        delta[i] = np.where(delta[i] > 0.5 * dimension, dimension - delta[i], delta[i])
    distances = np.sum(delta.T ** 2, axis=1)
    distances = np.sqrt(distances)
    barycenter_centrality = {}
    
    for i in range(g.node_count):
        barycenter_centrality[i] = distances[i]
    return barycenter_centrality
    

def regular_barycenter_centrality(g, weighted=False):
    coordinates = get_coordinates(g)
    barycenter = regular_barycenter(g, weighted)
    distances = np.array(coordinates).T - barycenter
    distances = np.sum(distances ** 2, axis=1)
    distances = np.sqrt(distances)
    barycenter_centrality = {}
    
    for i in range(g.node_count):
        barycenter_centrality[i] = distances[i]
    return barycenter_centrality
    

'''
!!
Analyze graphs in a graphs dictionary based on betas in betas list
!!
'''
# def analyze_graphs_with_function(func_type, verbose=0, xlabel=False, ylabel=False, num_trials=30, 
#                                  subplot=None, betas=None, individual_trials=False, values=False,
#                                 invert_betas=True, label=False, ):
#     '''
#     Analyze the set of graphs using the given func_type
#     params:
#         func_type: one of predetermined strings:
        
#             'degree_assortativity'
#             'fitness_assortativity' (fitness assortativity coefficient)
#             'laplacian_spectrum' (value of the second eigenvalue)
#             'clustering'
#             'geodesic'
#             'mean_degree'
#         verbose:
#             0: only print betas in line
#             1: print per beta
#             2: print per trial
#         subplot: pass 3 index list for subplot placement
#     '''
#     if(betas is None):
#         betas = []
#         for beta in target_thresholds:
#             betas.append(beta)
    
#     all_avg_values = []
#     all_betas = []
#     all_values = []
#     if(values):
#         print('values received')
#         all_avg_values = values['all_avg_values']
#         all_betas = values['all_betas']
#         all_values = values['all_values']
        
#     centrality_types = {
#         'degree_assortativity': r'Degree assortativity',
#        'laplacian_spectrum': r'2nd Smallest Eigenvalue, Lambda',
#        'clustering': r'Clustering (C)',
#        'geodesic': r'Mean geodesic distance (L)',
#        'connectance': r'Connectance',
#         'clustering_ratio': r'Clustering Ratio, (C / (N/E^2))',
#         'mean_degree': r'Mean degree <k>',
#         'average_edge_length': r'Average edge length',
#         'average_spatial_strength': r'Average spatial strength'
#     }
#     if(subplot is None):
#         plt.figure(figsize=(8, 8))
#     else:
#         plt.subplot(subplot[0], subplot[1], subplot[2])
#     if(not values):
#         for i, beta in enumerate(betas):
#             if(verbose == 0):
#                 print(beta, end=', ')
#             if(verbose > 0):
#                 print('Set %(set_num)d analysis: beta = %(beta)f' % {'beta': beta, 'set_num': i})
#             graph_array = graphs[beta]
#             value_sum = 0
#             xlabels = {'degree_assor'}
#             for j, graph in enumerate(graph_array):
#                 if(func_type == 'degree_assortativity'):
#                     value = nx.degree_assortativity_coefficient(graph)
#                 elif(func_type == 'fitness_assortativity'):
#                     value = nx.attribute_assortativity_coefficient(graph, 'fitness')
#                 elif(func_type == 'laplacian_spectrum'):
#                     value = nx.laplacian_spectrum(graph)[1]
#                 elif(func_type == 'clustering'):
#                     value = nx.average_clustering(graph)
#                 elif(func_type == 'connectance'):
#                     num_edges = len(graph.edges)
#                     num_nodes = graph.node_count
#                     value = num_edges / (num_nodes ** 2)
#                 elif(func_type == 'clustering_ratio'):
#                     num_edges = len(graph.edges)
#                     num_nodes = graph.node_count
#                     clustering = nx.average_clustering(graph)
#                     value = clustering / (num_edges / (num_nodes ** 2))
#                 elif(func_type == 'geodesic'):
#                     try:
#                         value = nx.average_shortest_path_length(graph)
#                     except Exception as e:
#                         print('Exception raised, graph likely not connected. \
#                         Trial %(trial_num)d, beta = %(beta)f' % {'beta': beta, 'trial_num': j})
#                 elif(func_type == 'mean_degree'):
#                     value = average_degree(graph)
#                 elif(func_type == 'average_edge_length'):
#                     value = average_edge_length(graph, False)
#                 elif(func_type == 'average_edge_length_normalized'):
#                     value = average_edge_length(graph)
#                 elif(func_type == 'average_spatial_strength'):
#                     c, value = spatial_strength_centrality(graph)
#                 else:
#                     print('Function type not recognized')
#                     return
#                 all_values.append(value)
#                 all_betas.append(-(beta))
#                 value_sum += value
#                 if(verbose > 1):
#                     print('Trial %(trial_num)d: value = %(value)f' % {'trial_num': j, 'value': value})

#             avg_value = value_sum / len(graph_array)
#             if(verbose > 0):
#                 print('Set complete, average value = %(avg_value)f' % {'avg_value': avg_value})
#                 print('-----------------------')
#             all_avg_values.append(avg_value)
#     values = {
#         'all_avg_values': all_avg_values,
#         'all_betas': all_betas,
#         'all_values': all_values
#     }
    
#     if(invert_betas):
#         for i in range(len(betas)):
#             betas[i] = -betas[i]
#         for i in range(len(all_betas)):
#             all_betas[i] = -all_betas[i]
    
#     if(str(type(subplot)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>"):
#         if(label is False)
#             label='mean value'
#         subplot.scatter(betas, all_avg_values, label=label)
#         if(individual_trials):
#             subplot.scatter(all_betas, all_values, alpha=0.15, label="trial values")
#             subplot.legend()
#         subplot.set_xlabel(r'$\beta$')
#         subplot.set_ylabel(centrality_types[func_type])
#         if(yticks):
#             subplot.set_yticks(yticks)
#         if(yrange):
#             subplot.set_ylim(yrange)
#     else:
#         if(label is False):
#             label='mean value'
#         plt.scatter(betas, all_avg_values, label=label)
#         if(individual_trials):
#             plt.scatter(all_betas, all_values, alpha=0.15, label="trial values")
#             plt.legend()
#         plt.xlabel(r'$\beta$')
#         plt.ylabel(centrality_types[func_type], labelpad=20)
#         if(yticks):
#             plt.yticks(yticks)
#         if(yrange):
#             plt.ylim(yrange)
#     # plt.scatter(betas, all_avg_values, label="mean value")
#     # if(individual_trials):
#     #     plt.scatter(all_betas, all_values, alpha=0.15, label="trial values")
#     #     plt.legend()
#     # plt.xlabel(r'$\beta$', fontsize=14)
#     # plt.ylabel(centrality_types[func_type], fontsize=16)
#     #plt.title('number of trials per beta = %(num_trials)d' % {'num_trials': num_trials})
#     return values
        
        
def analyze_graphs_with_function(func_type, verbose=0, xlabel=False, ylabel=False, num_trials=30, 
                                 subplot=None, betas=None, individual_trials=False, values=False,
                                invert_betas=True, yticks=None, yrange=None, label=False):
    '''
    Analyze the set of graphs using the given func_type
    params:
        func_type: one of predetermined strings:
        
            'degree_assortativity'
            'fitness_assortativity' (fitness assortativity coefficient)
            'laplacian_spectrum' (value of the second eigenvalue)
            'clustering'
            'geodesic'
            'mean_degree'
        verbose:
            0: only print betas in line
            1: print per beta
            2: print per trial
        subplot: pass 3 index list for subplot placement
    '''
    if(betas is None):
        betas = []
        for beta in target_thresholds:
            betas.append(beta)
    
    all_avg_values = []
    all_betas = []
    all_values = []
    if(values):
        print('values received')
        all_avg_values = values['all_avg_values']
        all_betas = values['all_betas']
        all_values = values['all_values']
        
    centrality_types = {
        'degree_assortativity': r'Degree assortativity',
       'laplacian_spectrum': r'2nd Smallest Eigenvalue, Lambda',
       'clustering': r'Mean clustering coefficient (C)',
       'geodesic': r'Mean geodesic distance (L)',
       'connectance': r'Connectance',
        'clustering_ratio': r'Clustering Ratio, (C / (N/E^2))',
        'mean_degree': r'Mean degree, \langle k \rangle',
        'average_edge_length': r'Mean edge length',
        'average_spatial_strength': r'Mean spatial strength (S)'
    }
    if(subplot is None):
        plt.figure(figsize=(8, 8))
    elif(type(subplot) == list):
        plt.subplot(subplot[0], subplot[1], subplot[2])
    if(not values):
        for i, beta in enumerate(betas):
            if(verbose == 0):
                print(beta, end=', ')
            if(verbose > 0):
                print('Set %(set_num)d analysis: beta = %(beta)f' % {'beta': beta, 'set_num': i})
            graph_array = graphs[beta]
            value_sum = 0
            xlabels = {'degree_assor'}
            for j, graph in enumerate(graph_array):
                if(func_type == 'degree_assortativity'):
                    value = nx.degree_assortativity_coefficient(graph)
                elif(func_type == 'fitness_assortativity'):
                    value = nx.attribute_assortativity_coefficient(graph, 'fitness')
                elif(func_type == 'laplacian_spectrum'):
                    value = nx.laplacian_spectrum(graph)[1]
                elif(func_type == 'clustering'):
                    value = nx.average_clustering(graph)
                elif(func_type == 'connectance'):
                    num_edges = len(graph.edges)
                    num_nodes = graph.node_count
                    value = num_edges / (num_nodes ** 2)
                elif(func_type == 'clustering_ratio'):
                    num_edges = len(graph.edges)
                    num_nodes = graph.node_count
                    clustering = nx.average_clustering(graph)
                    value = clustering / (num_edges / (num_nodes ** 2))
                elif(func_type == 'geodesic'):
                    try:
                        value = nx.average_shortest_path_length(graph)
                    except Exception as e:
                        print('Exception raised, graph likely not connected. \
                        Trial %(trial_num)d, beta = %(beta)f' % {'beta': beta, 'trial_num': j})
                elif(func_type == 'mean_degree'):
                    value = average_degree(graph)
                elif(func_type == 'average_edge_length'):
                    value = average_edge_length(graph, False)
                elif(func_type == 'average_edge_length_normalized'):
                    value = average_edge_length(graph)
                elif(func_type == 'average_spatial_strength'):
                    c, value = spatial_strength_centrality(graph)
                else:
                    print('Function type not recognized')
                    return
                all_values.append(value)
                all_betas.append(-(beta))
                value_sum += value
                if(verbose > 1):
                    print('Trial %(trial_num)d: value = %(value)f' % {'trial_num': j, 'value': value})

            avg_value = value_sum / len(graph_array)
            if(verbose > 0):
                print('Set complete, average value = %(avg_value)f' % {'avg_value': avg_value})
                print('-----------------------')
            all_avg_values.append(avg_value)
    values = {
        'all_avg_values': all_avg_values,
        'all_betas': all_betas,
        'all_values': all_values
    }
    
    if(invert_betas):
        for i in range(len(betas)):
            betas[i] = -betas[i]
        for i in range(len(all_betas)):
            all_betas[i] = -all_betas[i]
    if(False):
        pass
    if(str(type(subplot)) == "<class 'matplotlib.axes._subplots.AxesSubplot'>"):
        if(not label):
            label = 'mean value'
        subplot.scatter(betas, all_avg_values, label=label)
        if(individual_trials):
            subplot.scatter(all_betas, all_values, alpha=0.15, label="trial values")
            subplot.legend()
        subplot.set_xlabel(r'$\beta$')
        subplot.set_ylabel(centrality_types[func_type])
        if(yticks):
            subplot.set_yticks(yticks)
        if(yrange):
            subplot.set_ylim(yrange)
    else:
        if(not label):
            label = 'mean_value'
        plt.scatter(betas, all_avg_values, label=label)
        if(individual_trials):
            plt.scatter(all_betas, all_values, alpha=0.15, label="trial values")
            plt.legend()
        plt.xlabel(r'$\beta$')
        plt.ylabel(centrality_types[func_type], labelpad=20)
        if(yticks):
            plt.yticks(yticks)
        if(yrange):
            plt.ylim(yrange)
    #plt.title('number of trials per beta = %(num_trials)d' % {'num_trials': num_trials})
    return values
        


def plot_graph_character_correlation(centrality1=False, centrality2=False):
    '''
    Plot the correlations between main graph values - will plot a grid of correlations,
    each plot will be ordered by the value on the x-axis
    Returns the calculated centralities dictionary
    '''
    centralities = {
    }
    
    def for_all_graphs(func_type):
        values = []
        for beta in graphs:
            for graph in graphs[str(beta)]:
                if(func_type == 'degree_assortativity'):
                    value = nx.degree_assortativity_coefficient(graph)
                elif(func_type == 'fitness_assortativity'):
                    value = nx.attribute_assortativity_coefficient(graph, 'fitness')
                elif(func_type == 'laplacian_spectrum'):
                    value = nx.laplacian_spectrum(graph)[1]
                elif(func_type == 'clustering'):
                    value = nx.average_clustering(graph)
                elif(func_type == 'connectance'):
                    num_edges = len(graph.edges)
                    num_nodes = graph.node_count
                    value = num_edges / (num_nodes ** 2)
                elif(func_type == 'clustering_ratio'):
                    num_edges = len(graph.edges)
                    num_nodes = graph.node_count
                    clustering = nx.average_clustering(graph)
                    value = clustering / (num_edges / (num_nodes ** 2))
                elif(func_type == 'geodesic'):
                    value = nx.average_shortest_path_length(graph)
                elif(func_type == 'mean_degree'):
                    value = average_degree(graph)
                values.append(value)
        return values
    
    print('Gathering degree and betas')
    all_betas = []
    all_mean_degree = []
    for beta in graphs:
        for graph in graphs[str(beta)]:
            all_betas.append(beta)
            all_mean_degree.append(average_degree(graph))
    centralities['Beta'] = all_betas
    centralities['Mean Degree'] = all_mean_degree
    
    
    centrality_types = {
        'degree_assortativity': 'Degree Assortativity',
       'laplacian_spectrum': '2nd Eigenvalue',
       'clustering': 'Clustering',
       'geodesic': 'Mean Geodesic Distance'
    }
    
    for func_type in centrality_types:
        name = centrality_types[func_type]
        print('calculating values for ' + name)
        values = for_all_graphs(func_type)
        centralities[name] = values
    
    print('Plotting graphs')
    
    plt.figure(figsize=(16, 16))
    #plt.tight_layout()
     
    for i, key1 in enumerate(centralities):
        #key1 will be the y-axis
        for j, key2 in enumerate(centralities):
            plt.subplot(len(centralities), len(centralities), i * len(centralities) + j + 1)
            #plot histogram along diagonal
            if(i == j):
                pass
            else:
                x = centralities[key2]
                y = centralities[key1]
                plt.scatter(x, y, alpha=0.4)
                plt.xlabel(key2)
                plt.ylabel(key1)
    
    plt.tight_layout()
    return centralities


'''
!!
Spatial Strength Calculations
!!
'''
'''
Brainstorming ideas for graph measurements
'''
def average_edge_length(G, pbc=True, normalized=True):
    '''
    Calculate average edge length of graph
    params:
        pbc: whether pbc should be used
        normalized: normalize against the volume of the graph
    '''
    distances = pbc_distances(G, pbc=pbc)
    sum_distances = 0
    for e in G.edges:
        if(e[0] > e[1]):
            sum_distances += distances[e[1]][e[0]]
        else:
            sum_distances += distances[e[0]][e[1]]
    avg_distance = sum_distances / len(G.edges)
    if(normalized):
        volume = 1
        for size in G.size:
            volume = volume * size
        avg_distance = avg_distance / volume
    return avg_distance



def edge_distance_centrality(G, pbc=True, normalized=1):
    '''
    Note that this will double count edges, so if summing and taking the average,
        the value will most likely be different than calculating average edge length
    The ratio (+/- %) of these two should actually give a summary number for
        the distribution of edge lengths, or at least the skew of it
    Normalized:
        0: no normalization
        1: normalizes against max size of the network
        2: normalizes against max distance between nodes ("true size")
        3: normalizes against average length of edges in network and max size
        4: normalizes against average length of edges only
    '''
    distances = pbc_distances(G, pbc=pbc)
    distances = distances + distances.T
    centralities = {} #centralities[i] (before norm) is the edge length of node(v_i)

    max_length = 0
    for dimension in G.size:
        if(pbc):
            max_length = max_length + (dimension / 2) ** 2
        else:
            max_length = max_length + dimension ** 2
    max_length = np.sqrt(max_length)

    total_edge_length = 0 #total edge length of entire network
    num_edges = 0 #number of edges in the entire network

    for i in range(G.node_count):
        sum_distances = 0
        for e in G[i]:
            sum_distances += distances[i][e]
            total_edge_length += distances[i][e]
            num_edges += 1

        avg_distance = 0
        if(G.degree[i] == 0):
            avg_distance = 0
        else:
            avg_distance = sum_distances / G.degree[i]
        if(normalized == 1 or normalized == 3):
            avg_distance = avg_distance / max_length

        centralities[i] = avg_distance
    if(normalized == 3 or normalized == 4):
        if(num_edges == 0):
            average_edge_length = 0
        else:
            average_edge_length = total_edge_length / num_edges

        for i in range(G.node_count):
            if(average_edge_length == 0):
                centralities[i] = 0
            else:
                centralities[i] = centralities[i] / average_edge_length

    return centralities



def graph_edge_distance_distribution(G, pbc=True, normalized=True):
    '''
    histogram for the distance of edges
    normalize by the volume of the maximum edge distance of the network
    '''
    distances = pbc_distances(G, pbc=pbc)
    distances_array = []

    max_length = 0
    for dimension in G.size:
        if(pbc):
            max_length = max_length + (dimension / 2) ** 2
        else:
            max_length = max_length + dimension ** 2
    max_length = np.sqrt(max_length)

    for e in G.edges:
        distance = 0
        if(e[0] > e[1]):
            distance = distances[e[1]][e[0]]
        else:
            distance = distances[e[0]][e[1]]
        if(normalized):
            distance = distance / max_length
        distances_array.append(distance)
        
    sns.distplot(distances_array)



def average_neighbor_degree_centrality(G, normalized=True):
    '''
    centrality that measures the average degree of neighbors to each 
    node
    normalize by the average degree of the network (can exceed 1.0)
    '''
    centralities = {}
    avg_k = average_degree(G)
    
    for i in range(G.node_count):
        neighbor_edges = 0
        for edge in G.edges(i):
            neighbor_edges += G.degree(edge[1])
        
        if(G.degree(i) == 0):
            centralities[i] = 0
        else:
            if(normalized):
                centralities[i] = (neighbor_edges / G.degree(i)) / avg_k
            else:
                centralities[i] = neighbor_edges / G.degree(i)
    
    return centralities



def spatial_strength_centrality(G, pbc=True, normalized=4):
    '''
    centrality that attempts to measure whether spatial or topological effects are dominating
    only uses normalized
    calculates the ratio of edge distance to average neigbor degree
        - if the avg edge distance is low, spatial effects dominate
        - if the avg neighbor degree is high, topological effects dominate
    with normalized values: 
        - edge_distance takes values 0.0 - 1.0, where 1.0 is the 
            maximum length possible based on the network size
        - neighbor_degree takes values 0.0 - 1.0, where 1.0 is the number of nodes
            i.e. maximum possible degree of neighbors in a fully connected network

    spatial_strength = 1 / edge_distance * neighbor_degree
    normalized: based on edge_distances normalization    
    return:
        centralities, average_spatial_strength
    '''
    
    centralities = {
        'average_neighbor_degree': average_neighbor_degree_centrality(G),
        'edge_distance': edge_distance_centrality(G, pbc=pbc, normalized=normalized),
        'spatial_strength': {}
    }
    spatial_strength = centralities['spatial_strength']
    total_spatial_strength = 0
    for i in range(G.node_count):
        if(centralities['average_neighbor_degree'][i] == 0):
            spatial_strength[i] = 0
        else:
            spatial_strength[i] = \
                1 / (centralities['edge_distance'][i] * centralities['average_neighbor_degree'][i])
        total_spatial_strength += spatial_strength[i]
        
    average_spatial_strength = total_spatial_strength / G.node_count
    
    return centralities, average_spatial_strength
    


def spatial_strength_centrality2(G, pbc=True, normalized=True):
  '''
  calculate spatial strength centrality by dividing each edge distance
  by its corresponding degree so that we correctly weight the edges
  '''
  distances = pbc_distances(G, pbc=pbc)
  distances = distances + distances.T
  centralities = {}
  avg_k = average_degree(G)

  total_edge_length = 0
  num_edges = 0
  total_spatial_strength = 0

  for i in range(G.node_count):
    sum_distances = 0
    for e in G[i]:
      sum_distances += distances[i][e] / G.degree(e)
      total_edge_length += distances[i][e]
      num_edges += 1
    avg_distance = 0
    if(G.degree[i] != 0):
      avg_distance = sum_distances / G.degree[i]

    centralities[i] = avg_distance

  if(normalized):
    if(num_edges == 0):
      avg_edge_length = 0
    else:
      avg_edge_length = total_edge_length / num_edges

    for i in range(G.node_count):
      if(avg_edge_length == 0):
        centralities[i] = 0
      else:
        centralities[i] = centralities[i] / avg_edge_length
        total_spatial_strength += centralities[i] / avg_edge_length

  average_spatial_strength = total_spatial_strength / G.node_count

  return centralities, average_spatial_strength



def report_spatial_strength_centrality(G, pbc=True, ret=False, graph=True, print_result=True,
    normalized=4):
    centralities, average_spatial_strength = spatial_strength_centrality(G, pbc=pbc, normalized=normalized)
    c = []
    for i in range(G.node_count):
        c.append(centralities['spatial_strength'][i])
    if(graph):
        sns.distplot(c)
    if(print_result):
        print('Average Spatial Strength: ' + str(average_spatial_strength))
    if(ret):
        return centralities, average_spatial_strength



def report_spatial_strength_centrality2(G, pbc=True, ret=False, graph=True, print_result=True,
    normalized=True):
    centralities, average_spatial_strength = spatial_strength_centrality2(G, pbc=pbc, normalized=normalized)
    c = []
    for i in range(G.node_count):
      c.append(centralities[i])
    if(graph):
        sns.distplot(c)
    if(print_result):
        print('Average Spatial Strength: ' + str(average_spatial_strength))
    if(ret):
        return centralities, average_spatial_strength
