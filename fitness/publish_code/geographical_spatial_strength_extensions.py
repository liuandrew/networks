'''
This code was used in the production of "Spatial strength centrality and the 
effect of spatial embeddings on network architecture", Andrew Liu and Mason A. 
Porter 2020, Physical Review E.
'''

'''
This file contains functions used to create each of the 3 types of networks
in the paper, as well as functions to graph the spatial networks and compute
spatial strength centrality

We use the Python package NetworkX as the basis for creating networks
You may also need to install the seaborn package for some of the plotting functions

Both of these packages can be installed if you have pip via
  $ pip install networkx seaborn

==========================
Key functions in this file
==========================
gf_graph_model() - returns a geographical fitness network
spa_graph_model() - returns a spatial preferential attachment network
sc_graph_model() - returns a spatial configuration network
    (the seed network is a BA graph)
graph_spatial_network_plot() - plots a spatial graph in 2D
spatial_strength_centrality() - computes the spatial strength of a network
'''


import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import math

from base_network_extensions import *


'''
================
Model generators
================
These functions generate the specific network models desired
The key functions are
'''




'''
----------------------------
Geographical Fitness network
----------------------------
The function gf_graph_model() returns a geographical fitness network
We have to pass it a beta value that was in the paper (beta=0, 0.5, 1, 
1.5, 2, 2.5, 3), note that the values here are negative
If any of these are passed, the function will automatically use a saved value
from target_threshold.pickle file for the threshold (theta), such that 
<k> is roughly equal to 20. Otherwise, a value for threshold must be given manually
'''


'''
Load target_thresholds
'''
target_thresholds = pickle.load(open("target_thresholds.pickle", 'rb'))

def gf_graph_model(n=500, beta=1.0, size=[100, 100], threshold=False, keep_edge_distances=False):
    '''
    Generate a geographical fitness model

    params:
      n: number of nodes
      beta: spatial decay parameter
      size: size of spatial embedding
      threshold: theta value - threshold beyond which edges are formed
      keep_edge_distances: whether to save the lengths of edges in the graph

    If no threshold is specified, we will look for the threshold found for n=500 that gives
        mean degree 20
    '''
    beta = beta * -1

    if(not threshold):
        threshold = target_thresholds[beta]
    
    g = SimpleSpatialGraph(size=size)
    add_normal_nodes(g, n)

    add_vectorized_eval_edges(g, dimensions=size, threshold=threshold, beta=beta, keep_edge_distances=keep_edge_distances)
    return g





def add_vectorized_eval_edges(G, dimensions=[1, 1], threshold=0.05, beta=-1.5, keep_edge_distances=False):
    '''
    Add edges between each pair of nodes in a SimpleSpatialGraph with normally distributed fitness values
    
    params:
        threshold: Heavistep threshold for edge to form (theta in paper)
        beta: Power factor for distance
        c: scalar for distance (NOTE: removed this parameter as it is absorbed by threshold)
        keep_edge_distances: boolean that will store edge distances as edge attributes. Increases memory usage for
            the graph, but necessary for closeness centrality calculation
    '''
    coordinates = []
    fitnesses = []
    for i in G.nodes:
        coordinates.append(G.nodes[i]['coordinate'])
        fitnesses.append(G.nodes[i]['fitness'])
    
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
-------------------------------
Spatial Preferential Attachment
-------------------------------
The function spa_graph_model() returns an SPA network
'''

def spa_graph_model(m0=10, m=5, n=500, beta=1.0, verbose=False, pbc=True):
    '''
    Generate spatial preferential attachment network

    params:
      m0: how many nodes in seed network (these are fully connected)
      m: how many edges each new node forms
      n: how many total nodes to add
      beta: spatial decay parameter
      verbose: whether to print the beta value
      pbc: whether we should use periodic boundary conditions
    '''
    beta = beta * -1

    g = SimpleSpatialGraph(size=[100, 100])
    
    g.total_degree = (m0 * (m0 - 1)) / 2
    g.node_labels = []
    #create initial nodes
    for i in range(m0):
        g.add_node()

    #connect initial nodes
    for i in range(g.node_count):
        for j in range(i+1, g.node_count):
            g.add_edge(i, j)
    
    for i in range(g.node_count):
        distances = pbc_distances_single(g, node=i)
        for j in range(i+1, g.node_count):
            g[i][j]['distance'] = distances[j]

    if(verbose):
        print(beta)
        
    for i in range(n):
        new_node = g.node_count
        g.add_node()

        edges_added = 0
        p, distances = generate_weighted_distance_p(g, beta=beta)
        
        #counter to see if we should remove extreme probability nodes for efficiency
        potential_edges_selected = 0
        while(edges_added < m):
            potential_edges_selected += 1
            deg_num = np.random.randint(0, g.total_degree)
            node_edge = np.random.choice(np.arange(0, g.node_count), p=p)

            if(potential_edges_selected > 10):
                #after more than 10 trials reset p
                potential_edges_selected = 1
                p, distances = generate_weighted_distance_p(g, beta=beta, 
                                                  ignore_edges=g.edges(new_node), pbc=pbc)
            
            if(not g.has_edge(new_node, node_edge)):
                g.add_edge(new_node, node_edge)
                g[new_node][node_edge]['distance'] = distances[node_edge]
                edges_added += 1

    return g




def generate_weighted_distance_p(G, node=False, beta=0.0, ignore_edges=[], pbc=True):
    '''
    Generate weighted p to select nodes for spatial PA model
    i.e. give each node a probability of connection weighted by distance and degree
      for SPA model to pick edges from

    node: if a specific node label is used, compute the values of p for a specific node
    beta: spatial decay parameter
    ignore_edges: a list of edges to ignore (likely because prob is too high),
        taken from g.edges(node)
    pbc: use periodic boundary conditions
    '''
    if(node is False):
        node = G.node_count - 1
    distances = pbc_distances_single(G, node=node, pbc=pbc)
    distances_beta = distances ** beta
    distances_beta[node] = 0
    for edge in ignore_edges:
        distances_beta[edge[1]] = 0
    degrees = np.array(G.degree).T[1]
    weighted_degree = degrees * distances_beta
    
    total_weighted_degree = np.sum(weighted_degree)
    weighted_probability = weighted_degree / total_weighted_degree
    
    return weighted_probability, distances




'''
---------------------------
Spatial Configuration Model
---------------------------
sc_graph_model() returns a spatial configuration of a BA graph with 
n nodes and 5 edges added per new node
'''

def sc_graph_model(n=1000, beta=1.0):
    beta = beta * -1

    g = nx.barabasi_albert_graph(n, 5)
    g.node_count = n
    g.dimension = 2
    g.size = [100, 100]
    g = spatial_configuration(g, beta=beta)
    return g





def spatial_configuration(G, beta=0.0, pbc=True, verbose=False):
    '''
    perform spatial configuration model on the given graph
    i.e., given a network, we reassign each node a position and then
      connect edge stubs weighted by distance

    a copy of the graph will be made so the original is not modified
        warning - this correct add_node functionality since it returns a nx.Graph copy 
        and not a SimpleSpatialGraph
    

    create a dictionary of the number of stubs each node has based on its degree
    remove all edges
    shuffle positions of the nodes
    at each step, choose a stub at random, then another based spatially
    params:
        beta: determines strength of distance effects -> negative means stronger,
            0.0 implies no distance effects and original configuration model
            is recovered
        verbose: prints every 1000 stubs processed
    NOTE: allows multiedges
    '''
    g = G.copy()
    g.node_count = G.node_count
    g.size = G.size
    g.dimension = G.dimension
    
    # get initial stub conditions
    stubs = np.array(list(g.degree)).T[1]
    total_stubs_to_assign = np.sum(stubs)
    stub_weighting = 1 / total_stubs_to_assign
    normalized_stubs = stubs * stub_weighting
    
    #shuffle positions and calculate new distances
    shuffle_positions(g)
    distances = pbc_distances(g, pbc=pbc)
    
    #remove existing edges
    edges = []
    for e in g.edges:
        edges.append(e)
    for e in edges:
        g.remove_edge(e[0], e[1])
    
    while(total_stubs_to_assign > 0):
        try:
            stub_1 = select_from_list_with_multiplicity(normalized_stubs)
        except:
            print(exception)
            return (normalized_stubs, stubs)
        
        #weight distances of remaining stubs by distance from first stub
        d1 = distances.T[stub_1]
        d2 = distances[stub_1]
        d3 = d1 + d2
        weighted_distances = d3 ** beta
        weighted_distances[stub_1] = 0
        weighted_stubs = normalized_stubs * weighted_distances
        stub_2 = select_from_list_with_multiplicity(weighted_stubs)
        g.add_edge(stub_1, stub_2)
        
        stubs[stub_1] = stubs[stub_1] - 1
        stubs[stub_2] = stubs[stub_2] - 1
        normalized_stubs[stub_1] = normalized_stubs[stub_1] - stub_weighting
        normalized_stubs[stub_2] = normalized_stubs[stub_2] - stub_weighting
        if(stubs[stub_1] == 0):
            normalized_stubs[stub_1] = 0
        if(stubs[stub_2] == 0):
            normalized_stubs[stub_2] = 0

        total_stubs_to_assign = total_stubs_to_assign - 2
        if(verbose and total_stubs_to_assign % 1000 < 2):
            print(total_stubs_to_assign)
    return g




def shuffle_positions(G, method='uniform'):
    '''
    replace node positions in space randomly based on method
    params:
        method: type of random
            'uniform': uniform at random
            'exponential': exponentially from center
    '''
    for i in range(G.node_count):
        coordinate = []
        if(method is 'uniform'):
            for j in range(len(G.size)):
                coordinate.append(np.random.uniform(0, G.size[j]))
        if(method is 'exponential'):
            for j in range(len(G.size)):
                pass
        G.nodes[i]['coordinate'] = coordinate





def select_from_list_with_multiplicity(l):
    '''
    Uniformly at random select from a weighted list and return an index
    '''
    a = np.array(l)
    a = a / np.sum(a)
    selection = np.random.choice(np.arange(0, len(a)), p=a)
    return selection






'''
===============================================
Graph the spatial network and plot centralities
===============================================
'''
def graph_spatial_network_plot(G, values=False, graph_edges=True, hide_ticks=True, subplot=False,
    highlight_edges=False, highlight_recent=False, color_points=False, color_bar=False, bounds=False, pbc=True,
    edge_style=None, ss_title=False, beta=None, alpha=0.2, linewidth=2):
    '''
    This function creates a plot of the passed SimpleSpatialGraph
    The parameters passed will change how the plot looks, or ask it the function
        to compute things in different ways

    params:
        *important params
        G: passed in SimpleSpatialGraph
        pbc: whether to graph with periodic boundary conditions, default True

        *params to change visual style of graph (these ones are used in the paper)
        alpha: how dark should the edges be (0 - invisible, 1 - opaque)
        linewidth: how wide should the lines representing edges be drawn
        ss_title: if this is set to True, compute the mean spatial strength centrality
            and write it as a title

        *params to change visual style of graph (these ones don't get used in paper)
        values: if we choose to pass in a list of values, we can ask the function to
            color the points based on the value (set color_points, color_bar to True)
        graph_edges: plot edges of the graph
        highlight_edges: we can choose to pass in a list of tuples to ask the function
            to highlight these edges in red, e.g. highlight_edges=[[0, 1], [3, 6]]
        color_points: array of color labels if coloring points specifically
            look at matplotlib's scatter documentation to see how this should be passed
        bounds: given by list of lists. 
            e.g. if we wanted to look at [0.3, 0.5] x [0.1, 0.2], pass in
            bounds = [[0.3, 0.5], [0.1, 0.2]]
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
        title = r'$\langle S \rangle \approx {0:.3f}$'.format(spatial_strength)
        if(beta):
                title += r', $\beta$ = '
                title += str(beta)
        plt.title(title, y=1.02)
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
                        plt.plot(points[0], points[2], alpha=alpha, linewidth=linewidth, color='grey')
                        plt.plot(points[1], points[3], alpha=alpha, linewidth=linewidth, color='grey')
                    else:
                        x = (coordinate_i[0], coordinate_j[0])
                        y = (coordinate_i[1], coordinate_j[1])
                        plt.plot(x, y, alpha=alpha, linewidth=linewidth, color='grey')
        else:
                for i in G.edges:
                    coordinate_i = G.nodes[i[0]]['coordinate']
                    coordinate_j = G.nodes[i[1]]['coordinate']
                    x = (coordinate_i[0], coordinate_j[0])
                    y = (coordinate_i[1], coordinate_j[1])
                    plt.plot(x, y, color='grey', alpha=alpha, linewidth=linewidth)

    if(highlight_edges):
        for i in highlight_edges:
            coordinate_i = G.nodes[i[0]]['coordinate']
            coordinate_j = G.nodes[i[1]]['coordinate']
            x = (coordinate_i[0], coordinate_j[0])
            y = (coordinate_i[1], coordinate_j[1])
            plt.plot(x, y, alpha=0.2, color='red')

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
    
    


'''
========================

Analyze graphs in a graphs dictionary based on betas in betas list
========================
'''
def analyze_graphs_with_function(func_type, verbose=0, xlabel=False, ylabel=False, num_trials=30, 
                                 subplot=None, betas=None, individual_trials=False, values=False,
                                invert_betas=True, yticks=None, yrange=None, label=False, graphs=False):
    '''
    Analyze the set of graphs using the given func_type
    params:
        func_type: what centrality to compute and plot, use one of predetermined strings:
            'degree_assortativity'
            'fitness_assortativity' (fitness assortativity coefficient)
            'laplacian_spectrum' (value of the second eigenvalue)
            'clustering'
            'geodesic'
            'mean_degree'
            'connectance'
            'average_edge_length'
            'average_spatial_strength'
        verbose:
            0: only print betas in line
            1: print per beta
            2: print per trial
        subplot: pass 3 index list for subplot placement in matplotlib
        graphs: dictionary of graphs to analyze, keyed by value of beta
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
        'clustering': r'Mean local clustering coefficient',
        'geodesic': r'Mean geodesic distance',
        'connectance': r'Connectance',
        'clustering_ratio': r'Clustering Ratio, (C / (N/E^2))',
        'mean_degree': r'Mean degree, \langle k \rangle',
        'average_edge_length': r'Mean edge length',
        'average_spatial_strength': r'Mean spatial strength'
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
                        value = 0
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
        subplot.set_ylabel(centrality_types[func_type], labelpad=15)
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
        
        

'''
=============================
Spatial Strength Calculations
=============================
These functions are used to compute the spatial strength centralities of a
network
'''


def spatial_strength_centrality(G, pbc=True, normalized=4):
    '''
    centrality that attempts to measure whether spatial or topological effects are dominating
    
    calculates the ratio of edge distance to average neigbor degree
        - if the avg edge distance is low, spatial effects dominate
        - if the avg neighbor degree is high, topological effects dominate
    with normalized values: 
        - edge_distance takes values 0.0 - 1.0, where 1.0 is the 
            maximum length possible based on the network size
        - neighbor_degree takes values 0.0 - 1.0, where 1.0 is the number of nodes
            i.e. maximum possible degree of neighbors in a fully connected network

    spatial_strength = 1 / edge_distance * neighbor_degree
    normalized: what things to normalize - look at documentation for the
        edge_distance_centrality() function to see what each normalized value means
        in the paper we use normalized=4
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
    missed_nodes = 0
    for i in range(G.node_count):
        if(centralities['average_neighbor_degree'][i] == 0):
            spatial_strength[i] = 0
            missed_nodes += 1
        else:
            spatial_strength[i] = \
                1 / (centralities['edge_distance'][i] * centralities['average_neighbor_degree'][i])
        total_spatial_strength += spatial_strength[i]
        
    average_spatial_strength = total_spatial_strength / (G.node_count - missed_nodes)
    
    return centralities, average_spatial_strength
    



def report_spatial_strength_centrality(G, pbc=True, ret=False, graph=True, print_result=True,
    normalized=4):
    '''
    This is a quick function to calculate and plot the spatial strength centrality of a network
    Or just print out the value

    ret: whether to return spatial strength centralities and mean spatial strength centrality
    pbc: whether to use periodic boundary conditions
    graph: whether to graph the distribution of spatial strength centralities
    print_result: whether to print the average spatial strength value
    '''
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

