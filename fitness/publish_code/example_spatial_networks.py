'''
This code was used in the production of "Spatial strength centrality and the 
effect of spatial embeddings on network architecture", Andrew Liu and Mason A. 
Porter 2020, Physical Review E.
'''

'''
This file contains functions to generate certain toy networks and load
networks from fungal/street datasets

We use the Python package NetworkX as the basis for creating networks
You may also need to install the seaborn package for some of the plotting functions

Both of these packages can be installed if you have pip via
  $ pip install networkx seaborn
'''

from geographical_spatial_strength_extensions import *
import scipy.io as sio

'''
=========================
Generate Example Networks
=========================
These functions generate the example networks found in Section V C of the paper
'''

def generate_rgg(n, r, size=1, pbc=False):
    '''
    generate a random geometric graph where and edge exists between nodes
    if they are within a distance of r from each other
    '''
    G = SimpleSpatialGraph(size=[size, size])
    for _ in range(n):
        G.add_node()
        
    distances = pbc_distances(G, pbc=pbc)
    
    for i in range(G.node_count):
        for j in range(i+1, G.node_count):
            if(distances[i][j] < r*size):
                G.add_edge(i, j)
                G[i][j]['distance'] = distances[i][j]
                
    return G




def generate_lattice(rows=5, columns=5, size=1):
    '''
    generate a lattice graph defined by number of rows and columns where
    number of nodes = rows * columns, and nodes that are north, south,
    east, west of each other will be adjacent
    '''
    G = SimpleSpatialGraph(size=[size, size])
    row_step = size / rows
    column_step = size / columns
    
    #create nodes
    for i in range(rows):
        for j in range(columns):
            G.add_node(coordinate=[j * column_step + 0.5 * column_step, i * row_step + 0.5 * row_step])
    
    #create edges
    #for each node, add edges to the right and above
    distances = []
    for i in range(rows):
        for j in range(columns):
            node_num = i * columns + j
            #edge above
            if(i < rows - 1):
                G.add_edge(node_num, node_num + columns)
            #edge to the right
            if(j < columns - 1):
                G.add_edge(node_num, node_num + 1)
    return G
    



def generate_web(branches=3, layers=5, radius=1):
    '''
    generate a spatial cayley tree (layered star network)
    branches: branches per layer
    layers: number of layers
    radius: radius per layer
    '''
    G = SimpleSpatialGraph(size=[2 * radius * layers, 2 * radius * layers])
    center = [radius * layers, radius * layers]
    G.add_node(coordinate=center)
    backstep = math.floor(branches / 2)
    
    append_to_node = 0
    branch_count = 0
    current_node = 0
    add_to_label = 1
        
    for n in range(1, layers):
        theta_step = 2 * math.pi / (branches ** n) 
        for i in range(-backstep, branches ** n - backstep):
            branch_count += 1
            current_node += 1
            
            theta = theta_step * i
            x = math.cos(theta) * n * radius + center[0]
            y = math.sin(theta) * n * radius + center[1]
            label = add_to_label + (i % branches ** n)
            G.add_node(label=label, coordinate=[x, y])
            G.add_edge(append_to_node, label)
            
            if(branch_count == branches):
                branch_count = 0
                append_to_node += 1
                
        add_to_label += branches ** n
    
    return G



def generate_breaking_example():
    '''
    Generate the network example that gives rise to arbitrarily large 
    mean spatial strength centrality
    '''
    g = SimpleSpatialGraph(size=[100, 6])
    g.add_node(coordinate=[5,2])
    g.add_node(coordinate=[95, 2])
    g.add_node(coordinate=[48.5, 4])
    g.add_node(coordinate=[51.5, 4])
    g.add_edge(0, 1)
    g.add_edge(2, 3)
    return g





def generate_hub(spoke_length=0.25):
    '''
    Generate the hub network example with a given spoke length
    Lengths used in the paper are 0.5 and 0.25
    '''
    g = SimpleSpatialGraph(size=[8, 8])
    hubs = [[1, 2], [5, 7], [7, 4]]
    hub_labels = []
    for coord in hubs:
        hub = g.node_count
        hub_labels.append(hub)
        g.add_node(coordinate=coord)
        for i in range(10):
            coord2 = [coord[0] + spoke_length * math.cos(2 * math.pi * (i / 10)), 
                      coord[1] + spoke_length * math.sin(2 * math.pi * (i / 10))]
            g.add_node(coordinate=coord2)
            g.add_edge(g.node_count - 1, hub)
    for i in range(3):
        for j in range(i, 3):
            g.add_edge(hub_labels[i], hub_labels[j])

    return g





'''
==================
Load Data Networks
==================
These functions generate SimpleSpatialGraphs from the data from
Fungal Networks: S.H.Lee, M.D.Fricker, M.A.Porter 2017
Street Networks: S.H.Lee, P. Holme 2012

Note that this repository does not contain the complete sets of data
from these papers, but if you download them and add them to their appropriate
folders, they should work
'''

def load_fungal_network(filename):
    '''
    Load the fungal network with the given filename in the data folder
    '''
    data = sio.loadmat(open('datasets/fungal_data/' + filename, 'rb'))
    xmin = np.min(data['coordinates'].T[0])
    xmax = np.max(data['coordinates'].T[0])
    ymin = np.min(data['coordinates'].T[1])
    ymax = np.max(data['coordinates'].T[1])
    g = SimpleSpatialGraph(size=[xmax-xmin, ymax-ymin])
    for coord in data['coordinates']:
        coord = [coord[0] - xmin, coord[1] - ymin]
        g.add_node(coordinate=coord)
    for node in range(g.node_count):
        for node2, strength in enumerate(data['A'][node].toarray()[0]):
            if(strength > 0):
                g.add_edge(node, node2)
    return g







def load_city_network(continent, city):
    '''
    Load the street network in the given continent of the given city
    '''
    node_data = open('datasets/road_data_2km/' + continent + '_2km/' + city + '_networkx_unit_node.txt', 'r')
    edge_data = open('datasets/road_data_2km/' + continent + '_2km/' + city + '_networkx_unit_edge.txt', 'r')
    labels = []
    coordsx = []
    coordsy = []
    for line in node_data:
        data = line.split('\n')[0].split(' ')
        labels.append(data[0])
        coordsx.append(float(data[1]))
        coordsy.append(float(data[2]))
    xmin = np.min(coordsx)
    xmax = np.max(coordsx)
    ymin = np.min(coordsy)
    ymax = np.max(coordsy)
    coordsx = np.subtract(coordsx, xmin)
    coordsy = np.subtract(coordsy, ymin)
    g = SimpleSpatialGraph(size=[xmax-xmin, ymax-ymin])
    for i in range(len(labels)):
        g.add_node(coordinate=[coordsx[i], coordsy[i]])

    for line in edge_data:
        data = line.split('\n')[0].split(' ')
        g.add_edge(labels.index(data[0]), labels.index(data[1]))
    
    return g