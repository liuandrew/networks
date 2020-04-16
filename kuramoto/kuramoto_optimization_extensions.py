import numpy as np
from scipy.integrate import ode
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

def ode_system_setup(omegas, connectivity):
    '''
    return a kuramoto ODE function given omegas and connectivity
    
    '''
    dim = len(omegas)
    if(connectivity.shape[0] != dim or connectivity.shape[1] != dim):
        raise Exception("Connectivity does not match dimension of omegas")
        
    def f(t, y):
        '''
        note that the np.matmul(connectivity.T, S) only has entries in the diagonal
        so it might make sense later to convert this to a for loop to decrease memory
        costs
        '''
        S = np.zeros((dim, dim))
        for n in range(dim):
            S[n] = y[n]
        S = np.sin(S - S.T)
        coupling = np.matmul(connectivity.T, S)
        return np.diag(coupling) + omegas
        
    return f





def compute_phasic_average(res, partition, split=True, interval=False):
    '''
    compute the vector average of phase results based on partition
    partition should be given as a list of nodes
    if split, then give as a two column array with real and imaginary
    if interval, then only compute over given time interval
        interval should be given as a list/tuple such as (None, 500) or (100, None)
        noting None means to start from beginning or go to end
    '''
    if(interval is not False):
        num_points = len(res.y[0][interval[0] : interval[1]])
    else:
        num_points = len(res.y[0])
    vector_sum = np.zeros(num_points)
    for n in partition:
        if(interval is not False):
            vector_sum = vector_sum + np.exp(res.y[n][interval[0] : interval[1]] * 1j)
        else:
            vector_sum = vector_sum + np.exp(res.y[n] * 1j)
    #note that vector sum becomes average, don't change name to avoid memory allocation
    vector_sum = vector_sum / len(partition)
    if(split):
        return np.array([np.real(vector_sum), np.imag(vector_sum)])
    else:
        return vector_sum




    
def compute_phasic_vector_strength(phasic_avgs, split=True):
    '''
    compute the average vector strength of the phasic difference
    if split is true, we expect the phasic averages array to be split
    '''
    mean_vector_strength = 0
    if(split):
        mean_vector_strength = np.mean(np.abs(phasic_avgs[0] + 1j * phasic_avgs[1]))
    else:
        mean_vector_strength = np.mean(np.abs(phasic_avgs))
    return mean_vector_strength





def compute_pvs(res, partition, interval=False):
    '''
    directly compute average vector strength of partition
    '''
    vector_avg = compute_phasic_average(res, partition, interval=interval)
    return compute_phasic_vector_strength(vector_avg)





def undirected_graph_laplacian(connectivity):
    '''
    get the graph laplacian
    '''
    degrees = np.sum(connectivity, axis=1)
    connectivity = -1 * connectivity
    for n in range(len(degrees)):
        connectivity[n, n] = degrees[n]
    return connectivity






def erdos_renyi_matrix(n, p):
    '''
    random connectivity matrix for n p erdos renyi
    '''
    connectivity = np.random.uniform(0, 1, (n, n))
    connectivity = (connectivity < p) * 1
    np.fill_diagonal(connectivity, 0)
    connectivity = np.triu(connectivity)
    connectivity = connectivity + connectivity.T
    return connectivity





def signed_edge_matrix(connectivity, directed=True):
    '''
    get the signed edge matrix (N matrix, n x m where n number of nodes,
    m number of edges) based on connectivity matrix
    '''
    if(not directed):
        a2 = np.triu(connectivity)
    else:
        a2 = (connectivity != 0) * 1
    N = np.zeros((connectivity.shape[0], np.sum(a2)))
    edge_counter = 0
    for i in range(connectivity.shape[0]):
        for j in range(connectivity.shape[1]):
            if(a2[i,j] != 0):
                edge = np.zeros(connectivity.shape[0])
                if(i > j):
                    edge[i] = 1
                    edge[j] = -1
                else:
                    edge[i] = -1
                    edge[j] = 1
                N.T[edge_counter] = edge
                edge_counter = edge_counter + 1
    return N
            




def weighted_directed_laplacian(connectivity):
    '''
    get the weighted graph laplacian
    '''
    N = signed_edge_matrix(connectivity)
    m = np.sum((connectivity != 0) * 1)
    W = np.zeros((m, m))
    edge_counter = 0
    for i in range(connectivity.shape[0]):
        for j in range(connectivity.shape[1]):
            if(connectivity[i, j] != 0):
                W[edge_counter, edge_counter] = connectivity[i, j]
                edge_counter = edge_counter + 1
    return np.matmul(np.matmul(N, W), N.T)
    




'''
greedy optimization procedure from random partition, 
using both growing or shrinking partition
'''

def random_partition(n, p=0.5, force=False):
    '''
    get a uniform random partition over n nodes with probability p
    of nodes being in partition
    if force is given as a number, force a number of nodes
        additionally, will automatically calculate the best p to use for force
    '''
    if(force is not False):
        p = force / n
        partition = np.arange(0, n)
        while(len(partition) != force):
            partition = np.arange(0, n)
            partition = partition[np.random.uniform(0, 1, n) < p]
    else:
        partition = np.arange(0, n)
        partition = partition[np.random.uniform(0, 1, n) < p]
    return partition




def partition_remainder(n, partition):
    '''
    given a partition, get the remaining nodes
    '''
    temp = np.arange(n)
    new_part = np.array([])
    for n in temp:
        if(n not in partition):
            new_part = np.append(new_part, n)
    return np.array(new_part, int)





def random_greedy_optimization(res, n=50, p=0.5, partition=False, interval=False, verbose=0):
    '''
    perform a random greedy optimization at each step adding or removing
    a node from the partition which maximizes compute_pvs
    verbose: 1 - start and end vector strengths with number of changes
             2 - each change and vector strength at each step
    partition: give a starting partition
    interval: set interval to calculate pvs over
    '''
    if(partition is False):
        partition = random_partition(n, p)
    print(partition)
    max_vector_strength = compute_pvs(res, partition, interval=interval)
    all_nodes = np.arange(0, n)
    optimizing = True
    if(verbose > 0):
        print('original vector strength: ' + str(max_vector_strength))
    number_of_changes = 0

    while(optimizing):
        updated_this_round = False
        change = 'None'
        new_partition = None
        for n in all_nodes:
            if(n in partition):
                index = np.argwhere(partition == n)[0][0]
                test_part = np.append(partition[:index], partition[index+1:])
            else:
                test_part = np.append(partition, n)

            new_strength = compute_pvs(res, test_part, interval=interval)
            if(new_strength > max_vector_strength):
                max_vector_strength = new_strength
                new_partition = test_part
                updated_this_round = True
                if(n in partition):
                    change = 'Remove ' + str(n)
                else:
                    change = 'Add ' + str(n)
                            
        if(not updated_this_round):
            optimizing = False
        else:
            partition = new_partition
            if(verbose > 1):
                print(change)
                print('New vector strength: ' + str(max_vector_strength))
        number_of_changes = number_of_changes + 1

    if(verbose > 0):
        print('number of changes: ' + str(number_of_changes))
        print('final vector strength: ' + str(max_vector_strength))
    
    return partition, max_vector_strength, number_of_changes





def reverse_greedy_optimization(res, start, final_count, n=50, interval=False, verbose=0):
    '''
    perform greedy optimization of pvs adding nodes starting from n
    until final_count is reached
    '''
    all_nodes = np.arange(0, n)
    partition = np.array([start])
    node_history = []
    strength_history = []
    while(len(partition) < final_count):
        max_vector_strength = 0
        new_partition = None
        node_to_add = None
        for n in all_nodes:
            if(n not in partition):
                test_part = np.append(partition, n)
                new_strength = compute_pvs(res, test_part, interval=interval)
                if(new_strength > max_vector_strength):
                    max_vector_strength = new_strength
                    new_partition = test_part
                    node_to_add = n
        
        partition = new_partition
        node_history.append(node_to_add)
        strength_history.append(max_vector_strength)
        
        if(verbose > 1):
            print('Add ' + str(node_to_add))
            print('New vector strength: ' + str(max_vector_strength))
    if(verbose > 0):
        print('Final vector strength: ' + str(max_vector_strength))
        print('Final partition: ' + str(partition))
    return partition, max_vector_strength, strength_history, node_history
            