import torch
import gudhi
import numpy as np 
from scipy.sparse import csc_matrix, csgraph
from get_embeddings import get_activation, get_pairwise_distance


def rips_complex_to_sparse_matrix(distance_matrix, max_edge_length):
    '''
    '''
    check1 = (distance_matrix.dim() == 2) and torch.all(torch.abs(distance_matrix.transpose(0, 1) - distance_matrix) < 1e-5)
    err_dist_mat = f"Input matrix is not a distance matrix - it is either not two-dimensional, or not symmetric"
    assert check1, err_dist_mat
    
    # Create RipsComplex
    rips_complex = gudhi.RipsComplex(distance_matrix=distance_matrix, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=1)

    # Extract edges and weights
    edges = []
    weights = []
    for simplex in simplex_tree.get_filtration():
        if len(simplex[0]) == 2:  # Only consider edges (2-simplexes)
            edges.append([simplex[0][0], simplex[0][1]])
            weights.append(1)

    # Create adjacency matrix
    #num_points = len(points)
    adjacency_matrix = np.zeros(distance_matrix.shape)
    for i, j in edges:
        adjacency_matrix[i, j] = weights[edges.index([i, j])]
        adjacency_matrix[j, i] = weights[edges.index([i, j])]

    # Compute graph Laplacian
    laplacian_matrix = csgraph.laplacian(adjacency_matrix, normed=True)

    return laplacian_matrix



def extract_discrete_laplacian_matrix(x, model, layer, max_dist):
    
    act1 = get_activation(model, x, layer)
    
    laplacian_matrix = rips_complex_to_sparse_matrix(get_pairwise_distance(act1), max_dist)
    
    return laplacian_matrix



def extract_persistent_laplacian_target(x, y, model, layer, min_dist, max_dist, n):

    dist_scales = np.arange(min_dist, max_dist, (max_dist - min_dist)/n)
    
    res = []
    for dist_val in dist_scales:
        laplacian_matrix = extract_discrete_laplacian_matrix(x, model, layer, dist_val)
        laplacian_val = torch.Tensor(laplacian_matrix @ np.array(y))
        res.append(torch.sqrt(torch.square(laplacian_val).mean()))
    
    return dist_scales, res
