import numpy as np
import torch
from torch.autograd import Variable
from model import GCMC
from loss import Loss

# Run on GPU if CUDA is available
RUN_ON_GPU = torch.cuda.is_available()

# Set random seeds
SEED = 2019
np.random.seed(SEED)
torch.manual_seed(SEED)
if RUN_ON_GPU:
    torch.cuda.manual_seed(SEED)

def np_to_var(x):
    """
    Convert numpy array to Torch variable.
    """
    x = torch.from_numpy(x)
    if RUN_ON_GPU:
        x = x.cuda()
    return Variable(x)

def var_to_np(x):
    """
    Convert Torch variable to numpy array.
    """
    if RUN_ON_GPU:
        x = x.cpu()
    return x.data.numpy()

def to_sparse(x):
    """ converts dense tensor x to sparse format """
    x_typename = torch.typename(x).split('.')[-1]
    sparse_tensortype = getattr(torch.sparse, x_typename)

    indices = torch.nonzero(x)
    if len(indices.shape) == 0:  # if all elements are zeros
        return sparse_tensortype(*x.shape)
    indices = indices.t()
    values = x[tuple(indices[i] for i in range(indices.shape[0]))]
    return sparse_tensortype(indices, values, x.size())

def normalize(M):
    s = np.sum(M, axis=1)
    s[s == 0] = 1
    return (M.T / s).T

def create_models(feature_q, feature_i, feature_t, feature_dim, hidden_dim, M_qi, M_it, out_dim, drop_out=0.0):
    """
    Initialize the GCMC model for a tripartite query-item-tag graph.
    """
    feature_q = to_sparse(np_to_var(feature_q.astype(np.float32)))
    feature_i = to_sparse(np_to_var(feature_i.astype(np.float32)))
    feature_t = to_sparse(np_to_var(feature_t.astype(np.float32)))

    M_qi = to_sparse(np_to_var(M_qi.astype(np.float32)))
    M_it = to_sparse(np_to_var(M_it.astype(np.float32)))

    net = GCMC(feature_q, feature_i, feature_t, feature_dim, hidden_dim, M_qi, M_it, out_dim, drop_out)

    if RUN_ON_GPU:
        print('Moving models to GPU.')
        net.cuda()
    else:
        print('Keeping models on CPU.')

    return net

def epsilon_similarity_graph(X: np.ndarray, sigma=None, epsilon=0):
    """ X (n x d): coordinates of the n data points in R^d.
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    W = np.array([np.sum((X[i] - X)**2, axis=1) for i in range(X.shape[0])])
    typical_dist = np.mean(np.sqrt(W))
    c = 0.35
    if sigma is None:
        sigma = typical_dist * c
    
    mask = W >= epsilon
    
    adjacency = np.exp(- W / 2.0 / (sigma ** 2))
    adjacency[mask] = 0.0
    adjacency -= np.diag(np.diag(adjacency))
    return adjacency

def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    d = np.sum(adjacency, axis=1)
    d_sqrt = np.sqrt(d)
    D = np.diag(1 / d_sqrt)
    if normalize:
        L = np.eye(adjacency.shape[0]) - (adjacency.T / d_sqrt).T / d_sqrt
    else:
        L = np.diag(d) - adjacency
    return L

def initialize_loss(mask_qi, mask_it, laplacian_loss_weight):
    mask_qi = np_to_var(mask_qi.astype(np.float32))
    mask_it = np_to_var(mask_it.astype(np.float32))
    return Loss(mask_qi, mask_it, laplacian_loss_weight)