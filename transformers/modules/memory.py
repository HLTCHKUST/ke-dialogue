from logging import getLogger
import math
import itertools
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

EPSILON = 1e-10

###########
# Utility Functions
###########
def get_gaussian_keys(n_keys, depth, dim, normalized, seed):
    """
    Generate random Gaussian keys.
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_keys, depth, dim)
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)

def get_uniform_keys(n_keys, depth, dim, normalized, seed):
    """
    Generate random uniform keys (same initialization as nn.Linear).
    """
    rng = np.random.RandomState(seed)
    bound = 1 / math.sqrt(dim)
    X = rng.uniform(-bound, bound, (n_keys, depth, dim))
    if normalized:
        X /= np.linalg.norm(X, axis=1, keepdims=True)
    return X.astype(np.float32)


def create_keys(n_keys, depth, k_dim, keys_init, keys_normalized_init, random_seed):
    """
    This function creates keys and returns them.
    I guess you could see that from the name of the function and the fact that is has a return statement.
    """
    # random keys from Gaussian or uniform distributions
    init = get_gaussian_keys if keys_init == 'gaussian' else get_uniform_keys
    keys = torch.from_numpy(init(n_keys, depth, k_dim, keys_normalized_init, seed=random_seed))
    return keys
    
def mlp(input_size, output_size, bias=True, batchnorm=True):
    """
    Generate a feedforward neural network.
    """
    layers = []
    layers.append(nn.Linear(input_size, output_size, bias=bias))
    if batchnorm:
        layers.append(nn.BatchNorm1d(output_size))
    layers.append(nn.ReLU())
    return nn.Sequential(*layers)

###########
# Model funciton
###########
class HashingMemory(nn.Module):
    _ids = itertools.count(0)

    def __init__(self, input_dim, num_memory, args):
        super().__init__()
        self.id = next(self._ids)
        
        # For testing purpose
        self.retrieve_indices = None

        # global parameters
        self.keys_normalized_init = True # Normalize key matrix during initialization
        self.input_dim = input_dim
        self.num_memory = num_memory # Number of knowledge in the knowledge
        self.random_seed = args['random_seed']
        self.depth = args['mem_depth']
        self.k_dim = args['mem_k_dim']
        self.topk = args['mem_topk']
        self.keys_init = args['mem_keys_init']

        # scoring / re-scoring
        self.temperature = args['mem_temperature']
        self.score_softmax = args['mem_score_softmax']

        # dropout
        self.input_dropout_prob = args['mem_input_dropout']
        self.query_dropout_prob = args['mem_query_dropout']
        self.query_batchnorm = args['mem_query_batchnorm']

        # compute number of keys needed per level
        # TODO: Test unlimited index level  
        # self.n_key = np.power(self.num_memory, 1 / self.depth)
        if self.depth == 2:
            self.n_keys = np.sqrt(self.num_memory)
        elif self.depth == 3:
            self.n_keys = np.cbrt(self.num_memory)
        else : # self.depth == 4:
            self.n_keys = np.sqrt(np.sqrt(self.num_memory))

        if self.n_keys % 1 > 0:
            self.n_keys = np.floor(self.n_keys) + 1
        self.n_keys = int(self.n_keys)
        
        # initialize keys
        self.keys = nn.Parameter(create_keys(self.n_keys, self.depth, self.k_dim, self.keys_init, self.keys_normalized_init, self.random_seed))
        
        # Initialize dropout
        self.input_dropout = nn.Dropout(p=self.input_dropout_prob)
        self.query_dropout = nn.Dropout(p=self.query_dropout_prob)

        # query network
        self.query_proj = mlp(self.input_dim, self.depth * self.k_dim, bias=True, batchnorm=self.query_batchnorm)
    
    def get_key_logits(self, input):
        """
        Retrieve logits of all keys for accesing the memory
        """        
        # input dimensions
        assert input.shape[-1] == self.input_dim # (bs, k_dim)
        bs = input.shape[0]

        # compute query / store it
        input = self.input_dropout(input)                                         # input shape
        query = self.query_proj(input).view(bs, self.depth, self.k_dim)           # (bs, depth, k_dim)
        query = self.query_dropout(query)                                         # (bs, depth, k_dim)
        assert query.shape == (bs, self.depth, self.k_dim)

        assert query.shape[-1] == self.k_dim
        bs = query.size(0)

        # normalize queries
        query = query / (query.norm(2, dim=2, keepdim=True).expand_as(query) + EPSILON) # Avoid 0/0 by adding epsilon
        
        # nomrlaize key logits
        norm_keys = self.keys.norm(2, dim=2) + EPSILON

        # perform simple linear operation for query and keys
        logits = [F.linear(query[:,i,:], self.keys[:,i,:], bias=None) / norm_keys[:,i] for i in range(self.depth)] 
        logits = torch.stack(logits, dim=1)  # (bs, depth, n_keys)
        
        return logits
    
    def get_indices(self, logits):
        """
        Generate scores and indices given keys and unnormalized queries.
        """
        bs = logits.shape[0]                                                          # (bs, depth, n_keys)
        scores, indices = logits.topk(self.topk, dim=2, largest=True, sorted=True)    # (bs, depth, topk)
        
        # cartesian product on best candidate keys
        # TODO : Test unlimited index level
#         expand_size = (bs,) + tuple([self.topk for d in depth])
#         num_keys = torch.FloatTensor([self.n_keys]).to(logits.device)
#         all_scores = 0
#         all_indices = 0
#         for d in range(self.depth):            
#             view_size = (bs,) + tuple([self.topk if dim == d else 1 for dim in depth])
#             key_offset = torch.pow(num_keys, self.depth - 1 - d)
#             all_scores += scores[:,d,:].view(view_size).expand(expand_size)
#             all_indices += indices[:,d,:].view(view_size).expand(expand_size) * key_offset
                                                                   
        if self.depth == 2:
            all_scores = (
                scores[:,0,:].view(bs, self.topk, 1).expand(bs, self.topk, self.topk) +
                scores[:,1,:].view(bs, 1, self.topk).expand(bs, self.topk, self.topk)
            ).view(bs, -1)                                                                 # (bs, topk ** 2)
            all_indices = (
                indices[:,0,:].view(bs, self.topk, 1).expand(bs, self.topk, self.topk) * self.n_keys +
                indices[:,1,:].view(bs, 1, self.topk).expand(bs, self.topk, self.topk)
            ).view(bs, -1)                                                                  # (bs, topk ** 2)
        elif self.depth == 3:
            all_scores = (
                scores[:,0,:].view(bs, self.topk, 1, 1).expand(bs, self.topk, self.topk, self.topk) +
                scores[:,1,:].view(bs, 1, self.topk, 1).expand(bs, self.topk, self.topk, self.topk) +
                scores[:,2,:].view(bs, 1, 1, self.topk).expand(bs, self.topk, self.topk, self.topk)
            ).view(bs, -1)                                                                 # (bs, topk ** 3)
            all_indices = (
                indices[:,0,:].view(bs, self.topk, 1, 1).expand(bs, self.topk, self.topk, self.topk) * self.n_keys * self.n_keys +
                indices[:,1,:].view(bs, 1, self.topk, 1).expand(bs, self.topk, self.topk, self.topk) * self.n_keys +
                indices[:,2,:].view(bs, 1, 1, self.topk).expand(bs, self.topk, self.topk, self.topk)
            ).view(bs, -1)                                                                  # (bs, topk ** 3)                                                              # (bs, topk ** 2)
        else: # elif self.depth == 4:
            all_scores = (
                scores[:,0,:].view(bs, self.topk, 1, 1, 1).expand(bs, self.topk, self.topk, self.topk, self.topk) +
                scores[:,1,:].view(bs, 1, self.topk, 1, 1).expand(bs, self.topk, self.topk, self.topk, self.topk) +
                scores[:,2,:].view(bs, 1, 1, self.topk, 1).expand(bs, self.topk, self.topk, self.topk, self.topk) +
                scores[:,3,:].view(bs, 1, 1, 1, self.topk).expand(bs, self.topk, self.topk, self.topk, self.topk)
            ).view(bs, -1)                                                                 # (bs, topk ** 4)
            all_indices = (
                indices[:,0,:].view(bs, self.topk, 1, 1, 1).expand(bs, self.topk, self.topk, self.topk, self.topk) * self.n_keys * self.n_keys * self.n_keys +
                indices[:,1,:].view(bs, 1, self.topk, 1, 1).expand(bs, self.topk, self.topk, self.topk, self.topk) * self.n_keys * self.n_keys +
                indices[:,2,:].view(bs, 1, 1, self.topk, 1).expand(bs, self.topk, self.topk, self.topk, self.topk) * self.n_keys +
                indices[:,3,:].view(bs, 1, 1, 1, self.topk).expand(bs, self.topk, self.topk, self.topk, self.topk)
            ).view(bs, -1)                                                                  # (bs, topk ** 4)

        # select overall best scores and indices
        scores, best_indices = torch.topk(all_scores, k=self.topk, dim=1, largest=True, sorted=True)     # (bs, topk)
        indices = all_indices.gather(1, best_indices)                                                    # (bs, topk)

        # return scores with indices
        assert scores.shape == indices.shape == (bs, self.topk)
        return scores, indices
        
    def forward(self, input):
        """
        Retrieve indices for accesing the memory
        """        
        # retrieve key logits
        logits = self.get_key_logits(input)
        
        # get indices from key logits
        scores, indices = self.get_indices(logits)                       # (bs, depth, topk) ** 2
        
        # re-scoring
        if self.score_softmax:
            if self.topk == 1:
                scores = F.sigmoid(scores.float()).type_as(scores)              # (bs, topk)
            else:
                if self.temperature != 1:
                    scores = scores / self.temperature                          # (bs, topk)
                scores = F.softmax(scores.float(), dim=-1).type_as(scores)      # (bs, topk)

        return scores, indices

if __name__ == "__main__":
    print('=== TEST get_gaussian_keys FUNCTION ===')
    print('== gauss seed 0 - reproduce ==')
    gauss_1 = get_gaussian_keys(3, 2, 3, normalized=True, seed=0)
    gauss_2 = get_gaussian_keys(3, 2, 3, normalized=True, seed=0)
    print('sum', gauss_1.sum())
    print('is gauss same?', (gauss_1 == gauss_2).sum(), 'of', gauss_1.size)
    assert (gauss_1 == gauss_2).sum() == gauss_1.size

    print('== gauss seed 1 - reproduce ==')
    gauss_1 = get_gaussian_keys(3, 2, 3, normalized=True, seed=1)
    gauss_2 = get_gaussian_keys(3, 2, 3, normalized=True, seed=1)
    print('is gauss same?', (gauss_1 == gauss_2).sum(), 'of', gauss_1.size)
    assert (gauss_1 == gauss_2).sum() == gauss_1.size

    print('= normalized / unnormalized =')
    gauss_1 = get_gaussian_keys(3, 2, 3, normalized=True, seed=1)
    gauss_2 = get_gaussian_keys(3, 2, 3, normalized=False, seed=1)
    print('sum [gauss_1.abs() > 1] : ', (np.absolute(gauss_1) > 1).sum())
    print('sum [gauss_2.abs() > 1] : ', (np.absolute(gauss_2) > 1).sum())
    print('is gauss same?', (gauss_1 == gauss_2).sum(), 'of', gauss_1.size)
    assert ((gauss_1 == gauss_2).sum() > 0) != gauss_1.size

    print('PASSED get_gaussian_keys TEST')

    print('=== TEST get_uniform_keys FUNCTION ===')
    print('== uniform seed 0 - reproduce ==')
    uniform_1 = get_uniform_keys(3, 2, 3, normalized=True, seed=0)
    uniform_2 = get_uniform_keys(3, 2, 3, normalized=True, seed=0)
    print('sum', uniform_1.sum())
    print('is uniform same?', (uniform_1 == uniform_2).sum(), 'of', uniform_1.size)
    assert (uniform_1 == uniform_2).sum() == uniform_1.size

    print('== uniform seed 1 - reproduce ==')
    uniform_1 = get_uniform_keys(3, 2, 3, normalized=True, seed=1)
    uniform_2 = get_uniform_keys(3, 2, 3, normalized=True, seed=1)
    print('is uniform same?', (uniform_1 == uniform_2).sum(), 'of', uniform_1.size)
    assert (uniform_1 == uniform_2).sum() == uniform_1.size

    print('= normalized / unnormalized =')
    uniform_1 = get_uniform_keys(3, 2, 3, normalized=True, seed=1)
    uniform_2 = get_uniform_keys(3, 2, 3, normalized=False, seed=1)
    print('sum [uniform_1.abs() > 1] : ', (np.absolute(uniform_1) > 1).sum())
    print('sum [uniform_2.abs() > 1] : ', (np.absolute(uniform_2) > 1).sum())
    print('is uniform same?', (uniform_1 == uniform_2).sum(), 'of', uniform_1.size)
    assert ((uniform_1 == uniform_2).sum() > 0) != uniform_1.size

    print('PASSED get_uniform_keys TEST')
    
    print('=== TEST create_keys FUNCTION ===')
    n_keys = 3
    depth = 2
    k_dim = 3
    gauss_key = create_keys(n_keys, depth, k_dim, keys_init='gaussian', keys_normalized_init=True, random_seed=1)
    uniform_key = create_keys(n_keys, depth, k_dim, keys_init='uniform', keys_normalized_init=True, random_seed=0)
    print('is_same normalized gaussian key?', (gauss_key.numpy() == get_gaussian_keys(n_keys,depth,k_dim, True, 1)).sum(), 'of', gauss_key.numel())
    print('is_same normalized uniform key?', (uniform_key.numpy() == get_uniform_keys(n_keys,depth,k_dim, True, 0)).sum(), 'of', uniform_key.numel())
    assert (gauss_key.numpy() == get_gaussian_keys(n_keys,depth,k_dim, True, 1)).sum() == gauss_key.numel()
    assert (uniform_key.numpy() == get_uniform_keys(n_keys,depth,k_dim, True, 0)).sum() == uniform_key.numel()

    gauss_key = create_keys(n_keys, depth, k_dim, keys_init='gaussian', keys_normalized_init=False, random_seed=1)
    uniform_key = create_keys(n_keys, depth, k_dim, keys_init='uniform', keys_normalized_init=False, random_seed=0)
    print('is_same unormalized gaussian key?', (gauss_key.numpy() == get_gaussian_keys(n_keys,depth,k_dim, False, 1)).sum(), 'of', gauss_key.numel())
    print('is_same unormalized uniform key?', (uniform_key.numpy() == get_uniform_keys(n_keys,depth,k_dim, False, 0)).sum(), 'of', uniform_key.numel())
    assert (gauss_key.numpy() == get_gaussian_keys(n_keys,depth,k_dim, False, 1)).sum() == gauss_key.numel()
    assert (uniform_key.numpy() == get_uniform_keys(n_keys,depth,k_dim, False, 0)).sum() == uniform_key.numel()

    print('PASSED create_keys TEST')
    
    print('=== TEST mlp FUNCTION ===')
    input_size = 2 # input dim
    depth = 3 # multi index depth
    output_size = 4 # output dim
    print('bias 1 batchnorm 1')
    lin_mod = mlp(input_size, depth * output_size, bias=True, batchnorm=True)
    print(lin_mod)
    assert len(lin_mod) == 3
    print('bias 0 batchnorm 1')
    lin_mod = mlp(input_size, depth * output_size, bias=False, batchnorm=True)
    print(lin_mod)
    assert lin_mod[0].bias is None
    print('bias 0 batchnorm 0')
    lin_mod = mlp(input_size, depth * output_size, bias=False, batchnorm=False)
    print(lin_mod)
    assert len(lin_mod) == 2 and lin_mod[0].bias is None

    print('PASSED mlp TEST')
    
    print('=== TEST HashingMemory.__init__ FUNCTION ===')
    args = {
        'random_seed': 0,
        'mem_depth': 2,
        'mem_k_dim': 9,
        'mem_topk': 3,
        'mem_keys_init': 'uniform',
        'mem_normalize_query': True,
        'mem_normalize_key': True,
        'mem_temperature': 1.0,
        'mem_score_softmax': True,
        'mem_input_dropout': 0,
        'mem_query_dropout': 0,
        'mem_query_batchnorm': True
    }
    num_memory = 3000000
    input_dim = 10

    memory = HashingMemory(input_dim, num_memory, args)
    print(memory)

    assert memory.input_dropout.p == args['mem_input_dropout']
    assert memory.query_dropout.p == args['mem_query_dropout']
    assert memory.query_proj[0].in_features == input_dim

    print('PASSED HashingMemory.__init__ TEST')
    
    print('=== TEST HashingMemory.get_key_logits FUNCTION ===')
    bs = 8
    logits = memory.get_key_logits(torch.rand(bs, input_dim))
    print('logits.shape',logits.shape)
    assert logits.shape[0] == bs and logits.shape[1] == memory.depth and logits.shape[2] == memory.n_keys

    print('PASSED HashingMemory.get_key_logits TEST')

    print('=== TEST HashingMemory.get_indices FUNCTION ===')
    print('== depth 2 ==')
    args['mem_depth'] = 2
    memory = HashingMemory(input_dim, num_memory, args)
    logits = memory.get_key_logits(torch.rand(bs, input_dim))
    scores, indices = memory.get_indices(logits)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)

    print('== depth 3 ==')
    args['mem_depth'] = 3
    memory = HashingMemory(input_dim, num_memory, args)
    logits = memory.get_key_logits(torch.rand(bs, input_dim))
    scores, indices = memory.get_indices(logits)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)

    print('== depth 4 ==')
    args['mem_depth'] = 4
    memory = HashingMemory(input_dim, num_memory, args)
    logits = memory.get_key_logits(torch.rand(bs, input_dim))
    scores, indices = memory.get_indices(logits)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)

    print('PASSED HashingMemory.get_indices TEST')
    
    print('=== TEST HashingMemory.forward FUNCTION ===')
    args['mem_depth'] = 4

    print('== temperature 1 softmax False ==')
    args['mem_temperature'] = 1
    args['mem_score_softmax'] = False
    memory = HashingMemory(input_dim, num_memory, args)
    scores, indices = memory(torch.rand(bs, input_dim))
    print('scores', scores)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)

    print('== temperature 1 softmax True ==')
    args['mem_temperature'] = 1
    args['mem_score_softmax'] = True
    memory = HashingMemory(input_dim, num_memory, args)
    scores, indices = memory(torch.rand(bs, input_dim))
    print('scores', scores)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)

    print('== temperature 0.5 softmax False ==')
    args['mem_temperature'] = 0.5
    args['mem_score_softmax'] = False
    memory = HashingMemory(input_dim, num_memory, args)
    scores, indices = memory(torch.rand(bs, input_dim))
    print('scores', scores)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)

    print('== temperature 0.5 softmax True ==')
    args['mem_temperature'] = 0.5
    args['mem_score_softmax'] = True
    memory = HashingMemory(input_dim, num_memory, args)
    scores, indices = memory(torch.rand(bs, input_dim))
    print('scores', scores)
    print('scores.shape', scores.shape)
    print('indices.shape', indices.shape)
    assert scores.shape == indices.shape == (bs, memory.topk)
    
    print('PASSED HashingMemory.forward TEST')