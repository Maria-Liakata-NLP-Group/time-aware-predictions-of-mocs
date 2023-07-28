"""
Will use the difference between the current post and previous, and future posts
as a feature to identify whether the current post is a change or not.
"""
import torch


def cosine_similarity(v1, v2, dim=1):
    cos = torch.nn.CosineSimilarity(dim=dim, eps=1e-6)
    
    d = cos(v1,v2)
    
    return d

def dissimilarity(v1,v2,dim=1, type='cosine'):
    if type == 'cosine':
        d  = 1-cosine_similarity(v1,v2,dim=dim)
    if type == 'absolute_difference':
        d = torch.abs(v1-v2)
    if type == 'difference':
        d = v1-v2
    
    return d

