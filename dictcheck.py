import sys
import torch
import json


from vantage6.common.globals import STRING_ENCODING

def _to_json_dict_with_strings(dictionary):
    """
    Convert dict to dict with leafs only being strings. So it recursively makes keys to strings
    if they are not dictionaries.

    Use case:
        - saving dictionary of tensors (convert the tensors to strins!)
        - saving arguments from script (e.g. argparse) for it to be pretty

    e.g.

    """
    if type(dictionary) != dict:
        return str(dictionary)
    d = {k: _to_json_dict_with_strings(v) for k, v in dictionary.items()}
    return d


def to_json(dic):
    import types
    import argparse

    if type(dic) is dict:
        dic = dict(dic)
    else:
        dic = dic.__dict__
    return _to_json_dict_with_strings(dic)

dict1 = {
        'lin_layers.0.weight' : torch.randn((100,28*28), dtype=torch.double),
        'lin_layers.0.bias' : torch.randn((100), dtype=torch.double),
        'lin_layers.2.weight' : torch.randn((2,100), dtype=torch.double),
        'lin_layers.2.bias' : torch.randn((2), dtype=torch.double)
        }  

dict2 = {                    
        'conv_layers.0.weight': torch.randn((1,1,3,3)),
        'conv_layers.0.bias' : torch.randn(1),
        'lin_layers.0.weight' : torch.randn((2, 196)),
        'lin_layers.0.bias' : torch.randn(2)
        }

size1 = sys.getsizeof(dict1)
size1_json = sys.getsizeof(json.dumps(to_json(dict1)).encode(STRING_ENCODING))
size2 = sys.getsizeof(dict2)
size2_json = sys.getsizeof(json.dumps(to_json(dict2)).encode(STRING_ENCODING))

print(size1, size1_json)

print(size2, size2_json)