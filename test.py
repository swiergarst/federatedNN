from typing import Sized
import numpy as np
from helper_functions import init_params, heatmap
import torch
import docker


#client = docker.from_env()
client = docker.APIClient(base_url='unix://var/run/docker.sock')
try:
    print(client.pull("sgarst/federated-learning:2ClassNN2"))
except docker.errors.APIError as e:
    print(e)