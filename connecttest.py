from vantage6.node.server_io import NodeClient
from vantage6.client import Client 

client = NodeClient("http://0.0.0.0", 5000, "/api")
key = "2824ca26-c1e6-11eb-a20e-0242ac110003"
client.authenticate(key)


#researchClient = Client("http://localhost", 5000, "/api")
#researchClient = Client("http://0.0.0.0", 5000, "/api")
#researchClient.authenticate("researcher", "1234")