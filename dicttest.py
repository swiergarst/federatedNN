import numpy as np
import time

from vantage6.client import Client

client = Client("http://0.0.0.0", 5000, "/api")
client.authenticate("researcher", "1234")
privkey = "/home/swier/.local/share/vantage6/node/privkey_testOrg0.pem"
client.setup_encryption(privkey)

ids = [org['id'] for org in client.collaboration.get(1)['organizations']]



## setup contains 10 nodes
setting = 1 # 1 for the 'bad' dict, 2 for the 'good' one
rounds = 100

for round in range(rounds):
    task_list = np.empty(10, dtype=object)
    print("starting round", round)
    task = client.post_task(
            input_ = {
            'method': "dicttest",
            'kwargs' : {
                "setting" : setting
            }
        },
        name = "dicttest",
        image = "sgarst/federated-learning:dicttest3",
        organization_ids= ids,
        collaboration_id=1
    )
   # task_list[i] =  task

    finished = False
    while (finished == False):
        result = client.get_results(task_id=task.get("id"))
        if not (None in [result[0]["result"]]):
            finished = True
        print("waiting")
        time.sleep(1)
