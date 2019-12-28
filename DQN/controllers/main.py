import timeit
from .agent2 import DQNAgent
import json
from pprint import pprint
from django.http import HttpResponse




#result=agent.generate_greedy_query_conditions_order()
#print(result)

#f = open("model_vs_optimizer", "a")
#f.write(str(result))



def train(request):
    agent = DQNAgent()
    for i in range(100000):
        start = timeit.default_timer()
        pprint("learning" + str(i))
        agent.start_learning()
        agent.get_q_table()
        agent.saveModel()
        stop = timeit.default_timer()
        pprint('Time: ', stop - start)
    response = {}
    response["key"] = "done"
    return HttpResponse(json.dumps(response), content_type="application/json")