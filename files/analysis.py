import pandas as pd
import matplotlib.pyplot as plt
import mesa.model

from model import ColabNetwork, State,Popular
from mesa.batchrunner import batch_run

model=ColabNetwork(
            # num_nodes=500,
            # avg_node_degree=3,
            probability_repeated=0.4,
            probability_popular=0.00046,
            probability_embedded=0.0007,
            probability_gatekeeper=0.00064,
            active_ratio=0.3,
            max_colab=10,
            top_percent=.01,
            read_data=1)

while model.running and model.schedule.steps < 10:
    model.step()
print(model.schedule.steps)

model_out = model.datacollector.get_model_vars_dataframe()


for i in range(len(model_out)):
    model_out['productivity_rate']=1/(model_out['network_productivity'].iloc[i]/model_out['network_productivity'].iloc[:-1])
print(model_out)
# #

# model_out.to_csv('/Users/mahsa/Desktop/collaboration_%1_repeat_top.csv', index =True)
# model_out.Activated.plot()
# model_out.Idol.plot()
# model_out.Popular.plot()
model_out.network_productivity.plot()

plt.show()
#
# def pop_count(model):
#     pops_num=0
#     for agent in model.schedule.agents:
#         if Popular.POPULAR:
#             pops_num+=1
#         return pops_num
#
#
#
#
# fixed_params= {'num_nodes':20,
#             'avg_node_degree':3,
#             'probability_repeated':0.1,
#             'probability_embedded':0.1,
#             'probability_gatekeeper':0.1,
#             'active_ratio':0.3,
#             'max_colab':10,
#             'top_percent':0.3}
#
# variable_parms={'probability_popular':[0.1,0.2,0.3,0.4]}
#
# model_rep= {"popularcount": pop_count}
#
# param_sweep = batch_run(
# model_cls: ColabNetwork,
# parameters: fixed_params,
# number_processes: Optional[int] = None,
# iterations: int = 1,
# data_collection_period: int = - 1,
# max_steps: int = 10,
# display_progress: bool = True)
#
#
#
# param_sweep.run_all()
# df = param_sweep.get_model_vars_dataframe()
#
#
#
#
# plt.scatter(df.probability_gatekeeper, df.popularcount)
# plt.grid(True)
