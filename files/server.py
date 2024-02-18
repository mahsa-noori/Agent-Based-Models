import math

from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import UserSettableParameter
from mesa.visualization.modules import ChartModule
from mesa.visualization.modules import NetworkModule
from mesa.visualization.modules import TextElement
from .model import ColabNetwork, State, number_activated


def network_portrayal(G):
    print(G.nodes())
    # print(G.edges())
    # print(sorted(G))
    # The model ensures there is always 1 agent per node

    print(G.edges(data=True))
    for node in G.nodes():
        agent = G.nodes[node]["agent"][0]
        if agent.state == State.ACTIVATED:
            print(f'Node number {node} is {agent.state}')

    def node_color(agent):
        # print(agent.state)
        return {State.ACTIVATED: "#FF0000", State.IDOL: "#008000"}.get(
            agent.state, "#808080"
        )

    def edge_color(agent1, agent2):
        # a1 = agent1.unique_id
        # a2 = agent2.unique_id
        # print(agent1.unique_id)

        # if (a1,a2) in G.edges() or (a2,a1) in G.edges():
        #     print((a1,a2))
        #
        #     color = "#000000" ## hex black
        return "#000000"




        # if State.IDOL in (agent1.state, agent2.state):
        #     return "#000000"
        # return "#e8e8e8"

    def edge_width(agent1, agent2):


        # if State.IDOL in (agent1.state, agent2.state):
        #     # print('hi')
        #     return 3
        return

    def get_agents(source, target):
        # print(G.nodes[source]["agent"][0])
        # print(source)
        # for (_, agents) in G.nodes.data("agent"):
        #     print(f'unique_id={agents[0].unique_id}')
        # print(source)
        # print(G.nodes[source]["agent"][0])
        return G.nodes[source]["agent"][0], G.nodes[target]["agent"][0]

    portrayal = dict()
    portrayal["nodes"] = [
        {
            "size": 6,
            "color": node_color(agents[0]),
            # "tooltip": f"id: {agents[0].unique_id}<br>state: {agents[0].state.name}",
        }
        for (_, agents) in G.nodes.data("agent")
    ]

    portrayal["edges"] = [
        {
            "source": source,
            "target": target,
            "color": edge_color(*get_agents(source, target)),
            "width": edge_width(*get_agents(source, target)),
        }
        for (source, target) in G.edges
    ]

    return portrayal


# network = NetworkModule(network_portrayal, 500, 500, library="d3")
network = NetworkModule(network_portrayal, 500, 500)

chart = ChartModule(
    [
        {"Label": "ACTIVATED", "Color": "#FF0000"},
        {"Label": "IDOL", "Color": "#008000"},
    ]
)

model_params = {
    "num_nodes": UserSettableParameter(
        "slider",
        "Number of agents",
        100,## default value
        4,## start
        100,## end
        1,### step
        description="Choose how many agents to include in the model",
    ),
    # "avg_node_degree": UserSettableParameter(
    #     "slider", "Avg Node Degree", 1, 1, 30, 1, description="Avg Node Degree"
    # ),
    "active_ratio": UserSettableParameter(
        "slider", "Active Ratio", .3, .1, 1, .1, description="Avg Node Degree"
    ),
    "probability_repeated": UserSettableParameter(
        "slider",
        "Probability repeated",
        0.01, ## default value
        0.01, ## start
        0.9, ## end
        0.01, ### step
        description="Probability that a recovered agent will become "
        "resistant to this virus in the future",
    ),
    "probability_popular": UserSettableParameter(
        "slider",
        "Probability_popular",
        0.01,  ## default value
        0.01,  ## start
        0.9,  ## end
        0.01,  ### step
        description="Probability that a recovered agent will become "
                    "resistant to this virus in the future",
    ),
    "probability_embedded": UserSettableParameter(
        "slider",
        "Probability embedded",
        0.01,  ## default value
        0.01,  ## start
        0.9,  ## end
        0.01,  ### step
        description="Probability that a recovered agent will become "
                    "resistant to this virus in the future",
    ),
    "probability_gatekeeper": UserSettableParameter(
        "slider",
        "Probability gatekeeper",
        0.01,  ## default value
        0.01,  ## start
        0.9,  ## end
        0.01,  ### step
        description="Probability that a recovered agent will become "
                    "resistant to this virus in the future",
    ),
    "read_data": UserSettableParameter(
        "slider",
        "Random or not",
        1,  ## default value
        0,  ## start
        1,  ## end
        1,  ### step
        description="Choose between random and non-random graph!",
    ),
# "read_populars": UserSettableParameter(
#         "slider",
#         "read pops or not",
#         0,  ## default value
#         0,  ## start
#         1,  ## end
#         1,  ### step
#         description="Choose between random and non-random graph!",
#     ),

}

server = ModularServer(
   ColabNetwork, [network, chart], "Collaboration Network", model_params
)
server.port = 4444