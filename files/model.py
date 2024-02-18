import math
import random
from enum import Enum
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
from mesa import Agent, Model, agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from mesa.space import NetworkGrid


PATH = "C:/Users/siina/Documents/Python Scripts/Mahsa Folder/OriginalData.csv"
# data = pd.read_csv('/Users/mahsa/Downloads/Data/OriginalData.csv', usecols=['Author_s_ID'], squeeze = True)
data = pd.read_csv(PATH, usecols=['Author_s_ID'], squeeze = True)

# index=data.index
# print(index)
sampledata=data.sample(500,axis=0,random_state=42)




def construct_G():

    # all_papers = data.sample(num_nodes, axis=0)
    all_papers=sampledata

    G = nx.Graph()
    edges = []

    all_authors = {}  ## (a1,a2): 10

    for paper in all_papers:
        # print(f'new paper ={paper}\n')
        authors_str = paper.split(';')[:-1]
        authors = []
        for author_str in authors_str:
            authors.append(int(author_str))

        authors.sort()
        # map_id = []
        # k=1
        # for id in authors:
        #     map_id.append((id,k))
        #     k+=1
        # print(f'map_id{map_id}')


        for author_1 in authors:
            for author_2 in authors:
                if author_1 < author_2:
                    if (author_1, author_2) in all_authors:
                        all_authors[(author_1, author_2)] += 1
                    else:
                        all_authors[(author_1, author_2)] = 1



    for (author_1, author_2) in all_authors:
        new_edge = (author_1, author_2)
        G.add_edge(author_1, author_2, weight=all_authors[(author_1, author_2)])
        edges.append(new_edge)

    return G





class State(Enum):
    ACTIVATED = 1
    IDOL = 0



class Popular(Enum):
    POPULAR = 1
    NON_POPULAR = 0


class GateKeeper(Enum):
    GATEKEEPER = 1
    NON_GATEKEEPER = 0


class Embedded(Enum):
    EMBEDDED = 1
    NON_EMBEDDED = 0


PROBABILITIES = {'REPEATED': 0.4, 'POPULAR': 0.015, 'GATEKEEPER': 0.001, 'EMBEDDED': 0.01}


def number_state(model, state):
    return sum(1 for a in model.grid.get_all_cell_contents() if a.state is state)

def nodes_name(model):
    node_list=[]
    for a in model.all_agents:
        node_list.append(a.id)
    return node_list
def activated_nodes_name(model):
    node_list = []
    for a in model.all_agents:
        if a.state is State.ACTIVATED:
            node_list.append(a.id)
    return node_list

def number_activated(model):
    return number_state(model, State.ACTIVATED)


def number_idol(model):
    return number_state(model, State.IDOL)


def number_popular(model):
    # for a in model.all_agents:
    #     print(type(a))
    return sum(1 for a in model.all_agents if a.Popular is Popular.POPULAR)


def number_gatekeeper(model):
    return sum(1 for a in model.all_agents if a.Gatekeeper is GateKeeper.GATEKEEPER)


def number_embedded(model):
    return sum(1 for a in model.all_agents if a.Embedded is Embedded.EMBEDDED)

def networkproduction(model):
    return model.num_edges




def populars(G, top_percent):
    num_nodes = G.number_of_nodes()
    num_top_nodes = round(num_nodes * top_percent)

    degrees = dict(G.degree(G.nodes()))
    authors_degrees = {k: v for k, v in sorted(degrees.items(), key=lambda item: item[1], reverse=True)}
    ordered_authors = list(authors_degrees.items())
    populars = []

    for i in range(num_top_nodes):
        new_author = ordered_authors[i][0]
        populars.append(new_author)
    return populars


def gatekeepers(G, top_percent):
    num_nodes = G.number_of_nodes()
    num_top_nodes = round(num_nodes * top_percent)

    betweennesses = nx.betweenness_centrality(G)
    authors_betweennesses = {k: v for k, v in sorted(betweennesses.items(), key=lambda item: item[1], reverse=True)}
    ordered_betweennesses = list(authors_betweennesses)
    gatekeepers = []

    for i in range(num_top_nodes):
        new_author = ordered_betweennesses[i]
        gatekeepers.append(new_author)

    return gatekeepers


def embeddeds(G, top_percent):
    num_nodes = G.number_of_nodes()
    num_top_nodes = round(num_nodes * top_percent)

    clustering_coeffs = nx.clustering(G)
    clustering_coeffs = {k: v for k, v in sorted(clustering_coeffs.items(), key=lambda item: item[1], reverse=True)}
    sorted_clustering_coeffs = list(clustering_coeffs.items())
    embedded_scis = []

    for i in range(num_top_nodes):
        new_author = sorted_clustering_coeffs[i][0]
        embedded_scis.append(new_author)

    return embedded_scis


class ColabNetwork(Model):
    """A collaboration model with some number of agents"""

    def __init__(
            self,
            num_nodes=4,
            avg_node_degree=3,
            probability_repeated=PROBABILITIES['REPEATED'],
            probability_popular=PROBABILITIES['POPULAR'],
            probability_embedded=PROBABILITIES['EMBEDDED'],
            probability_gatekeeper=PROBABILITIES['GATEKEEPER'],
            active_ratio=0.3,
            max_colab=10,
            top_percent=.3,
            read_data=0,
            # read_gatekeepers=1,
            # read_populars=1,
            # read_embeddeds=1,
            # read_repeateds=1
    ):
        # Set probability values for repeated, gatekeepers etc. Value p for repeated for instance means that at each
        # step each active agent collaborates with any of its coauthor with probability p. Similarly for the critical
        # actors
        #
        # self.read_repeateds=read_repeateds
        # self.read_embedded=read_embeddeds
        # self.read_populars=read_populars
        # self.read_gatekeepers=read_gatekeepers
        self.probability_repeated = probability_repeated  ### repeated authors collaborate with this probability
        self.probability_popular = probability_popular
        self.probability_embedded = probability_embedded
        self.probability_gatekeeper = probability_gatekeeper

        self.probabilities = {'REPEATED': probability_repeated, 'POPULAR': probability_popular,
                            'GATEKEEPER': probability_embedded, 'EMBEDDED': probability_gatekeeper}




        # Set number of nodes and generate the random graph we will work with throughout.  In case of working with real
        # data ignore this part.

        self.num_nodes = num_nodes
        self.read_data = read_data

        if self.read_data:

            self.G = construct_G()

        else:
            self.G = nx.erdos_renyi_graph(n=self.num_nodes, p=avg_node_degree / self.num_nodes)
            for (u, v) in self.G.edges():
                self.G[u][v]['weight'] = 1



        # mapping = {i: f'Author {i}' for i in range(num_nodes)}
        # self.G = nx.relabel_nodes(self.G, mapping)

        # Set three important parameters:
        # 1) active_ratio denotes the portion of agents activated at each step
        # 2) max_colab denotes the maximum number of collaboration each agent may make at each network's step
        # 3) top_percent denotes the portion that determines the critical actors in each sub-category e.g., populars

        self.active_ratio = active_ratio
        self.max_colab = max_colab
        self.top_percent = top_percent

        self.criticals = {}

        self.criticals['populars'] = populars(self.G, self.top_percent)
        # print(self.criticals['populars'])
        # if not self.read_populars:
        #     self.G.remove_nodes_from(self.criticals['populars'])
        #     # for i in self.criticals['populars']:
        #     #     self.G.remove_node(self.criticals['populars'])
        # print(self.G.nodes)
        self.criticals['gatekeepers'] = gatekeepers(self.G, self.top_percent)
        self.criticals['embeddeds'] = embeddeds(self.G, self.top_percent)


        self.neighbors = {}
        #
        for node in self.G.nodes():

            self.neighbors[node] = self.G.neighbors(node)

        # Define the set of all authors
        self.all_authors = self.G.nodes()




        # Using active_ratio gets the number of most (active) authors
        self.num_top_authors = range(round(self.num_nodes * self.top_percent))
        self.num_active_authors = math.floor(self.active_ratio * len(self.all_authors))
        self.num_edges=0





        self.grid = NetworkGrid(self.G)
        self.schedule = RandomActivation(self)
        self.datacollector = DataCollector(
            {
                "Node_ID": nodes_name,
                "Active_node_ID":activated_nodes_name,
                "Activated": number_activated,
                "Idol": number_idol,
                "Popular": number_popular,
                "Gatekeeper": number_gatekeeper,
                "Embedded": number_embedded,
                "network_productivity":networkproduction,
            }
        )

        ## Create agents using the grid.place_agent functionality of MESA. More importanly, we add each agent to the
        # schedule so that they get their turns when the network.schedule is running.

        for i, node in enumerate(self.all_authors):
            ## Consider a new set for each neighbors of each author and pass it to the author class.
            # colab_dict = {}
            # colab_dict['neighbors'] = self.G.neighbors(node)  ## type = dic_keyiterator

            a = Author(
                unique_id=node,
                model=self,
                neighbors=self.neighbors,
                criticals=self.criticals,
                G=self.G,
                initial_state=State.IDOL,
                probabilities=self.probabilities,
                max_colab=self.max_colab,
            )

            self.schedule.add(a)
            # Add the agent to the node
            # x = self.random.randrange(self.grid.width)
            # y = self.random.randrange(self.grid.height)
            self.grid.place_agent(a, node)

        self.all_agents = self.grid.get_cell_list_contents(self.all_authors)

        self.running = True
        self.datacollector.collect(self)

    def step(self): ## model step

        # print(self.criticals)

        ## At each step, we determin the critical actors. Recall that they are in three different categories:
        # 1) Populars, 2) Gatekeepers and 3) Embedded Scientists
        self.num_edges=self.G.number_of_edges()

        popular_authors = populars(self.G, self.top_percent)
        self.criticals['populars'] = popular_authors
        self.popular_agents = self.grid.get_cell_list_contents(popular_authors)
        for agent in self.popular_agents:
            agent.Popular = Popular.POPULAR

        gatekeepers_authors = gatekeepers(self.G, self.top_percent)
        self.criticals['gatekeepers'] = gatekeepers_authors
        self.gatekeepers_agents = self.grid.get_cell_list_contents(gatekeepers_authors)
        for agent in self.gatekeepers_agents:
            agent.Gatekeeper = GateKeeper.GATEKEEPER

        embeddeds_authors = embeddeds(self.G, self.top_percent)
        self.criticals['embeddeds'] = embeddeds_authors
        self.embeddeds_agents = self.grid.get_cell_list_contents(embeddeds_authors)
        for agent in self.embeddeds_agents:
            agent.Embedded = Embedded.EMBEDDED

        # Active some nodes
        self.active_agents = self.random.sample(self.all_agents, self.num_active_authors)

        for agent in self.all_agents:
            agent.state = State.IDOL

        for agent in self.active_agents:
            agent.state = State.ACTIVATED

        for node in self.G.nodes():
            self.neighbors[node] = self.G.neighbors(node)  ## {node_A: set_A, node_B:set_B, ... }



        self.schedule.step()
        self.datacollector.collect(self)

    def run_model(self, n):
        for i in range(n):
            self.step()




class Author(Agent):
    def __init__(
            self,
            unique_id,
            neighbors, ## dictionary containing information about all the other neighbors
            criticals,
            G,
            model,
            initial_state,
            probabilities,
            max_colab=10,
    ):
        super().__init__(unique_id, model)
        self.id = unique_id
        self.G = G
        self.neighborhood = list(neighbors[unique_id])  ## iterator
        self.populars = criticals['populars']
        if self.id in self.populars:
            self.Popular = Popular.POPULAR
        else:
            self.Popular = Popular.NON_POPULAR

        self.gatekeepers = criticals['gatekeepers']

        if self.id in self.gatekeepers:
            self.Gatekeeper= GateKeeper.GATEKEEPER
        else:
            self.Gatekeeper = GateKeeper.NON_GATEKEEPER

        self.embeddeds = criticals['embeddeds']

        if self.id in self.embeddeds:
            self.Embedded= Embedded.EMBEDDED
        else:
            self.Embedded = Embedded.NON_EMBEDDED

        self.criticals = criticals

        # for x in self.neighbors:
        #     print(f'{x} is a neighbor of {unique_id}')
        # self.num=len(neighbors)
        self.max_colab = max_colab
        self.state = initial_state ###### self.state is state

        self.rpt_prob = probabilities['REPEATED']
        self.gtek_prob = probabilities['GATEKEEPER']
        self.pop_prob = probabilities['POPULAR']
        self.emb_prob = probabilities['EMBEDDED']

        self.num = self.G.number_of_nodes()

    def step(self):

        # for i in self.neighborhood:
        #     print(f'{self.id} is adjacent to {i}')
        if self.state is State.ACTIVATED:
            for author in self.neighborhood:
                if random.random() < self.rpt_prob:
                    self.G[author][self.id]['weight'] += 1
                    # print(f'{author} and {self.id} work together')
        # Colloboration with populars

            for author in self.populars:

                if random.random() < self.pop_prob:
                    self.G.add_edge(author, self.id)
                    # # print(f'{author} and {self.id} work together! POPS')
                    # if (author, self.id) not in self.G.edges():
                    #     self.G.add_edge(author,self.id)
                    # else:
                    #     self.G[author][self.id]['weight'] += 1

            for author in self.gatekeepers:


                if random.random() < self.gtek_prob:
                    self.G.add_edge(author, self.id)
            #
            # for author in self.populars:
            #
            #     if random.random() < self.pop_prob:
            #         self.G.add_edge(author, self.id)

            for author in self.embeddeds:

                if random.random() < self.emb_prob:
                    self.G.add_edge(author, self.id)
