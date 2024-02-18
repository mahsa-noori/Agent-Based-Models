### ColabNetwork Documentation

### Introduction

The ColabNetwork module (model.py) is designed to simulate a collaborative network among a set of agents (authors) and observe the dynamics of their collaborations over time. It is implemented using the Mesa library, which provides tools for building agent-based models.

### Dependencies

The following Python libraries are required to use this module:

math

random

enum

networkx

pandas

matplotlib.pyplot

mesa

Make sure these libraries are installed in your Python environment before using the ColabNetwork module.

### Usage

To use the ColabNetwork module, follow these steps:

Import the Module: Import the ColabNetwork module in your Python script or interactive session.

from ColabNetwork import ColabNetwork

Initialize the Model: Create an instance of the ColabNetwork class by specifying the parameters for the simulation.


model = ColabNetwork(
    num_nodes=100,  # Number of nodes (authors) in the network
    avg_node_degree=3,  # Average node degree for random graph generation
    probability_repeated=0.4,  # Probability of repeated collaboration
    probability_popular=0.015,  # Probability of collaborating with popular authors
    probability_embedded=0.01,  # Probability of collaborating with embedded authors
    probability_gatekeeper=0.001,  # Probability of collaborating with gatekeeper authors
    active_ratio=0.3,  # Ratio of active authors at each step
    max_colab=10,  # Maximum number of collaborations each author can make at each step
    top_percent=0.3,  # Percentage of critical authors in each category
    read_data=0  # Flag indicating whether to read data from a file (0 for random graph generation)
)
Run the Model: Execute the simulation by calling the run_model method and specifying the number of steps.

model.run_model(n=100)  # Run the model for 100 steps

Data Collection: Access the collected data using the datacollector attribute of the model.

data = model.datacollector.get_model_vars_dataframe()

### Explanation of Parameters

num_nodes: Number of nodes (authors) in the network.

avg_node_degree: Average node degree for random graph generation.

probability_repeated: Probability of repeated collaboration.

probability_popular: Probability of collaborating with popular authors.

probability_embedded: Probability of collaborating with embedded authors.

probability_gatekeeper: Probability of collaborating with gatekeeper authors.

active_ratio: Ratio of active authors at each step.

max_colab: Maximum number of collaborations each author can make at each step.

top_percent: Percentage of critical authors in each category.

read_data: Flag indicating whether to read data from a file (0 for random graph generation).

### Model Dynamics

The model simulates the collaboration network dynamics over multiple steps. At each step, a subset of authors is activated based on the active_ratio parameter.
Authors collaborate with each other based on predefined probabilities and critical author categories (popular, gatekeeper, embedded).
Collaboration probabilities and critical author categories can be adjusted through model parameters.
Data regarding the state of the network (e.g., number of activated authors, collaborations) are collected at each step.

### Customization

Users can customize the model by adjusting the parameters according to their specific research questions or hypotheses.

Additional functionality can be implemented by modifying the ColabNetwork class or extending its functionality.

### Example

# Initialize the model
model = ColabNetwork(
    num_nodes=100,
    avg_node_degree=3,
    probability_repeated=0.4,
    probability_popular=0.015,
    probability_embedded=0.01,
    probability_gatekeeper=0.001,
    active_ratio=0.3,
    max_colab=10,
    top_percent=0.3,
    read_data=0
)

### Run the model

model.run_model(n=100)

### Access collected data

data = model.datacollector.get_model_vars_dataframe()

### Conclusion

The ColabNetwork module provides a flexible framework for simulating and analyzing collaboration networks among authors. By adjusting various parameters, researchers can explore different scenarios and gain insights into the dynamics of collaborative research networks.





Is this conversation helpful so far?
