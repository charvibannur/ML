# Recommendation engine with a graph
video: https://drive.google.com/file/d/1jhSWPg59MJTbnyivRCoCdrJs8FPg7RWH/view?usp=sharing

The purpose is to build a recommendation engine based on graph by using the Adamic Adar measure.
The more the measure is high, the closest are the two nodes.
The measures between all movies are NOT pre-calculated, in order to determine the list of recommendation films, we are going to explore the neighborhood of the target film
# How to take in account of the description?
First idea ...
In order to take in account the description, the movie are clustered by applying a KMeans clustering with TF-IDF weights
So two movies that belong in a group of description will share a node.
The fewer the number of films in the group, the more this link will be taken into account

*but it doesn't work because clusters are too unbalanced

Second idea ...
In order to take in account the description, calcul the TF-IDF matrix
and for each film, take the top 5 of similar descriptions and create a node Similar_to_this. This node will be taken in account in the Adamic Adar measure.
# Adamic Adar measure
It is a measure used to compute the closeness of nodes based on their shared neighbors.

x and y are 2 nodes (2 Movies)
N(one_node) is a function that return the set of adjacent nodes to one_node
adamicAdar(x,y)=∑u∈N(x)∩N(y)1log(N(u)) 

«say otherwise, for each node u in common to x and y, add to the measure 1/log(N(u))»

The quantity  1log(N(u))  determine the importance of u in the measure.

if x and y share a node u that has a lot of adjacent nodes, this node is not really relevant. → N(u) is high → 1/log(N(u)) is not high
if x and y share a node u that does not have a lot of adjacent nodes, this node is really relevant.
→ N(u) is not high → 1/log(N(u)) is higher

# Cosine similarity
is a metric, helpful in determining, how similar the data objects are irrespective of their size. We can measure the similarity between two sentences in Python using Cosine Similarity. In cosine similarity, data objects in a dataset are treated as a vector.
Formula: Cos(x, y) = x . y / ||x|| * ||y||

# Load the graph (undirected graph)
Nodes are :

Movies
Person ( actor or director)
Category
Country
Cluster (description)
Sim(title) top 5 similar movies in the sense of the description

Edges are :
ACTED_IN : relation between an actor and a movie
CAT_IN : relation between a category and a movie
DIRECTED : relation between a director and a movie
COU_IN : relation between a country and a movie
DESCRIPTION : relation between a cluster and a movie
SIMILARITY in the sense of the description
«so, two movies are not directly connected, but they share persons, categories,clusters and countries»
