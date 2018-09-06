---
title: Infomap Notebooks - Hands-on Tutorials on Higher-Order Data Analytics
permalink: /
---

Network-based data mining techniques such as graph mining, (social) network analysis, link prediction and graph clustering form an important foundation for data science applications in computer science, computational social science, and the life sciences. They help to detect patterns in large data sets that capture dyadic relations between pairs of genes, species, humans, or documents and they have improved our understanding of complex networks.

While the potential of analysing graph or network representations of relational data is undisputed, we increasingly have access to data on networks that contain more than just dyadic relations. Consider, e.g., data on user click streams in the Web, time-stamped social networks, gene regulatory pathways, or time-stamped financial transactions. These are examples for time-resolved or sequential data that not only tell us who is related to whom but also when and in which order relations occur. Recent works have exposed that the timing and ordering of relations in such data can introduce higher-order, non-dyadic dependencies that are not captured by state-of-the-art graph representations. This oversimplification questions the validity of graph mining techniques in time series data and poses a threat for interdisciplinary applications of network analytics.


To address this challenge, researchers have developed advanced graph modelling and representation techniques based on higher- and variable-order Markov models, which enable us to model non-Markovian characteristics in time series data on networks. Introducing this exciting research field, the goal of this tutorial is to give an overview of cutting-edge higher-order data analytics techniques. Key takeaways for attendees will be (i) a solid understanding of higher-order network modelling and representation learning techniques, (ii) hands-on experience with state-of-the-art higher-order network analytics and visualisation packages, and (iii) a clear demonstration of the benefits of higher-order data analytics in real-world time series data on technical, social, and ecological systems.

A detailed summary of the topics, literature, and tools covered in this hands-on tutorial can be found in the [tutorial paper](https://www.researchgate.net/publication/325168357_Beyond_Graph_Mining_Higher-Order_Data_Analytics_for_Temporal_Network_Data).

# Prerequisites

A detailed description on how to set up the environment can be found in the [setup instructions](/infomap-notebooks/setup).

## Introduction to Higher-Order Graph Clustering with [Infomap](http://www.mapequation.org)

Unit | Topic | Notebook
----|----|----
1.1 | Introduction to [`Infomap`](http://www.mapequation.org) | [.ipynb](https://github.com/mapequation/infomap-notebooks/blob/master/1_1_infomap_intro.ipynb) |
1.2 | Explore flight path data with Infomap and [`interactive visualisations`](http://www.mapequation.org/apps.html) | [.ipynb](https://github.com/mapequation/infomap-notebooks/blob/master/1_2_explore_flight_data.ipynb) |
1.3 | Introduction to [`sparse higher-order networks`](http://www.mapequation.org/publications.html#Edler-Etal-2017-MappingHigherOrder) | [.ipynb](https://github.com/mapequation/infomap-notebooks/blob/master/1_3_sparse_state_lumping.ipynb) |
1.4 | Sparse networks for flight data | [.ipynb](https://github.com/mapequation/infomap-notebooks/blob/master/1_4_sparse_flight_data.ipynb) |

