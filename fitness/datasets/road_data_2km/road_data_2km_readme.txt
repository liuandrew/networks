The dataset is the one used in Fig. 4 of http://link.aps.org/doi/10.1103/PhysRevLett.108.128701, and we chose 20 largest cities in terms of population for US, Europe, Asia, Latin America, and Africa (so 100 cities in total) according to Wikipedia. We used the Merkaator program http://merkaartor.be/ to (manually) extract road structures inside a 2 km * 2km of representative area for each city (the data is freely available under the OpenStreetMap project: http://en.wikipedia.org/wiki/OpenStreetMap )

If you unzip the attached file, directories are named as the five continents, and for each city in the directories, there are three files:

(1) [city_name]_networkx_unit_node.txt: node data
#node_ID longitude latitude
...
[each node's longitude and latitude, along with the unique
node ID (not necessarily from 0 or 1) that will be used in the edge data]

(2) [city_name]_networkx_unit_edge.txt: edge data
#node_ID_1 node_ID_2
(node_ID_1 is connected to node_ID_2)

(3) [city_name]_networkx_unit.pdf: pdf file for the illustrated
structure, where x (y) axis is the longitude (latitude), respectively.
