import heapq

"""
Krushais algorithm and Prims algorithm both differ in the ways in which they find and add their edge cases.
In each section below I have given a brief description as to both Algorithms
"""


def setting(v, big, seed):
    big[v] = v
    seed[v] = 0


def spot(v, big):
    if big[v] != v:
        big[v] = spot(big[v], big)
    return big[v]


def link(v1, v2, big, seed):
    origin1 = spot(v1, big)
    origin2 = spot(v2, big)
    if origin1 != origin2:
        if seed[origin1] > seed[origin2]:
            big[origin2] = origin1
        else:
            big[origin1] = origin2
        if seed[origin1] == seed[origin2]:
            seed[origin2] += 1


def Prims(G, begin):
    """
    Prims algorithm is seen through its root vertex that views all edges and builds
    a full minimum spanning tree. The edges are then selected with the smallest weight that is still attached
    within the minimum spanning tree.
    """
    MST = set()  # Return list of tuples
    done = {begin}  # Letter and cost of edges from beginning of vertex
    ends = [(w, begin, found)
            for found, w in G[begin].items()]
    heapq.heapify(ends)  # Convert list into heap
    while len(ends) > 0:
        e = heapq.heappop(ends)  # Find edge that has smaller weight
        w, branch, found = e
        if found not in done:  # See if vertex has been visited
            done.add(found)  # If not then add
            MST.add(e)
            for f_second, w in G[found].items():
                if f_second not in done:
                    heapq.heappush(ends, (w, found, f_second))
                return sorted(MST)


def Kruskal(G):
    """
    Kruskai's algorithm is set with its vertices in the tree. The edges are then checked to their
    minimum weight.
    """
    MST = set()
    big = dict()
    seed = dict()
    for v in G['Vertex']:
        setting(v, big, seed)
    ends = sorted(list(G['Edges']))
    for e in ends:
        w, v1, v2 = e
        if spot(v1, big) != spot(v2, big):  # Link v1 and v2 and add to MST
            link(v1, v2, big, seed)
            MST.add(e)
    return sorted(MST)


if __name__ == "__main__":
    prims_dict = {'A': {'B': 2, 'C': 3}, 'B': {'A': 2, 'C': 1, 'D': 3, 'E': 2},
                  'C': {'A': 3, 'B': 1, 'E': 1}, 'D': {'B': 3, 'E': 5}, 'E': {'B': 2, 'C': 1, 'D': 5}}
print("Prims Algorithm Results")
print(Prims(prims_dict, 'A'))
print(Prims(prims_dict, 'B'))
print(Prims(prims_dict, 'C'))
print(Prims(prims_dict, 'D'))
print(Prims(prims_dict, 'E'))

kruskal_dict = {
    'Vertex': ['A', 'B', 'C', 'D', 'E'],
    'Edges': {(1, 'C', 'E'), (1, 'B', 'C'), (2, 'A', 'B'), (2, 'B', 'E'), (3, 'A', 'C'), (3, 'B', 'D'), (5, 'D', 'E')}}
print("Kruskal's Algorithm Results")
print(Kruskal(kruskal_dict))

'''
Cited Sources:
Author: DURepo
Date: Exploration Practice
URL: https://github.com/DURepo/CS_325_Exercises/blob/main/Graph-prims.py
Code: 
def prims(G):
    # Prim's Algorithm in Python


    INF = 9999999
    # number of vertices in graph
    V = 5
    # create a 2d array of size 5x5
    # for adjacency matrix to represent graph

    # create a array to track selected vertex
    # selected will become true otherwise false
    selected = [0, 0, 0, 0, 0]
    # set number of edge to 0
    no_edge = 0
    # the number of egde in minimum spanning tree will be
    # always less than(V - 1), where V is number of vertices in
    # graph
    # choose 0th vertex and make it true
    selected[0] = True
    # print for edge and weight
    print("Edge : Weight\n")
    while (no_edge < V - 1):
        # For every vertex in the set S, find the all adjacent vertices
        # , calculate the distance from the vertex selected at step 1.
        # if the vertex is already in the set S, discard it otherwise
        # choose another vertex nearest to selected vertex  at step 1.
        minimum = INF
        x = 0
        y = 0
        for i in range(V):
            if selected[i]:
                for j in range(V):
                    if ((not selected[j]) and G[i][j]):
                        # not in selected and there is an edge
                        if minimum > G[i][j]:
                            minimum = G[i][j]
                            x = i
                            y = j
        print(str(x) + "-" + str(y) + ":" + str(G[x][y]))
        selected[y] = True
        no_edge += 1

G = [[0, 9, 75, 0, 0],
         [9, 0, 95, 19, 42],
         [75, 95, 0, 51, 66],
         [0, 19, 51, 0, 31],
         [0, 42, 66, 31, 0]]

prims(G)
-------------------------------------------------------------------------
Author: GeeksforGeeks - Divyanshu Mehta
Date: Jan 18, 2022
URL: https://www.geeksforgeeks.org/prims-minimum-spanning-tree-mst-greedy-algo-5/
Code:
import sys # Library for INT_MAX

class Graph():

    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                    for row in range(vertices)]

    # A utility function to print the constructed MST stored in parent[]
    def printMST(self, parent):
        print ("Edge \tWeight")
        for i in range(1, self.V):
            print (parent[i], "-", i, "\t", self.graph[i][parent[i]])

    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):

        # Initialize min value
        min = sys.maxsize

        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v

        return min_index

    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):

        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V

        parent[0] = -1 # First node is always the root of

        for cout in range(self.V):

            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)

            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True

            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):

                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False and key[v] > self.graph[u][v]:
                        key[v] = self.graph[u][v]
                        parent[v] = u

        self.printMST(parent)

g = Graph(5)
g.graph = [ [0, 2, 0, 6, 0],
            [2, 0, 3, 8, 5],
            [0, 3, 0, 0, 7],
            [6, 8, 0, 0, 9],
            [0, 5, 7, 9, 0]]

g.primMST();
-----------------------------------------------------------------------------------------
Author: GeeksforGeeks - Neelam Yadav
Date: Feb 11, 2022
URL: https://www.geeksforgeeks.org/kruskals-minimum-spanning-tree-algorithm-greedy-algo-2/
Code:
from collections import defaultdict

# Class to represent a graph


class Graph:

    def __init__(self, vertices):
        self.V = vertices  # No. of vertices
        self.graph = []  # default dictionary
        # to store graph

    # function to add an edge to graph
    def addEdge(self, u, v, w):
        self.graph.append([u, v, w])

    # A utility function to find set of an element i
    # (uses path compression technique)
    def find(self, parent, i):
        if parent[i] == i:
            return i
        return self.find(parent, parent[i])

    # A function that does union of two sets of x and y
    # (uses union by rank)
    def union(self, parent, rank, x, y):
        xroot = self.find(parent, x)
        yroot = self.find(parent, y)

        # Attach smaller rank tree under root of
        # high rank tree (Union by Rank)
        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        # If ranks are same, then make one as root
        # and increment its rank by one
        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    # The main function to construct MST using Kruskal's
        # algorithm
    def KruskalMST(self):

        result = []  # This will store the resultant MST

        # An index variable, used for sorted edges
        i = 0

        # An index variable, used for result[]
        e = 0

        # Step 1:  Sort all the edges in
        # non-decreasing order of their
        # weight.  If we are not allowed to change the
        # given graph, we can create a copy of graph
        self.graph = sorted(self.graph,
                            key=lambda item: item[2])

        parent = []
        rank = []

        # Create V subsets with single elements
        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        # Number of edges to be taken is equal to V-1
        while e < self.V - 1:

            # Step 2: Pick the smallest edge and increment
            # the index for next iteration
            u, v, w = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            # If including this edge does't
            #  cause cycle, include it in result
            #  and increment the indexof result
            # for next edge
            if x != y:
                e = e + 1
                result.append([u, v, w])
                self.union(parent, rank, x, y)
            # Else discard the edge

        minimumCost = 0
        print ("Edges in the constructed MST")
        for u, v, weight in result:
            minimumCost += weight
            print("%d -- %d == %d" % (u, v, weight))
        print("Minimum Spanning Tree" , minimumCost)

# Driver code
g = Graph(4)
g.addEdge(0, 1, 10)
g.addEdge(0, 2, 6)
g.addEdge(0, 3, 5)
g.addEdge(1, 3, 15)
g.addEdge(2, 3, 4)

# Function call
g.KruskalMST()
-------------------------------------------------------------------------
Author: License: Creative Commons -Attribution -ShareAlike 4.0 (CC-BY-SA 4.0)
Date: N/A
URL: https://www.educative.io/edpresso/how-to-implement-a-graph-in-python
Code:
def add_vertex(v):
  global graph
  global vertices_no
  global vertices
  if v in vertices:
    print("Vertex ", v, " already exists")
  else:
    vertices_no = vertices_no + 1
    vertices.append(v)
    if vertices_no > 1:
        for vertex in graph:
            vertex.append(0)
    temp = []
    for i in range(vertices_no):
        temp.append(0)
    graph.append(temp)

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
    global graph
    global vertices_no
    global vertices
    # Check if vertex v1 is a valid vertex
    if v1 not in vertices:
        print("Vertex ", v1, " does not exist.")
    # Check if vertex v1 is a valid vertex
    elif v2 not in vertices:
        print("Vertex ", v2, " does not exist.")
    # Since this code is not restricted to a directed or 
    # an undirected graph, an edge between v1 v2 does not
    # imply that an edge exists between v2 and v1
    else:
        index1 = vertices.index(v1)
        index2 = vertices.index(v2)
        graph[index1][index2] = e

# Print the graph
def print_graph():
  global graph
  global vertices_no
  for i in range(vertices_no):
    for j in range(vertices_no):
      if graph[i][j] != 0:
        print(vertices[i], " -> ", vertices[j], \
        " edge weight: ", graph[i][j])

# Driver code        
# stores the vertices in the graph
vertices = []
# stores the number of vertices in the graph
vertices_no = 0
graph = []
# Add vertices to the graph
add_vertex(1)
add_vertex(2)
add_vertex(3)
add_vertex(4)
# Add the edges between the vertices by specifying
# the from and to vertex along with the edge weights.
add_edge(1, 2, 1)
add_edge(1, 3, 1)
add_edge(2, 3, 3)
add_edge(3, 4, 4)
add_edge(4, 1, 5)
print_graph()
print("Internal representation: ", graph)
-------------------------------------------
Code:
# Add a vertex to the dictionary
def add_vertex(v):
  global graph
  global vertices_no
  if v in graph:
    print("Vertex ", v, " already exists.")
  else:
    vertices_no = vertices_no + 1
    graph[v] = []

# Add an edge between vertex v1 and v2 with edge weight e
def add_edge(v1, v2, e):
  global graph
  # Check if vertex v1 is a valid vertex
  if v1 not in graph:
    print("Vertex ", v1, " does not exist.")
  # Check if vertex v2 is a valid vertex
  elif v2 not in graph:
    print("Vertex ", v2, " does not exist.")
  else:
    # Since this code is not restricted to a directed or 
    # an undirected graph, an edge between v1 v2 does not
    # imply that an edge exists between v2 and v1
    temp = [v2, e]
    graph[v1].append(temp)

# Print the graph
def print_graph():
  global graph
  for vertex in graph:
    for edges in graph[vertex]:
      print(vertex, " -> ", edges[0], " edge weight: ", edges[1])

# driver code
graph = {}
# stores the number of vertices in the graph
vertices_no = 0
add_vertex(1)
add_vertex(2)
add_vertex(3)
add_vertex(4)
# Add the edges between the vertices by specifying
# the from and to vertex along with the edge weights.
add_edge(1, 2, 1)
add_edge(1, 3, 1)
add_edge(2, 3, 3)
add_edge(3, 4, 4)
add_edge(4, 1, 5)
print_graph()
# Reminder: the second element of each list inside the dictionary
# denotes the edge weight.
print ("Internal representation: ", graph)
------------------------------------------------------------------------------

'''
