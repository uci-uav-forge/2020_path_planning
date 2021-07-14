from networkx import nx
import numpy as np
from numpy.random import f
from pathplanner import bdc
import pytest


@pytest.mark.parametrize(
   ("test_input, test_output"),
   [
      (np.random.uniform(1e-4, np.pi*2-1e-4), False),
      (2*np.pi, True),
      (0, True)
   ]
)
def test_rotate_graph(test_input, test_output):
   nodes = [(i, {'points': np.random.normal(size=(2,))}) for i in range(100)]

   edges = []
   for i in range(100):
      n1 = np.random.choice(range(100))
      n2 = np.random.choice(range(100))
      edges.append((n1, n2))
   
   G = nx.DiGraph()
   G.add_nodes_from(nodes)  
   G.add_edges_from(edges)

   H = bdc.rotate_graph(G, theta=test_input)

   assert len(G.nodes) == len(H.nodes)
   for gn, hn in zip(G.nodes, H.nodes):
      xd = np.abs(H.nodes[hn]['points'][0] - G.nodes[gn]['points'][0])
      yd = np.abs(H.nodes[hn]['points'][1] - G.nodes[gn]['points'][1])
      print(xd, yd)
      
      xtrue = xd < 1e-12
      ytrue = yd < 1e-12
      assert xtrue == test_output
      assert ytrue == test_output