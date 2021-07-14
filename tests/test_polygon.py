from scipy.spatial.qhull import QhullError
from pathplanner import polygon
from scipy import spatial
import numpy as np
import pytest
from contextlib import contextmanager

def test_cluster_points():
   points = polygon.beta_clusters(clusters=5, ppc=10)
   assert(points.shape == (50,2))

def test_del_tri():
   dt = spatial.Delaunay(np.random.beta(4, 4, size=(100,2)))
   rm = np.random.randint(100)
   s_shape, n_shape = dt.simplices.shape, dt.neighbors.shape
   dt = polygon.del_tri(dt, rm)
   assert dt.simplices.shape[0] == s_shape[0] - 1
   assert dt.neighbors.shape[0] == n_shape[0] - 1

def test_created_polygon():
   G = polygon.polygon(points=np.random.beta(4,4,size=(100,2)), holes=1, removals=30)
   for n in G.nodes:
      assert G.nodes[n]['points'] is not None
   for e1, e2 in G.edges:
      assert G[e1][e2]['weight'] is not None

@contextmanager
def does_not_raise():
   yield

@pytest.mark.parametrize(
   ("test_input, expectation"),
   [(i, does_not_raise()) for i in range(4, 9)]
   + [
      (3, pytest.raises(Exception)),
      (2, pytest.raises(QhullError)),
      (1, pytest.raises(QhullError)),
      (0, pytest.raises(ValueError))
   ]
)
def test_polygon_creation(test_input, expectation):
   points = np.random.beta(4,4,size=(test_input, 2))
   with expectation:
      polygon.polygon(points)
