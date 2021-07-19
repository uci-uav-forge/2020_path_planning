import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from shapely import geometry

def contract_cell(cell: np.ndarray, ax):
   poly = geometry.Polygon(shell=cell)
   x, y = poly.exterior.xy
   ax.plot(x,y)
   # shrink_or_swell_shapely_polygon(poly, ax)


def get_cells(R: nx.DiGraph, J: nx.DiGraph):
   for n in R.nodes:
      # each node of reeb graph has a cell which is a collection of nodes in J
      # that point to a specific point 'points', which form the basis of this array
      cell_pts = [J.nodes[p]['points'] for p in J.subgraph(R.nodes[n]['cell']).nodes]
      yield np.array(cell_pts)

def shrink_or_swell_shapely_polygon(my_polygon, ax, factor=0.10, swell=False):
   ''' returns the shapely polygon which is smaller or bigger by passed factor.
      If swell = True , then it returns bigger polygon, else smaller '''
   #my_polygon = mask2poly['geometry'][120]
   shrink_factor = 0.10 #Shrink by 10%
   xs = list(my_polygon.exterior.coords.xy[0])
   ys = list(my_polygon.exterior.coords.xy[1])
   x_center = 0.5 * min(xs) + 0.5 * max(xs)
   y_center = 0.5 * min(ys) + 0.5 * max(ys)
   min_corner = geometry.Point(min(xs), min(ys))
   max_corner = geometry.Point(max(xs), max(ys))
   center = geometry.Point(x_center, y_center)
   shrink_distance = center.distance(min_corner)*0.10

   if swell:
      my_polygon_resized = my_polygon.buffer(shrink_distance) #expand
   else:
      my_polygon_resized = my_polygon.buffer(-shrink_distance) #shrink

   x, y = my_polygon.exterior.xy
   ax.plot(x,y)
   x, y = my_polygon_resized.exterior.xy
   ax.plot(x,y)

   return my_polygon_resized