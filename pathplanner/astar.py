from collections import defaultdict
import numpy as np
import networkx as nx
from scipy import spatial

def h(P: nx.Graph, n1: int, n2: int):
    p1, p2 = P.nodes[n1]['points'], P.nodes[n2]['points']
    d = spatial.distance.minkowski(p1, p2, 1)
    d = spatial.distance.euclidean(p1, p2)
    return d

def astar(P: nx.Graph, start: tuple, goal: tuple):
    openset = {start}
    camefrom = {}

    gscore = defaultdict(lambda: 1.0e10)
    gscore[start] = 0

    fscore = defaultdict(lambda: 1.0e10)
    fscore[start] = h(P, start, goal)

    while len(openset) != 0:
        current = min(openset, key=lambda t: fscore[t])

        if current == goal:
            return reconstruct_path(camefrom, current)
        openset.remove(current)
        
        for n in P.neighbors(current):
            tentative_gscore = gscore[current] + h(P, current, n)
            if tentative_gscore < gscore[n]:
                camefrom[n] = current
                gscore[n] = tentative_gscore
                fscore[n] = gscore[n] + h(P, n, goal)
                if n not in openset:
                    openset.add(n)
    raise Exception('failed to find path!')

def reconstruct_path(camefrom, current):
    total = [current]
    while current in camefrom:
        current = camefrom[current]
        total.append(current)
    return total