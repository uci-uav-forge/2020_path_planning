'''
Polygon Generation Module

This module generates random "interesting" (non-convex) polygons. It
works by building a cluster of [x,y] point pairs, calculating a Delaunay
Triangulation of the cluster, and randomly removing traingles from the
edge of the cluster. Degenerate polygons are avoided:

    - Polygons with only a single point touching two parts of the poly
    - Polygons which are actually split into multiple polygons
    - Polygons whose edges cross over one another

The choice of clustering algorithm influences how polygons are formed,
because they influence initial Delaunay Triangulation.

to use, call gen_poly. The algorithm is something like n^3 so be sparing with points.

Written by Mike Sutherland
'''
import numpy as np
import networkx as nx
from scipy import spatial
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

def cluster_points(no_clusters=3, cluster_n=10, cluster_size=1, cluster_dist=1):
    ''' generate clusters of points '''
    pts = np.zeros((no_clusters * cluster_n, 2))
    loc = np.array([0,0], dtype='float64')
    for c in range(no_clusters):
        pts[c * cluster_n:(c+1)* cluster_n, :] = np.random.normal(loc=loc, scale=cluster_size, size=(cluster_n, 2))
        loc += np.random.uniform(low=-cluster_dist, high=cluster_dist, size=np.shape(loc))
    return pts

def aspect(dt, tri):
    pts = dt.points[dt.simplices[tri]]
    v1 = pts[0,:] - pts[2,:]
    v2 = pts[1,:] - pts[0,:]
    v3 = pts[2,:] - pts[1,:]
    a = np.sqrt(v1.dot(v1.T))
    b = np.sqrt(v2.dot(v2.T))
    c = np.sqrt(v3.dot(v3.T))
    return a * b * c / ( (b + c - a) * (c + a - b) * (a + b - c) )

def count_tris(dt):
    ''' count tris of each type '''
    e0, e1, e2 = 0, 0, 0
    for s in dt.neighbors:
        c = (s == -1).sum()
        if c == 0:
            e0 += 1
        elif c == 1:
            e1 += 1
        elif c == 2:
            e2 += 1
    return e0, e1, e2
    
def count_outer_edges(dt):
    ''' return the number of edges '''
    _, e1, e2 = count_tris(dt)
    return e1 + 2*e2

def id_bps(dt):
    '''Identify points within 2 of a boundary.'''
    unsafe = set()
    for neigh, tri in zip(dt.neighbors, dt.simplices):
        if -1 in neigh:
            # all points in this tri
            unsafe.update(set([x for x in tri]))
            for neigh2 in [x for x in neigh if x != -1]:
                # all points in neighboring tris also
                unsafe.update(set(dt.simplices[neigh2].flatten()))
    return set(list(range(len(dt.points)))) - unsafe, unsafe

def id_tris(dt):
    ''' ???? '''
    unsafe = set()
    for i, neigh in enumerate(dt.neighbors):
        if (neigh == -1).sum() > 0:
            unsafe.add(i)
    return set(list(range(len(dt.simplices)))) - unsafe, unsafe

def id_tris2(dt):
    '''Identify tris that have points within a single point of a boundary.'''
    unsafe_tris, safe_tris = set(), set()
    safe_pts, unsafe_pts = id_bps(dt)
    for i, tri in enumerate(dt.simplices):
        if set(tri).issubset(safe_pts):
            safe_tris.add(i)
        else:
            unsafe_tris.add(i)
    return safe_tris, unsafe_tris

def centroid(pts):
    ''' find centroid of Mx2 points '''
    k = np.shape(pts)[0]
    return np.array(np.sum(pts, axis=0)/k, dtype='float64')

def del_tri(dt, rm) -> None:
    ''' Alters dt in place to remove tri at `rm` ''' 
    # each neighbor in dt will have -1 where the neighboring simplex used to be.
    for i, nb in enumerate(dt.neighbors):
        for j, s in enumerate(nb):
            if rm == s:
                dt.neighbors[i,j] = -1
    # we have to decrement all references to simplexes above rm because we're going to remove that simplex
    decrement_idx = np.zeros(np.shape(dt.neighbors), dtype='int32')
    for i, nb in enumerate(dt.neighbors):
        for j, s in enumerate(nb):
            if dt.neighbors[i,j] > rm:
                decrement_idx[i,j] = -1
    dt.neighbors += decrement_idx
    # perform the deletion
    dt.simplices = np.delete(dt.simplices, rm, axis=0)
    dt.neighbors = np.delete(dt.neighbors, rm, axis=0)

def check_edgesafety(dt, rm_candidate, edge_tris) -> bool:
    critpts = dt.simplices[rm_candidate, (dt.neighbors[rm_candidate,:] == -1)]
    unsafe = id_tris(dt)[1]
    check = True
    # check all unsafe triangles for that critical point
    for utri in unsafe:
        for pt in critpts:
            if utri != rm_candidate and pt in dt.simplices[utri]:
                check = False
    return check

def shape_graph(points, blist):
    '''make the graphs that define the polygon'''
    graphs = []
    for boundary, outer in blist:
        G = nx.DiGraph()
        elist, nlist = [], []
        
        # check if we flip 
        cw = 0.0
        for k, v in boundary.items():
            x1, y1 = tuple(points[k])
            x2, y2 = tuple(points[v[0]])
            cw += (x2 - x1) * (y2 + y1)
        if cw >= 0 and outer:
            flip = False
            weight = int(1)
        elif cw >= 0 and not outer:
            weight = int(2)
            flip = True
        elif cw < 0 and outer:
            weight = int(1)
            flip = True
        else:
            weight = int(2)
            flip = False
        # build the boundary
        for k, v in boundary.items():
            nlist.append(k)
            if flip:
                elist.append( (v[0], k, weight) )
            else:
                elist.append( (k, v[0], weight) )

        G.add_nodes_from(nlist)
        G.add_weighted_edges_from(elist)
        graphs.append(G)
    G = nx.compose_all(graphs)
    return G
     

def order_boundaries(dt):
    '''Order the boundaries by their topological structure'''
    is_outer = np.asarray(dt.neighbors == -1, dtype='bool')
    tris = [list(x.compressed()) for x in np.ma.MaskedArray(dt.simplices, is_outer)]
    graph = {}
    # create a graph of nodes, which are indices to points:
    # child nodes are the neighbors of parent nodes
    for i, n in enumerate(tris):
        for j, m in enumerate(tris):
            if len(n) < 3 and len(m) < 3 and i != j:                    
                if set(n) & set(m) != set():
                    key = list(set(n) & set(m))[0]
                    val = tuple(set(n) ^ set(m))
                    graph[key] = val

    ordered_objs = []
    b = set(graph.keys())
    minpt = min(b, key=lambda i: dt.points[i, 0])
    while(len(b) != 0):
        t = set()
        # depth first traversal
        def dfs(g, s, p=[]):
            if s not in p:
                p.append(s)
                if s not in g:
                    return p
                for n in g[s]:
                    p = dfs(g, n, p)
            return p
        # traverse and return as array
        t = dfs(graph, list(b)[0])
        # t.append(t[0])
        st = set(t)
        outer_bound = False
        if minpt in st: 
            outer_bound = True
        ordered_objs.append((t, outer_bound))
        b = b - st
    return ordered_objs

def make_boundaries(dt):
    '''=create boundary dict structure
    keys are boundary point idx
    values are tuple (prev idx, next idx, bool outer)
    '''
    boundaries = []
    for boundary, is_outer in order_boundaries(dt):
        # iterate through each boundary and create dict:
        # key = point
        # values = (prev point, next point, outer True/False)
        bdict = OrderedDict()
        for i, pt in enumerate(boundary):
            if i != 0 and i != len(boundary) - 1:
                bdict[pt] = (boundary[i - 1], boundary[i + 1])
            elif i == 0:
                bdict[pt] = (boundary[-1], boundary[i + 1])
            elif i == len(boundary) - 1:
                bdict[pt] = (boundary[i - 1], boundary[0])
        boundaries.append((bdict, is_outer))
    return boundaries

def gen_poly_dt(points, jaggedness, holes=0):
    '''Generate the polgyon.'''
    dt = spatial.Delaunay(points)
    n_edges = count_outer_edges(dt)
    # holes
    for h in range(holes):
        safe, _ = id_tris2(dt)
        if not safe:
            break
        rm = np.random.choice(list(safe), replace=False)
        del_tri(dt, rm)
    # jaggedness
    edges_desired = int(n_edges * (jaggedness + 1))
    while n_edges < edges_desired:
        rm = None
        # get 1-edge tris by aspect ratio
        etris = sorted(list(id_tris(dt)[1]), key=lambda tri: aspect(dt, tri))
        for etri in etris:
            if check_edgesafety(dt,etri, etris) == True:
                rm = etri
                break
        if rm is not None:
            del_tri(dt, rm)
        else:
            break
        n_edges = count_outer_edges(dt)
    return dt

def plot_tri(dt, ax):
    '''Plot self.dt on mpl axes `ax`

    Parameters
    ----------
    ax : matplotlib `ax` object
        The axis on which to plot
    '''
    centers = np.sum(dt.points[dt.simplices], axis=1, dtype='int')/3.0
    centr = centroid(centers)
    colors = np.array([ (x - centr[0])**2 + (y - centr[1])**2 for x, y in centers])
    ax.tripcolor(dt.points[:,0], dt.points[:,1], dt.simplices, facecolors=colors, cmap='YlGn', edgecolors='darkgrey')
    ax.set_aspect('equal')
    ax.set_facecolor('lightblue')

def chart(points, G, ax, cm=plt.cm.tab10):
    '''
    Plot the graph `G` on `ax`
    '''
    pos = {}
    for n in G.nodes:
        pos[n] = points[n]
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    nx.draw(G, pos, node_size=16, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=cm)
    ax.set_aspect('equal')
    def offset(ax, x, y):
        return offset_copy(ax.transData, x=x, y=y, units='dots')
    for n in G.nodes:
        x, y = points[n][0], points[n][1]
        ax.text(x, y, str(n), fontsize=9, transform=offset(ax, 0, 5))

def gen_poly(points=cluster_points(), jaggedness=2, holes=3):
    polydt = gen_poly_dt(points,jaggedness, holes)
    boundaries = make_boundaries(polydt)
    G = shape_graph(points, boundaries)
    return points, G