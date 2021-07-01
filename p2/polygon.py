from datetime import datetime
from os import PRIO_USER
from scipy import spatial
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from matplotlib.transforms import offset_copy

def cluster_points(no_clusters=3, cluster_n=10, cluster_size=1, cluster_dist=1):
    ''' generate clusters of points '''
    pts = np.zeros((no_clusters * cluster_n, 2))
    loc = np.array([0,0], dtype='float64')
    for c in range(no_clusters):
        pts[c * cluster_n:(c+1)* cluster_n, :] = np.random.normal(loc=loc, scale=cluster_size, size=(cluster_n, 2))
        loc += np.random.uniform(low=-cluster_dist, high=cluster_dist, size=np.shape(loc))
    return pts

def removable_interiors(dt):
    '''find indices of interior simplices that are safe to remove in dt'''
    # all outer tris no exception
    outer_tris = (dt.neighbors == -1).any(axis=1)
    # this is the set of simplices present in outer triangles
    crit_pts = dt.simplices[outer_tris][dt.neighbors[outer_tris] != -1]
    s_unsafe = np.isin(dt.simplices, crit_pts)
    # get indices of safe/unsafe simplices
    safe, = np.where(s_unsafe.any(axis=1)==False)
    unsafe, = np.where(s_unsafe.any(axis=1)==True)
    return safe, unsafe

def removable_exteriors(dt, points, ax=None):
    ''' find indices to safe '''
    # set of all edge simplices
    et, = np.where( (dt.neighbors == -1).any(axis=1) == True)
    et_idx = np.empty(dt.neighbors[et].shape[0], dtype=bool)

    cps = []
    # find critical points
    for i, e in enumerate(et):
        cp = dt.simplices[e][dt.neighbors[e] != -1]
        if cp.shape[0] == 1:
            cps.extend(list(dt.simplices[e]))
        else:
            cps.extend(list(cp))

    # check critical points
    for i, (cps2, e) in enumerate(zip(dt.neighbors[et] == -1, et)):
        safe = ~np.isin(dt.simplices[e][cps2], cps)
        if safe.shape[0] != 1:
            et_idx[i] = False
        else:
            et_idx[i] = safe[0]

    # plot
    if ax is not None:
        xys = points[dt.simplices[et[et_idx]]].sum(axis=1)/3
        xyu = points[dt.simplices[et[~et_idx]]].sum(axis=1)/3
        ax.plot(xys[:,0], xys[:,1], 'b^')
        ax.plot(xyu[:,0], xyu[:,1], 'r^')

        xyt = points[dt.simplices[et]].sum(axis=1)/3
        for i, e in enumerate(et):
            ax.text(xyt[i,0], xyt[i,1], str(e))

    return et_idx.any(), et[et_idx]

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
    return dt

def ar(M):
    '''aspect ratio'''
    a = M[0,:] - M[1,:]
    b = M[1,:] - M[2,:]
    c = M[2,:] - M[0,:]
    dotself = lambda x: x.dot(x)
    a, b, c = map(dotself, (a, b, c))
    s = (a+b+c)/2.0
    return (a*b*c/(8.0*(s-a)*(s-b)*(s-c))).mean()


def polygon(points: np.array, holes: int, removals=30) -> nx.DiGraph():
    dt = spatial.Delaunay(points)
    # holes
    for i in range(holes):
        safe, _ = removable_interiors(dt)
        if safe.size > 0:
            dt = del_tri(dt, np.random.choice(safe))
    # exteriors
    n=0
    while removable_exteriors(dt, points)[0] and n < removals:
        n+=1
        _, ets = removable_exteriors(dt, points)
        ets = sorted(ets, key=lambda x: ar(points[dt.simplices[x]]))
        dt = del_tri(dt, np.random.choice(ets))

    outers=[]
    for s, n in zip(dt.simplices, dt.neighbors):
        if (n==-1).any():
            # opposite interior
            oppint = tuple(s[n != -1])
            # doggie ear case
            if len(oppint) == 1:
                oppint, = oppint
                u, w = tuple(s[s != oppint])
                outers.append((u, oppint))
                outers.append((oppint, w))
            elif len(oppint) == 2:
                outers.append(oppint)
            # not possible
            else:
                raise(Exception('Tri identified as outside that is not on the outside!'))
 
    G = nx.Graph()
    G.add_edges_from(outers)
    H = nx.DiGraph()
    H.add_edges_from(list(nx.edge_dfs(G)))
    for n in H.nodes:
        H.nodes[n]['points'] = points[n]

    leftpt = sorted(H.nodes, key = lambda n: points[n][0])
    # for each loop 
    outputgraphs = []
    for cyc in nx.simple_cycles(H):
        M = H.subgraph(cyc).copy()
        cw = 0
        # outer has leftmost point
        if leftpt[0] in cyc:
            # set both nodes and edges
            for (e1, e2), c in zip(M.edges, cyc):
                outer = True
                cw += addcw(H, e1, e2)
        else:
            for (e1, e2), c in zip(M.edges, cyc):
                outer = False
                cw += addcw(H, e1, e2)
        cw = cw >= 0
        M = H.subgraph(cyc).copy()
        if cw and outer:
            outputgraphs.append(M)
        elif not cw and outer:
            outputgraphs.append(nx.reverse(M, copy=True))
        elif not cw and not outer:
            outputgraphs.append(M)
        elif cw and not outer:
            outputgraphs.append(nx.reverse(M, copy=True))
    return nx.compose_all(outputgraphs)
    
def addcw(H, e1, e2):
    p1, p2 = H.nodes[e1]['points'], H.nodes[e2]['points']
    return (p2[0] - p1[0]) * (p2[1] + p1[1])


def draw_G(G, ax, posattr='points'):
    pos = nx.get_node_attributes(G, 'points')
    nx.draw_networkx_edges(G, pos, ax=ax,
        node_size=4)
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_shape='.',
        node_color='k',
        node_size=30,)
    ax.autoscale(tight=False)
    return ax

def chart(points, G, ax, cm=plt.cm.tab10):
    '''
    Plot the graph `G` on `ax`
    '''
    pos = {}
    for n in G.nodes:
        pos[n] = points[n]
    edges = G.edges
    nx.draw(G, pos, node_size=16, ax=ax, edgelist=edges, edge_cmap=cm)
    ax.set_aspect('equal')
    def offset(ax, x, y):
        return offset_copy(ax.transData, x=x, y=y, units='dots')
    for n in G.nodes:
        x, y = points[n][0], points[n][1]
        ax.text(x, y, str(n), fontsize=9, transform=offset(ax, 0, 5))

if __name__ == '__main__':
    points = np.random.poisson(lam=100, size=(180, 2))
    points = np.random.normal(size=(180, 2))
    G = polygon(points, 8)
    fig, ax = plt.subplots()
    draw_G(G, ax)
    plt.show()