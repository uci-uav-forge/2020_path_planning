from scipy import spatial
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
import copy

def beta_clusters(clusters: int=3, ppc: int=20, alpha:float=4.0, beta: float=4.0) -> np.ndarray:
    pts = np.zeros((clusters*ppc, 2))
    loc = np.array([0,0], dtype=float)
    for c in range(clusters):
        # each cluster is a "block" in the pts array between (c) : (c+1) multiplied by ppc.
        pts[c*ppc : (c+1)*ppc] = np.random.beta(a=alpha, b=beta, size=(ppc, 2) ) + loc
        # move center point of cluster
        loc += np.random.uniform(low=-1, high=1, size=loc.shape)
    return pts

def remove_close_points(points: np.ndarray, eps: float = 0.07):
    def dist(p1, p2): 
        line = p2-p1
        return np.linalg.norm(line)
    del_list = []
    for i, p1 in enumerate(points):
        for j, p2 in enumerate(points[i:]):
            j += i
            if dist(p1, p2) < eps and i != j:
                avg = (p2 + p1)/2
                points[i,:] = avg
                del_list.append(j)
    mask = np.ones(points.shape[0], dtype=bool)
    mask[del_list] = False
    points = points[mask]
    if points.shape[0] < 3:
        raise ValueError('Deleted too many close points! Epsilon may be set too high.')
    return points

def removable_interiors(dt: spatial.Delaunay) -> tuple:
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

def removable_exteriors(dt: spatial.Delaunay, points: np.ndarray, ax: plt.Axes=None) -> tuple:
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

def del_tri(dt:spatial.Delaunay, rm: int) -> spatial.Delaunay:
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

def _dotself(x: np.ndarray) -> np.ndarray:
    return x.dot(x)

def ar(M: np.ndarray) -> np.float64:
    '''aspect ratio'''
    # a, b, c are the 3 vectors which form the triangle
    a = M[0,:] - M[1,:]
    b = M[1,:] - M[2,:]
    c = M[2,:] - M[0,:]
    # dot each with self
    a, b, c = map(_dotself, (a, b, c))
    # half chord
    s = (a+b+c)/2.0
    # https://codegolf.stackexchange.com/questions/101234/evaluate-the-aspect-ratio-of-a-triangle
    ar = (a*b*c/(8.0*(s-a)*(s-b)*(s-c))).mean()
    return ar


def polygon(points: np.ndarray, holes: int = 0, removals: int = 30) -> nx.DiGraph():
    '''Create a non-convex polygon from points. This algorithm works by first creating a convex
    Delaunay triangulation over the points, and then removing triangles to form a non-convex 
    polygon. Thus, the general shape of the initial point distribution will infrom the subsequent
    non-convex polygon.

    This algorithm is slow for large numbers of points.

    Parameters
    ----------
    points : np.ndarray, holes, optional
        The number of interior holes in the polygon, default 0

    Returns
    -------
    nx.Digraph
        Graph containing the polygon.
    '''
    dt = spatial.Delaunay(points)
    dt_orig = copy.deepcopy(dt)
    # remove holes
    for _ in range(holes):
        safe, _ = removable_interiors(dt)
        if safe.size > 0:
            dt = del_tri(dt, np.random.choice(safe))
    # remove exterior tris
    n=0
    while removable_exteriors(dt, points)[0] and n < removals:
        n+=1
        _, ets = removable_exteriors(dt, points)
        ets = sorted(ets, key=lambda x: ar(points[dt.simplices[x]]))
        dt = del_tri(dt, np.random.choice(ets))
    # traverse and find outer edges
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
    # first we make an undirected graph, then we create the directed graph by traversing the 
    # undirected in a DFS.
    G = nx.Graph()
    G.add_edges_from(outers)
    H = nx.DiGraph()
    H.add_edges_from(list(nx.edge_dfs(G)))
    # store point coordinates into each node
    for n in H.nodes:
        H.nodes[n]['points'] = points[n]
    # leftmost point is guaranteed to be on the outer boundary, therefore any node connected to it
    # is also on the outer boundary (and any node not connected to it is on an interior boundary.)
    leftpt = sorted(H.nodes, key = lambda n: points[n][0])
    # we are going to break the whole graph down into subgraphs (which are simple cycles) so that we
    # can orient each cycle depending on whether it is on the outside or not
    outputgraphs = []
    for cyc in nx.simple_cycles(H):
        M = H.subgraph(cyc).copy()
        cw = 0
        # outer has leftmost point
        if leftpt[0] in cyc:
            # set both nodes and edges
            for (e1, e2), c in zip(M.edges, cyc):
                outer = True
                M[e1][e2]['weight'] = 1
                cw += addcw(H, e1, e2)
        else:
            for (e1, e2), c in zip(M.edges, cyc):
                M[e1][e2]['weight'] = 2
                outer = False
                cw += addcw(H, e1, e2)
        cw = cw >= 0        
        # categorize nodes
        # append
        if cw and outer:
            outputgraphs.append(M)
        elif not cw and outer:
            outputgraphs.append(nx.reverse(M, copy=True))
        elif not cw and not outer:
            outputgraphs.append(M)
        elif cw and not outer:
            outputgraphs.append(nx.reverse(M, copy=True))
    out_graph = nx.compose_all(outputgraphs)
    return out_graph, dt, dt_orig
    
def addcw(H: nx.DiGraph, e1: int, e2: int) -> float:
    '''determine which way the edge is pointing
             Q2  │  Q1
            ─────┼─────
             Q3  │  Q4
    A "positive" cw value goes from Q2 to Q4. A "negative"
    cw value means the edge goes from Q4 to Q2. It's relative
    so we only look at directions not absolute positions.
    '''
    p1, p2 = H.nodes[e1]['points'], H.nodes[e2]['points']
    return (p2[0] - p1[0]) * (p2[1] + p1[1])


def stupid_spiky_polygon(R_inner:int, R_outer:int, n=10, ):
    delta = 2 * np.pi / n
    theta = 0
    points = []
    while theta < 2 * np.pi:
        r = np.random.uniform(R_inner,R_outer)
        points.append([r,theta])
        theta += delta
    points = np.array(points)
    G = nx.Graph()
    def polar2cart(p):
        x = p[0] * np.cos(p[1])
        y = p[0] * np.sin(p[1])
        return np.array([x,y])
    for i, p in enumerate(points):
        G.add_node(i, points=polar2cart(p))
    for i in range(points.shape[0]):
        if i == points.shape[0]-1:
            G.add_edge(i, 0)
        else:
            G.add_edge(i, i+1)
    return points, G
        


def draw_G(
    G: nx.DiGraph, 
    ax: plt.Axes, 
    posattr: str ='points', 
    arrows: bool =False, 
    nodecolor: str = 'k', 
    ecolor: str = None,
    ecolorattr: str = 'weight',
    ecmap = 'tab10',
    node_text: bool = True,
    style: str = '-',
    m: str = '.',
    draw_nodes = True,
    ) -> plt.Axes:
    '''Draw a DiGraph `G` with points stored in `posattr` onto `ax`'''
    pos = nx.get_node_attributes(G, posattr)
    if not ecolor:
        try:
            ecolor = [G[u][v][ecolorattr] for u, v in G.edges()]
        except:
            ecolor = [5 for _ in G.edges()]
    nx.draw_networkx_edges(G, pos, ax=ax,
        node_size=4,
        arrows=arrows,
        edge_color=ecolor,
        style=style,
        edge_cmap=plt.get_cmap(ecmap))
    if draw_nodes:
        nx.draw_networkx_nodes(
            G, pos, ax=ax,
            node_shape=m,
            node_color=nodecolor,
            node_size=30,)
    if node_text:
        for n in G.nodes: ax.text(pos[n][0], pos[n][1], s=str(n))
    ax.autoscale(tight=False)
    return ax
