from polyskel import polyskel
import networkx as nx
import numpy as np
import copy, enum
import matplotlib.pyplot as plt
from pathplanner import polygon
from shapely import geometry

class Event(enum.Enum):
    CLOSE=1
    OPEN=2
    SPLIT=3
    MERGE=4
    INFLECTION=5
    INTERSECT=6

def rad2degree(rad): return rad*180/np.pi
def degree2rad(deg): return deg*np.pi/180


def discretize(J: nx.DiGraph, R:nx.Graph, gridsz:float, theta=None):
    if theta is not None:
        matr = make_rot_matrix(-theta)
        revmatr = make_rot_matrix(theta)
        K = nx.DiGraph()
        K = copy.deepcopy(J)
        for n in K:
            K.nodes[n]['points'] = matr @ K.nodes[n]['points']
    else:
        matr = np.eye((2,2))
        revmatr = matr
        K = J
    # make grid points
    pts = np.array([K.nodes[n]['points'] for n in K])
    xmin, xmax = np.min(pts[:,0]), np.max(pts[:,0])
    ymin, ymax = np.min(pts[:,1]), np.max(pts[:,1])
    x, y = np.meshgrid(np.arange(xmin, xmax, gridsz), np.arange(ymin, ymax, gridsz))
    P = nx.grid_2d_graph(*x.shape)
    # polygons
    polygons = []
    for n in R.nodes:
        cellpts = np.dot(matr, R.nodes[n]['cellpts'].T).T
        polygons.append(geometry.Polygon(cellpts), n)

    delnodes = []
    for n in P.nodes:
        point = np.array([x[n], y[n]])
        if any([p.contains(geometry.Point(point)) for p, _ in polygons]):
            P.nodes[n]['points'] = np.dot(revmatr, point.T).T
        else:
            delnodes.append(n)
    P.remove_nodes_from(delnodes)
    return P

def iscw(points: np.ndarray) -> bool:
    '''Determine orientation of a set of points

    Parameters
    ----------
    points : np.ndarray
        An ordered set of points

    Returns
    -------
    bool
        True if clockwise, False if not clockwise
    '''
    c = 0
    for i, _ in enumerate(points[:-1,:]):
        p1, p2 = points[i+1,:], points[i,:]
        c += (p2[0] - p1[0]) * (p2[1] + p1[1])
    if c > 0: return True
    else: return False

def line_sweep(G: nx.DiGraph, theta: float = 0, posattr: str = 'points') -> tuple:
    # sorted node list becomes our list of events
    H = rotate_graph(G, theta, posattr=posattr)
    # sort left to right and store sorted node list in L.
    def left_to_right(n): return H.nodes[n][posattr][0]
    L = sorted(list(H.nodes), key=left_to_right)
    # sweep left-to-right
    crits, cells = [], {}
    for v in L:
        etype = check_lu(H, v)
        if etype == Event.SPLIT or etype == Event.MERGE:
            splitmerge_points(H, v)
        # crits
        if etype in (Event.OPEN, Event.SPLIT, Event.MERGE, Event.CLOSE):
            crits.append(v)
    for c in crits:
        cells = append_cell(H, c, cells)
    # rotate the graph back into original orientation
    J = rotate_graph(H, -theta, posattr=posattr)
    # build the reebgraph of J
    R = build_reebgraph(J, cells)
    S = create_skelgraph(R, J)
    return H, J, R, S

def rg_centroid(R: nx.Graph, H: nx.DiGraph, cell: set) -> np.ndarray:
    p = np.zeros((2,), dtype=np.float64)
    for c in cell:
        p += H.nodes[c]['points']
    p /= len(cell)
    return p

def build_reebgraph(H: nx.DiGraph, cells: list) -> None:
    '''wowwww dude refactor this'''
    rgedges, rgcells = [], {}
    for i, a in enumerate(cells.values()):
        for j, b in enumerate(cells.values()):
            union = tuple(set(a) & set(b))
            if len(union) == 2:
                w = None
                try: w=H[union[0]][union[1]]['weight']
                except: pass
                try: w=H[union[1]][union[0]]['weight']
                except: pass
                if w == 3 or w == 4:
                    rgedges.append((i, j, {'shared' : union}))
                    rgcells[i] = a
                    rgcells[j] = b

    R = nx.Graph()
    R.add_edges_from(rgedges)
    for n in R.nodes:
        R.nodes[n]['cell'] = rgcells[n]
        centroid = rg_centroid(R, H, rgcells[n])
        R.nodes[n]['centroid'] = centroid
        cellpts = [H.nodes[c]['points'] for c in rgcells[n]]
        # close the polygon
        cellpts.append(cellpts[0])
        cellpts = np.array(cellpts)
        R.nodes[n]['cellpts'] = cellpts
        
        # plt.plot(cellpts[:,0], cellpts[:,1])
        # triangles can't have straight skeleton
        if len(rgcells[n]) > 3:
            skel = polyskel.skeletonize(cellpts, holes=[])
            skel = fix_polyskel(skel)
            T = traverse_polyskel(R, n, skel)
        else:
            T = nx.Graph()
            T.add_node(0, points=centroid, interior=True, 
            parent=(R.nodes[n]['cell'], R.nodes[n]['cellpts'])
            )
        for e1, e2 in T.edges:
            T[e1][e2]['interior'] = True
        R.nodes[n]['skel_graph'] = T
    return R

def create_skelgraph(R: nx.Graph, H: nx.DiGraph):
    # in preparation to join, make nodes unique
    unique_node = 0
    for n in R.nodes:
        unique_node = remap_nodes_unique(unique_node, R.nodes[n]['skel_graph'])

    visited = set()
    for this, that in nx.eulerian_circuit(nx.eulerize(R)):
        if frozenset((this, that)) not in visited:
            visited.add(frozenset((this, that)))
            # midpoint is the midpoint of the line between cells of 'this' and 'that' nodes
            # on reeb graph. We walk through each transition between cells, finding the midpoint
            # and adding it to the straight skeletons in R.
            midp = get_midpoint_shared(H, R, this, that)
            this_closest = get_closest_on_skel(R, this, midp)
            that_closest = get_closest_on_skel(R, that, midp)
            # get midpoint node
            midp_node = -unique_node
            unique_node += 1
            R.nodes[this]['skel_graph'].add_node(midp_node, interior=False, points=midp, parent=None)
            R.nodes[this]['skel_graph'].add_edge(this_closest, midp_node, interior=False)
            R.nodes[that]['skel_graph'].add_node(midp_node, interior=False, points=midp, parent=None)
            R.nodes[that]['skel_graph'].add_edge(that_closest, midp_node, interior=False)
    # compose all
    for n in R.nodes:
        S = nx.compose_all([R.nodes[n]['skel_graph'] for n in R.nodes])
    for e1, e2 in S.edges:
        S[e1][e2]['distance'] = np.linalg.norm(S.nodes[e1]['points'] - S.nodes[e2]['points'])
    return S
    
def get_closest_on_skel(R: nx.graph, rnode: int, midp):
    this_connectors = []
    for n in R.nodes[rnode]['skel_graph'].nodes:
        if R.nodes[rnode]['skel_graph'].nodes[n]['interior']:
            thispt = R.nodes[rnode]['skel_graph'].nodes[n]['points']
            dist = dotself(midp - thispt)
            this_connectors.append((n, dist))
    # return closest
    return sorted(this_connectors, key=lambda t: t[1])[0][0]


def fix_polyskel(skel: list):
    newskel = []
    def strip(leaf): return np.array(leaf)
    for origin, height, leafs in skel:
        origin = np.array(origin)
        leafs = list(map(strip, leafs))
        newskel.append((origin, height, leafs))
    return newskel

def traverse_polyskel(R: nx.Graph, rn: int, skel: list):
    epsilon = 1e-5
    elist = []
    for i, (pt1, _, _) in enumerate(skel):
        for j, (_, _, lfs2) in enumerate(skel):
            for l in lfs2:
                if dotself(pt1 - l) <= epsilon:
                    elist.append((i, j))
    T = nx.Graph(elist)
    for n in T.nodes:
        T.nodes[n]['points'] = skel[n][0]
        T.nodes[n]['interior'] = True
        T.nodes[n]['parent'] = R.nodes[rn]['cell'], R.nodes[rn]['cellpts']
    return T



    # for n in R.nodes:
    #     c = R.nodes[n]['cell']
    #     poly = []
    #     for e1, _ in nx.find_cycle(nx.Graph(H.subgraph(c))):
    #         poly.append(H.nodes[e1]['points'])
    #     poly = np.array(poly)
    #     if iscw(poly): poly = np.flip(poly, axis=0)
    #     # TODO: all of this is bad because polyskel. It doesn't seem
    #     # too hard to rewrite polyskel to store connectivity from the
    #     # get-go.
    #     print(iscw(poly))
    #     def skl_sortkey(skl): return skl[1]
    #     skels = polyskel.skeletonize(poly, holes=[])
    #     # good lord
    #     newskels = []
    #     for u, v, w in skels:
    #         u = np.array(list(u))
    #         neww = []
    #         for x in w:
    #             neww.append(np.array(list(x)))
    #         newskels.append((u, v, neww))
    #     skels = newskels
    #     epsilon = np.array([1e-5, 1e-5])
    #     elist = []
    #     # WHY??
    #     for i, (pt1, _, _) in enumerate(skels):
    #         for j, (_, _, neighbors2) in enumerate(skels):
    #             if i != j or j != i:
    #                 # WHY????
    #                 for n2 in neighbors2:
    #                     if np.all(np.abs(pt1 - n2) <= epsilon):
    #                         # IM DEAD
    #                         elist.append((i, j))
    #     if not elist:
    #         middle = rg_centroid(R, H, c)
    #         T = nx.Graph()
    #         T.add_node(1, points=middle)
    #     else:
    #         middle = list(max(skels, key=skl_sortkey)[0])
    #         T = nx.Graph(elist)
    #         for n_ in T.nodes:
    #             T.nodes[n_]['points'] = skels[n_][0]

    #     R.nodes[n]['centroid'] = np.array(middle)
    #     # graph of skeleton. we will traverse this when 
    #     # passing directly through a cell without scanning
    #     R.nodes[n]['tgraph'] = T

def get_points_array(H:nx.Graph, nlist:list = None, posattr:str='points'):
    if nlist == None:
        nlist = list(H.nodes())
    points = np.empty((len(nlist), 2))
    for i, n in enumerate(nlist):
        points[i] = H.nodes[n]['points']
    return points
    
def get_midpoint_shared(H: nx.DiGraph, R: nx.Graph, e1: int, e2: int) -> np.ndarray:
    n1, n2 = R[e1][e2]['shared']
    p1, p2 = H.nodes[n1]['points'], H.nodes[n2]['points']
    v = (p2-p1)/2
    return p1 + v

def dotself(x): return np.dot(x, x)


def remap_nodes_unique(new_node: int, T: nx.Graph):
    mapping = {}
    for n in T.nodes:
        mapping[n] = -new_node
        new_node += 1
    nx.relabel_nodes(T, mapping, copy=False)
    return new_node

def make_skelgraph(H: nx.DiGraph, R: nx.Graph):
    S = nx.Graph()
    # TODO: sooo, this is only necessary because we didn't index by 
    # point directly, but instead by index. WHOOPS! We'd have to
    # rewrite probably the entire reeb graph fuction if we wanted to
    # do without this proxy node list...
    new_node = 0
    eulerian = nx.eulerian_circuit(nx.eulerize(R))
    for n in R.nodes:
        new_node = remap_nodes_unique(new_node, R.nodes[n]['tgraph'])
        nx.set_edge_attributes(R.nodes[n]['tgraph'], True, 'original')
        nx.set_node_attributes(R.nodes[n]['tgraph'], True, 'original')

    for e1, e2 in eulerian:
        T_this: nx.Graph = R.nodes[e1]['tgraph']
        T_other: nx.Graph = R.nodes[e2]['tgraph']
        for n in T_this.nodes:
            T_this.nodes[n]['cell'] = R.nodes[e1]['cell']
            T_this.nodes[n]['middle'] = False
            if T_this.degree(n) > 1:
                T_this.nodes[n]['end'] = False
            else:
                T_this.nodes[n]['end'] = True
        for n in T_other.nodes:
            T_other.nodes[n]['cell'] = R.nodes[e2]['cell']
            T_other.nodes[n]['middle'] = False
            if T_other.degree(n) > 1:
                T_other.nodes[n]['end'] = False
            else:
                T_other.nodes[n]['end'] = True
        # get midpoint and add the node to both graphs
        midp = get_midpoint_shared(H, R, e1, e2)
        # find closest in both skelgraphs to this new node
        close_this = min(
            [n for n in T_this.nodes if T_this.nodes[n]['end']], 
            key=lambda n: dotself(T_this.nodes[n]['points'] - midp))
        close_other = min(
            [n for n in T_other.nodes if T_other.nodes[n]['end']], 
            key=lambda n: dotself(T_other.nodes[n]['points'] - midp))
        # add midpoint to both graphs after finding closest
        midp_node = -new_node
        T_this.add_node(
            midp_node, points=midp, cell=None, 
            original=False, middle=True,
            entry_cell=R.nodes[e1]['cell'])
        T_other.add_node(midp_node, points=midp, cell=None, original=False, middle=True)
        new_node += 1
        # add edge to both subgraphs
        T_this.add_edge(close_this, midp_node, original=False)
        T_other.add_edge(midp_node, close_other, original=False)
        S = nx.compose_all( (S, T_this, T_other) )

    for e1,e2 in S.edges:
        S[e1][e2]['dist'] = np.sqrt(dotself(S.nodes[e2]['points'] - S.nodes[e1]['points']))
    return S

def cul_de_sac_check(S:nx.Graph, n): 
    deg = S.degree(n) == 1
    orig = S.nodes[n]['original'] == 1
    cell = S.nodes[n]['cell']
    if cell is not None:
        cellch = len(cell) > 3
    else:
        cellch = False
    return deg and orig and cellch


def rcross(H: nx.DiGraph, u:int, v:int, w:int) -> float:
    '''compute the vector cross product of edges `u`,`v` and `v`,`w` on `H`.
    This one is more expensive than qcross but returns a numerical value.

    Parameters
    ----------
    H : nx.DiGraph
        Graph
    u : int
        idx of node
    v : int
        idx of node
    w : int
        idx of node

    Returns
    -------
    float
        cross product
    '''
    a = H.nodes[v]['points'] - H.nodes[u]['points']
    b = H.nodes[w]['points'] - H.nodes[v]['points']
    a /= np.linalg.norm(a)
    b /= np.linalg.norm(b)
    return a[0] * b[1] - a[1] * b[0]

def append_cell(H: nx.DiGraph, v: int, cells:dict) -> list:
    '''[summary]

    Parameters
    ----------
    H : nx.DiGraph
        graph on which to append cells
    v : int
        vertex from which to start
    cells : list
        list of cells.

    Returns
    -------
    list
        new list of cells
    '''
    prev_node = v
    for path_start in H.adj[v]:
        path_end = False
        path = []
        node = path_start
        while not path_end:
            cvals = []
            for candidate in H.adj[node]:
                if candidate != prev_node:
                    cval = rcross(H, prev_node, node, candidate)
                    cvals.append( (candidate, cval) )
            def cvalsort(c): return c[1]
            cvals.sort(key=cvalsort)
            # get most clockwise cval
            best = cvals[0][0]
            prev_node = node
            node = best
            path.append(best)
            if best == path_start:
                path_end = True
        to_add = frozenset(path)
        if to_add not in cells.keys():
            cells[to_add] = path
    return cells

def splitmerge_points(H: nx.DiGraph, v: int):
    a, b = get_intersects(H, v)
    if a is not None:
        ai = max(H.nodes()) + 1
        assert ai not in set(H.nodes)
        H.add_node(ai, points=a['pt'])
        # add new edge to H
        H.add_edge(ai, v, weight=3) # close
        H.add_edge(v, ai, weight=4) # open
        # split the edge with which there was a collision in half
        H.add_edge(a['e1'], ai, weight=a['weight'])
        H.add_edge(ai, a['e2'], weight=a['weight'])
        H.remove_edge(a['e1'], a['e2'])
    if b is not None:
        bi = max(H.nodes()) + 1
        assert bi not in set(H.nodes)
        H.add_node(bi, points=b['pt'])
        # add new edge to H
        H.add_edge(v, bi, weight=3) # open
        H.add_edge(bi, v, weight=4) # close
        # split the edge with which there was a collision in half
        H.add_edge(b['e1'], bi,  weight=b['weight'])
        H.add_edge(bi, b['e2'],  weight=b['weight'])
        H.remove_edge(b['e1'], b['e2'])

def get_intersects(H: nx.DiGraph, v: int) -> tuple:
    '''Check for intersects a vertical line passing through v on the graph H.
    Think of it like this: We basically draw a line upwards from v until we 
    reach a line on H; then we register a collision, which contains the origin 
    vertex of the collision in 'vx', the point of the collision in 'pt', the edge
    with which the line collided in 'edge', and the weight of that edge in 'weight'.

    This function returns a maximum of two collisions, one above and one below, 
    which correspond to the closest edge with which the line starting at `v` 
    collided. If there is no such edge, it returns `None`, otherwise returns a
    dict with information about the collision specified above.

    Parameters
    ----------
    H : nx.DiGraph
        graph
    v : int
        event idx

    Returns
    -------
    tuple
        collisions above or/and below v on edges of H
    '''
    # list of all collisions with vertex v on the vertical
    collisions = []
    # check each edge
    for edge in H.edges:
        # check whether there could be a collision
        if check_edge(H, v, edge):
            e1, e2 = edge
            p1, p2, pv = H.nodes[e1]['points'], H.nodes[e2]['points'], H.nodes[v]['points']
            ipt = intersect_vertical_line(p1, p2, pv)
            # store attrs of collision
            collisions.append({
                'vx' : v,
                'pt' : ipt,
                'e1' : e1,
                'e2' : e2,
                'weight' : H[e1][e2]['weight']
            })
    above, below = None, None
    y = H.nodes[v]['points'][1]
    # the difference between the collision point and the vertex point
    def ydiff(c): return c['pt'][1] - y
    if collisions:
        # the minimum collision point above `v`
        above = min([c for c in collisions if c['pt'][1] > y], key=ydiff, default=None)
        # the maximum collision point below `v`
        below = max([c for c in collisions if c['pt'][1] < y], key=ydiff, default=None)
    return above, below

def intersect_vertical_line(p1: np.ndarray, p2: np.ndarray, pv: np.ndarray) -> np.ndarray:
    m = abs(pv[0] - p1[0]) / abs(p2[0] - p1[0])
    # y = mx + b
    py = m * (p2[1] - p1[1]) + p1[1]
    return np.array([pv[0], py])

def check_edge(H: nx.DiGraph, v: int, edge: tuple) -> bool:
    '''Check whether an edge [edge=(e1, e2)] on H intersects with a straight vertical line
    at `v`

    Parameters
    ----------
    H : nx.DiGraph
        graph
    v : int
        vertex idx
    edge : tuple of (int, int)
        edge idxs

    Returns
    -------
    bool
        Whether the x coordinates of the point `v` are beteen the edge x-coords
    '''
    e1, e2 = edge
    p1, p2, pv = H.nodes[e1]['points'], H.nodes[e2]['points'], H.nodes[v]['points']
    if p1[0] > pv[0] and p2[0] < pv[0]: return True
    elif p1[0] < pv[0] and p2[0] > pv[0]: return True
    else: return False

def check_lu(H: nx.DiGraph, v:int) -> int:
    l, u, above = lower_upper(H, v)
    lpoint = H.nodes[l]['points'] # lower point
    upoint = H.nodes[u]['points'] # upper point
    vpoint = H.nodes[v]['points'] # vertex point

    event = None
    # both points on the right of v
    if lpoint[0] > vpoint[0] and upoint[0] > vpoint[0]:
        if above: event = Event.OPEN
        else: event = Event.SPLIT
    # both points on the left of v
    elif lpoint[0] < vpoint[0] and upoint[0] < vpoint[0]:
        if above: event = Event.CLOSE
        else: event = Event.MERGE
    # lower right, upper left
    elif lpoint[0] > vpoint[0] and upoint[0] < vpoint[0]:
        event = Event.INFLECTION
    # lower left, upper right
    elif lpoint[0] < vpoint[0] and upoint[0] > vpoint[0]:
        event = Event.INFLECTION

    if event is None:
        raise(ValueError('Event was not categorized correctly!'))
    return event
    
def lower_upper(H: nx.DiGraph, v: int):
    '''Check the neighbors of `v` to determine which is the lower neighbor 
    and which is the upper neighbor. If the predecessor is the upper neighbor,
    also returns True 

    Parameters
    ----------
    H : nx.DiGraph
        must contain 'weight' as a key.
    v : int
        The vertex to check

    Returns
    -------
    tuple of (int, int, bool)
        lower vertex, upper vertex, above
    '''
    # filter only for neighboring nodes which were not constructed by a split,
    # ie node-->neighbor or neighbor-->node edge weight is 1 or 2.
    vA, vB = None, None
    for p in H.predecessors(v):
        w = H[p][v]['weight']
        vA = p
        if w == 1 or w == 2: 
            vA = p
    for s in H.successors(v):
        w = H[v][s]['weight']
        vB = s
        if w == 1 or w == 2:
            vB = s
    # cannot have trailing nodes
    assert(vA is not None)
    assert(vB is not None)
    # if vA is None or vB is None:
    #    raise(ValueError('No valid predecessors or successors to node {}'.format(v)))
    # compute cross product. if qcross is true (i.e. positive cross product), then 
    # the vertex vA is "above" the vertex vB. ["above" here means "above" in a 
    # topological sense; it's not necessarily above in the cartesian sense.]
    # 
    # Note that when "above" is true, vB and vA are flipped (i.e. the successor
    # is stored into lower, and the predecessor is stored into upper.
    above = qcross(H, vA, v, vB)
    if above: 
        lower, upper = vB, vA
    else: 
        lower, upper = vA, vB
    return lower, upper, above

def qcross(H: nx.DiGraph, vA: int, v: int, vB: int) -> bool:
    '''compute a quick cross product over nodes `vA`, `v`, `vB` on `H`.'''
    # find vectors with tails at `v` and pointing to vA, vB
    p1 = H.nodes[vA]['points']
    p2 = H.nodes[vB]['points']
    pv = H.nodes[v]['points']

    a = pv - p1
    b = pv - p2
    # this is roughly a measure of the topological orientation of vA, vB
    if a[0] * b[1] - b[0] * a[1] >= 0: return True
    else: return False

def rotate_graph(G: nx.DiGraph, theta: float, posattr: str = 'points') -> nx.DiGraph:
    '''Rotate a graph `G`. This makes a copy of `G` and returns it, leaving `G` untouched.

    Parameters
    ----------
    G : nx.DiGraph
        input Graph
    theta : float
        rotation angle, radians
    posattr : str, optional
        which key the points are stored under, by default 'points'

    Returns
    -------
    nx.DiGraph
        the new rotated graph
    '''
    H = copy.deepcopy(G)
    rotmatr = make_rot_matrix(theta)
    for node in G.nodes:
        original = G.nodes[node]['points']
        rotated = original @ rotmatr
        H.nodes[node]['points'] = rotated
    return H

def make_rot_matrix(theta: float) -> np.ndarray:
    '''Create a rotation matrix

    Parameters
    ----------
    theta : float
        Angle

    Returns
    -------
    np.ndarray
        2x2 Rotation Matrix created from Angle.
    '''
    rotmatr = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return rotmatr

