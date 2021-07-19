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


def discretize(J: nx.DiGraph, R:nx.Graph, gridsz:float, theta=None) -> nx.Graph:
    '''Discretize a graph `J`, with grid size gridsz, and grid angle theta.

    Parameters
    ----------
    J : nx.DiGraph
        The bdc-composed world graph`
    R : nx.Graph
        The reeb graph
    gridsz : float
        The size of the resulting discretized grid. Must be >0.
    theta : float, optional
        Angle for the discretization, by default None

    Returns
    -------
    nx.Graph
        The discretized grid. Each node of the graph is joined by max 4 neighbors:
        2 x neighbors and 2 y neighbors (if theta is supplied, these xy axes correspond
        to an angled frame of reference)
    '''
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
    '''Perform a boustrophedon line sweep over a world graph G.

    Parameters
    ----------
    G : nx.DiGraph
        World graph. Contains an outer boundary, which is made up of a clockwise ordering of edges
        with weight=1, and optionally contains holes, which are counterclockwise orderings of edges 
        with weight=2. Worlds must be non-degenerate planar graphs!
    theta : float, optional
        Angle of the line sweep from x-axis, by default 0
    posattr : str, optional
        attribute of points in `G`, by default 'points'

    Returns
    -------
    tuple
        H, which is a graph of the new bdc-composed world rotated to `theta`
        J, which is the graph of the new bdc-compoased world rotated back to its original orientation
        R, which is the reeb graph containing connectvitity information between each cell
        S, which is the skel graph containing connected straight skeletons of each cell in `R`.
    '''
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

def rg_centroid(H: nx.DiGraph, cell: set) -> np.ndarray:
    '''Get centroid of a cell `cell` made up of nodes in `H`

    Parameters
    ----------
    H : nx.DiGraph
        The world graph
    cell : set
        An ordered or unordered list of nodes in `H`

    Returns
    -------
    np.ndarray
        the centroid as an xy point.
    '''
    p = np.zeros((2,), dtype=np.float64)
    for c in cell:
        p += H.nodes[c]['points']
    p /= len(cell)
    return p

def build_reebgraph(H: nx.DiGraph, cells: list) -> nx.Graph:
    '''Build the reebgraph on `H` using the cell information contained in 'cells'

    Parameters
    ----------
    H : nx.DiGraph
        The bdc composed world
    cells : list
        each entry in cells is a list of nodes in `H` which form a closed cell in the bdc composition

    Returns
    -------
    nx.Graph
        Graph, `R` which represents connectivity of each cell in `H`
    '''
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

def create_skelgraph(R: nx.Graph, H: nx.DiGraph) -> nx.Graph:
    '''Create a "skelgraph" from a bdc world `H` and its reeb graph, `R`.
    A skelgraph is a graph of the straight skeletons of each cell of a world `H`,
    joined by the midpoints of each cell wall on `H`. 

    Traversing the skelgraph of `H` means visiting each cell the boustrophedon
    decomposition of `H`.

    Parameters
    ----------
    R : nx.Graph
        The reeb graph of the world
    H : nx.DiGraph
        The BDC of the world

    Returns
    -------
    nx.Graph
        An undirected graph joining the straight skeleton of each cell.
    '''
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
    
def get_closest_on_skel(R: nx.graph, rnode: int, midp: np.array) -> np.array:
    '''Get the closest point of a skel graph stored in the `rnode` of `R` to the midpoint `midp`.

    Parameters
    ----------
    R : nx.graph
        Reeb graph containing straight skeleton in key 'skel_graph'
    rnode : int
        node on that reeb graph
    midp : np.array
        the point to test

    Returns
    -------
    int
        a node on skel_graph which is closest to `midp`
    '''
    this_connectors = []
    for n in R.nodes[rnode]['skel_graph'].nodes:
        if R.nodes[rnode]['skel_graph'].nodes[n]['interior']:
            thispt = R.nodes[rnode]['skel_graph'].nodes[n]['points']
            dist = dotself(midp - thispt)
            this_connectors.append((n, dist))
    # return closest
    return sorted(this_connectors, key=lambda t: t[1])[0][0]


def fix_polyskel(skel: list):
    '''this is a helper function for casting the results of `polyskel` to
    numpy arrays, rather than Euler3 points

    Parameters
    ----------
    skel : list
        results of `polyskel` function call

    Returns
    -------
    list
        each element of this list contains origin, height, and leafs. 
        see polyskel documentation for more
    '''
    newskel = []
    def strip(leaf): return np.array(leaf)
    for origin, height, leafs in skel:
        origin = np.array(origin)
        leafs = list(map(strip, leafs))
        newskel.append((origin, height, leafs))
    return newskel

def traverse_polyskel(R: nx.Graph, rn: int, skel: list) -> nx.Graph:
    '''This is a helper function for casting the list returned by `polyskel` to 
    a networkx Graph.

    This function also adds the list from `skel` to a reebgraph stored in `R` at node `rn`.

    Parameters
    ----------
    R : nx.Graph
        Reeb Graph
    rn : int
        node on the reeb graph
    skel : list
        list returned by polyskel

    Returns
    -------
    nx.Graph
        the straight skel graph
    '''
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

def get_points_array(H:nx.Graph, nlist:list = None, posattr:str='points') -> np.ndarray:
    '''Get an array of points from `H` which stores points in `posattr` from indices in `nlist`

    Returns all points in H by default.

    Parameters
    ----------
    H : nx.Graph
        world graph
    nlist : list, optional
        subset of nodes on H to get points from, by default None
    posattr : str, optional
        attribute that the points are stored under, by default 'points'

    Returns
    -------
    np.ndarray
        Mx2 array for M points 
    '''
    if nlist == None:
        nlist = list(H.nodes())
    points = np.empty((len(nlist), 2))
    for i, n in enumerate(nlist):
        points[i] = H.nodes[n]['points']
    return points
    
def get_midpoint_shared(H: nx.DiGraph, R: nx.Graph, e1: int, e2: int) -> np.ndarray:
    '''Get the midpoint of the line which joins two cells, e1 and e2.

    E.g, returns the point x from two neighboring rectangular cells:

    ┌──────────┐
    │          │
    │    e1    │
    │          │
    └───┬──x───┴──┐
        │         │
        │    e2   │
        │         │
        └─────────┘

    Parameters
    ----------
    H : nx.DiGraph
        the world graph
    R : nx.Graph
        the reeb graph of the world `H`
    e1 : int
        first shared edge. Node of `R`
    e2 : int
        second shared edge. Node of `R`

    Returns
    -------
    np.ndarray
        the midpoint
    '''
    n1, n2 = R[e1][e2]['shared']
    p1, p2 = H.nodes[n1]['points'], H.nodes[n2]['points']
    v = (p2-p1)/2
    return p1 + v

def dotself(x:np.ndarray) -> float: 
    '''dot a vector x with itself to produce a scalar

    Parameters
    ----------
    x : np.ndarray
        the vector

    Returns
    -------
    float
        scalar value |x|^2
    '''
    return np.dot(x, x)


def remap_nodes_unique(new_node: int, T: nx.Graph) -> int:
    '''Alters T in place, replacing its non-unique nodes by iterating on `new_node`.
    we can run this function on several graphs, thus guaranteeing that their nodes
    do not collide.

    Parameters
    ----------
    new_node : int
        starting value for new nodes
    T : nx.Graph
        A graph with integer nodes

    Returns
    -------
    int
        ending value for new nodes
    '''
    mapping = {}
    for n in T.nodes:
        mapping[n] = -new_node
        new_node += 1
    nx.relabel_nodes(T, mapping, copy=False)
    return new_node

def make_skelgraph(H: nx.DiGraph, R: nx.Graph):
    '''Make the "straight skeleton graph" over the world `H` with 
    its reeb-graph `R`.

    Parameters
    ----------
    H : nx.DiGraph
        The world. Must already be BDC decomposed.
    R : nx.Graph
        Reeb graph of the world.

    Returns
    -------
    nx.Graph
        the "skeleton" graph. This undirected graph is made up of the straight skeleton
        of each cell, connected by the midpoints of the dividing lines between cells.
    '''
    S = nx.Graph()
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

def splitmerge_points(H: nx.DiGraph, v: int) -> None:
    '''Alters H in place to produce split/merge points
    from an event on node `v`

    Parameters
    ----------
    H : nx.DiGraph
        Shape graph
    v : int
        index of the event vertex
    '''
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
    '''Get line intersect on the line formed by [p1, p2] for the point at `pv`

    Parameters
    ----------
    p1 : np.ndarray
        end point of line
    p2 : np.ndarray
        end point of line
    pv : np.ndarray
        point of intersect

    Returns
    -------
    np.ndarray
        line from `pv` to the point of intersection
    '''
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

