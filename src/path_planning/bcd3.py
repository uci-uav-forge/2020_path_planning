import numpy as np
from enum import Enum
from gen_polygon import ConvPolygon
import matplotlib.pyplot as plt
import networkx as nx
import math, itertools

class Event(Enum):
    CLOSE=1
    OPEN=2
    SPLIT=3
    MERGE=4
    INFLECTION=5
    INTERSECT=6

def intersect_line(x, a, b):
    '''Check if a vertical line at x=`x` intersects
    the line segment made by points `a`, `b`.

    Parameters
    ----------
    x : float
        x coord of vertical
    a : [float, float]
        point in 2d space
    b : [float, float]
        point in 2d space

    Returns
    -------
    `[x, y]` or `None`
        Point of intersection if intersection; none if not.
    '''
    is_intersect = False

    if a[0] < x and x < b[0]:
        is_intersect = True
        p1 = a
        p2 = b
    elif a[0] > x and x > b[0]:
        is_intersect = True
        p1 = b
        p2 = a

    if is_intersect:
        px = x
        m = abs(x - p1[0]) / abs(p2[0] - p1[0])
        py = p1[1] + m * (p2[1] - p1[1])

        return [px, py]
    else:
        return None

def check_edge(x, edge, points):
    '''Check if the vertical line at x=`x` could concievably
    interact with the edge `edge`

    Parameters
    ----------
    x : float
        x location
    edge : tuple of (int, int, ...)
        indices describing the edge
    points : Mx2 array of float
        Location of the points

    Returns
    -------
    Bool
        `True` if edge could result in intersection, `False` if not.
    '''
    if points[edge[0]][0] > x and x > points[edge[1]][0]:
        return True
    elif points[edge[1]][0] > x and x > points[edge[0]][0]:
        return True
    else:
        return False

def get_intersects(event, G, points):
    '''Get the first upper and lower intersects at vertex `event`,
    if they exist. 

    Parameters
    ----------
    event : int
        The index (in points) of the vertex we are looking at
    G : nx.DiGraph
        The polygon graph
    points : Mx2 array of float
        The list of points

    Returns
    -------
    tuple of 2x1 array of float or None
        The `above, below` points of intersection. Either `above` or `below` can be
        None if no intersection is found there.
    '''
    x, y = points[event][0], points[event][1]
    collisions = []
    # get all intersects
    for edge in G.edges():
        if check_edge(x, edge, points):
            # get the point of intersection
            ipt = intersect_line(x, points[edge[0]], points[edge[1]])
            # store its x, y, edge, edgew
            collision = {
                'vx' : event, # the vertex associated with the collision
                'pt' : ipt, # the point of the collision
                'edge' : edge, # the edge with which the line collided
                'edgew' : G[edge[0]][edge[1]]['weight'] # the weight of that edge
            }
            collisions.append(collision)
    above, below = None, None
    if collisions:
        above = min([c for c in collisions if c['pt'][1] > y], key=lambda x: abs(x['pt'][1]-y), default=None)
        below = min([c for c in collisions if c['pt'][1] < y], key=lambda x: abs(x['pt'][1]-y), default=None)
    return above, below

def qcross(points, u, v, w):
    '''Quick cross product of 3 points in space u, v, w

    Parameters
    ----------
    points : Mx2 array of float
        The list of points in 2D space
    u : int
        first point
    v : int
        second point
    w : int
        third point

    Returns
    -------
    Bool
        `True` if angle is positive, `False` if angle is negative.
    '''

    a = points[v] - points[u]
    b = points[v] - points[w]

    if a[0] * b[1] - b[0] * a[1] >= 0:
        return True
    else:
        return False

def lower_upper(points, event, G):
    '''For an `event` (node) on DiGraph `G`, re-order the edges so that
    the lower one is `lower` and the upper one is `upper`. Whether the
    incoming edge is above the outgoing edge is stored in the var `above`.

        U          i.e., lets take the edges U, V, W in the diagram at left.
        ^          Lets assume the edge V,U _exits_ and W,V _enters_. Then, 
         \         (V,U) is the "upper" edge, since it's above; and (W,V) is 
          \        the "lower" edge. So (V, W) , (V, U) is returned, and
           \       'above' is set to `False`, because the "upper" edge exits
            V      and the "lower" edge enters.
           /      
          /             
         ^             
        W              
           
     
    Parameters
    ----------
    points : Mx2 array of float
        points in 2d space
    event : int
        index to the event
    G : nx.DiGraph
        the DiGraph that encodes information about
        points surrounding `event`.

    Returns
    -------
    tuple of (int, int, bool)
        index to the lower neighbor, index of upper neighbor,
        and whether the lower and upper entered, exited or exited,
        entered.
    '''
    # we only consider the non-added edges
    # weight 3 means that edges were added by
    # prior split/merge sequences...
    for p in G.predecessors(event):
        if G[p][event]['weight'] != 3:
            vA = p
    for s in G.successors(event):
        if G[event][s]['weight'] != 3:
            vB = s

    above = False
    if qcross(points, vA, event, vB):
        above = True
        lower, upper = vB, vA
    else:        
        lower, upper = vA, vB
    return lower, upper, above

def check_lu(points, event, G):
    lower, upper, above = lower_upper(points, event, G)
    # both right
    if points[lower][0] > points[event][0] and points[upper][0] > points[event][0]:
        # entering above
        if above:
            return Event.OPEN
        # entering below
        else:
            return Event.SPLIT
    # both left
    elif points[lower][0] < points[event][0] and points[upper][0] < points[event][0]:
        # entering above
        if above:
            return Event.CLOSE
        # entering below
        else:
            return Event.MERGE
    # lower right, upper left
    elif points[lower][0] > points[event][0] and points[upper][0] < points[event][0]:
        return Event.INFLECTION
    # lower left, upper right
    elif points[lower][0] < points[event][0] and points[upper][0] > points[event][0]:
        return Event.INFLECTION

def node_classify(v, A, crits, vtypes, points, G):
    etype = check_lu(points, v, G)
    add = (None, None)
    if etype == Event.SPLIT or etype == Event.MERGE:
        add, G, points = make_splitmerge_points(v, etype, G, points)
        A[v] = add
        print('\tv: {}, type: {}, new points: {}'.format(v, etype, add))
    if etype in [Event.OPEN, Event.SPLIT, Event.MERGE, Event.CLOSE]:
        crits.append((v, etype))
    vtypes[v] = etype
    for a in add:
        if a is not None:
            vtypes[a] = Event.INTERSECT
    return A, crits, vtypes, points, G

def get_addpt_neighbors(a, G, want3=False):
    '''Get the neighbors of a point, `a`,
    which are /are not weight 3 neighbors. 
    i.e. they were/were not added by a 
    split/merge operation

    Parameters
    ----------
    a : int
        the node which may have weight 3
        edges attached
    G : nx.DiGraph
        the graph that encodes all of this
    want3: bool
        if we want the weight3 edges or not

    Returns
    -------
    set
        set containing neighbors of `a`
    '''
    neigh = set()
    if not want3:
        for p in G.predecessors(a):
            if G[p][a]['weight'] != 3:
                neigh.add(p)
        for s in G.successors(a):
            if G[a][s]['weight'] != 3:
                neigh.add(s)
    else:
        for p in G.predecessors(a):
            if G[p][a]['weight'] == 3:
                neigh.add(p)
        for s in G.successors(a):
            if G[a][s]['weight'] == 3:
                neigh.add(s)
    return neigh

def make_splitmerge_points(event, event_type, G, points):
    '''returns add list, updated G, updated points'''
    a, b = get_intersects(event, G, points)
    a_i, b_i = None, None
    if a:
        # add point to points list
        points = np.concatenate([points, np.array([a['pt']])])
        # index is the last member of new points array
        a_i = points.shape[0] - 1
        # add the new edge to G
        if event_type == Event.SPLIT:
            G.add_edge(a_i, event, weight=3) # close
            G.add_edge(event, a_i, weight=4) # open
        elif event_type == Event.MERGE:
            G.add_edge(a_i, event, weight=3) # open
            G.add_edge(event, a_i, weight=4) # close
        G.add_edge(a['edge'][0], a_i, weight=a['edgew'])
        G.add_edge(a_i, a['edge'][1], weight=a['edgew'])
        G.remove_edge(a['edge'][0], a['edge'][1])
    if b:
        points = np.concatenate([points, np.array([b['pt']])])
        b_i = points.shape[0] - 1
        if event_type == Event.SPLIT:
            G.add_edge(event, b_i, weight=3) # open
            G.add_edge(b_i, event, weight=4) # close
        elif event_type == Event.MERGE:
            G.add_edge(event, b_i, weight=3) # open
            G.add_edge(b_i, event, weight=4) # close
        G.add_edge(b['edge'][0], b_i, weight=b['edgew'])
        G.add_edge(b_i, b['edge'][1], weight=b['edgew'])
        G.remove_edge(b['edge'][0], b['edge'][1])
    return (a_i, b_i), G, points

def neigh(v, G):
    neigh = set()
    if v:
        neigh |= set(G.succ[v])
        neigh |= set(G.pred[v])
    return neigh

def line_sweep(poly, ax):
    # List of events (vertices/nodes)
    L = sorted(poly.G.nodes, key=lambda t: poly.points[t][0])
    # List of closed cells
    C = []
    # Additional points found in splits & merges
    A = {}
    crits, vtypes = [], {}
    print('Preliminary Split/Merge Check...')
    for v in L:
        A, crits, vtypes, poly.points, poly.G = node_classify(v, A, crits, vtypes, poly.points, poly.G)
    print('Done!')
    
    for v in crits:
        C = process_events(v, vtypes, C, poly.points, poly.G)
    '''
    for ci, crit in enumerate(crits):
        lookback = 1
        for critj in reversed(crits[:ci]):
            if critj[1] == Event.SPLIT or critj[1] == Event.MERGE:
                break
            lookback += 1
        C = __process_events(crits[ci - lookback], crits[ci], C, poly.points, poly.G)
    '''
    return C

def right_turn(points, u, v, w):
    a = points[v] - points[u]
    b = points[w] - points[v]
    a = a/np.linalg.norm(a)
    b = b/np.linalg.norm(b)
    return a[0] * b[1] - b[0] * a[1]

def process_events(v, vtypes, C, points, G):
    print('---')
    print(v)
    prev_node = v[0] # inititate prior for c product
    for path_start in G.adj[v[0]]:
        print('\tPath Start: {}'.format(path_start))
        path_end = False
        path = []
        # start the path
        node = path_start
        n = 0
        while path_end == False:
            cvals = []
            for possible_node in G.adj[node]:
                if possible_node != prev_node:
                    # calculate cross product and append
                    cval = right_turn(points, prev_node, node, possible_node)
                    cvals.append( (possible_node, cval) )
            
            cvals = sorted([cval for cval in cvals], key=lambda t: t[1])
            # choose node with most CW pointing cval
            best = cvals[0][0]
            print('\t\tCvals: {} --> Best: {}'.format(cvals, best))
            # replace previous with current
            prev_node = node
            # replace node with the one we chose
            node = best
            # append to list
            path.append(best)
            # if we're back at the original node, then we know we have formed a loop
            # and therefore have formed a cell!
            if best == path_start:
                print('\t\tPath Closed --> Loop Created')
                path_end = True
            elif n >= 1e5:
                raise(Exception('Path Not Closed --> Exceeded Max Path Length!'))
            n += 1
        if set(path) not in C:
            C.append(set(path))
    return C
    


def __process_events(v_l, v_r, C, points, G):
    leftv, rightv = v_l[0], v_r[0]
    epsilon = 1e-5
    wfunc = lambda n: points[n][0] >= points[leftv][0] - epsilon and points[n][0] <= points[rightv][0] + epsilon
    window = [n for n in G.nodes if wfunc(n)]
    H = nx.subgraph(G, window)
    scycs = nx.simple_cycles(H)
    open_cells = []
    for cyc in scycs:
        if len(cyc) > 2:
            wts = []
            for _,_,d in nx.subgraph(H, cyc).edges(data=True):
                wts.append(d['weight'])
            # if all the same wt, this isn't really a cell for some reason
            if wts.count(wts[0]) != len(wts):
                open_cells.append(cyc)
    
    if len(open_cells) == 3:
        # find centroid y coordinate of points
        ycentroid = lambda cyc: sum([points[n][1] for n in cyc])/len(cyc)
        # sort the list by centroid (include indices in sort)
        l_m_u = sorted(enumerate([c for c in open_cells]), key=lambda t: ycentroid(t[1]))
        # first index-->lower cell, last index-->upper cell
        lo, up = l_m_u[0][0], l_m_u[-1][0]
        # extend C with lower, upper 
        C.append(open_cells[lo]) 
        C.append(open_cells[up])
    else:
        for c in open_cells:
            C.append(c)
    return C
    

if __name__ == '__main__':
    seed = np.random.randint(0,1e5)
    np.random.seed(60431)
    # 69246
    print(seed)

    colormap = plt.get_cmap('viridis')
    poly = ConvPolygon(points=(3, 30, 10, 40), jaggedness=10, holes=5)
    fig0, fig1, fig2 = plt.figure('Input Shape'), plt.figure('Cell Decomposition'), plt.figure('Cells')
    ax0 = fig0.add_subplot()
    
    poly.chart(ax0, poly.G, cm=colormap)
    ax1 = fig1.add_subplot()
    bcd = line_sweep(poly, ax1)
    poly.chart(ax1, poly.G, cm=colormap)

    nrows = 3
    row, col = 3, math.ceil(len(bcd)/3)
    ax2 = fig2.subplots(nrows=row, ncols=col)
    for i in range(row):
        for j in range(col):
            ci = i*col + j
            if ci < len(bcd):
                H = nx.subgraph(poly.G, bcd[ci])
                poly.chart(ax2[i,j], H)
    
    plt.show()
    print(seed)