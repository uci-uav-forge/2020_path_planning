import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
from datetime import datetime
from collections import OrderedDict, deque
from scipy import linalg
import networkx as nx
from collections import defaultdict
from enum import Enum

np.random.seed(3)

class Event(Enum):
    CLOSE=1
    OPEN=2
    SPLIT=3
    MERGE=4
    INFLECTION=5

class ConvPolygon(object):
    def __init__(self, points=(2, 40, 10, 40), jaggedness=2, holes=0):
        self.points = self.gen_cluster_points(*points)
        self.gen_poly(jaggedness=jaggedness, holes=holes)
        self.blist = self.make_boundaries()
        self.G = self.shape_graph()

        
    def gen_cluster_points(self, no_clusters, cluster_n, cluster_size, cluster_dist):
        '''Generate clusters of normally distributed points

        This generates multiple clusters of points, at a 

        Parameters
        ----------
        no_clusters : [type]
            [description]
        cluster_n : [type]
            [description]
        cluster_size : [type]
            [description]
        cluster_dist : [type]
            [description]

        Returns
        -------
        [type]
            [description]
        '''
        pts = np.zeros((no_clusters * cluster_n, 2))
        loc = np.array([0,0], dtype='float64')
        for c in range(no_clusters):
            pts[c * cluster_n:(c+1)* cluster_n, :] = np.random.normal(loc=loc, scale=cluster_size, size=(cluster_n, 2))
            loc += np.random.uniform(low=-cluster_dist, high=cluster_dist, size=np.shape(loc))
        return pts

    def shape_graph(self):
        '''make the graphs that define the polygon
        '''
        graphs = []
        for boundary, outer in self.blist:
            G = nx.Graph()
            elist, nlist = [], []
            for k, v in boundary.items():
                nlist.append(k)
                if outer:
                    elist.append( (v[0], k, 1) )
                else:
                    elist.append( (v[0], k, 2) )

            G.add_nodes_from(nlist)
            G.add_weighted_edges_from(elist)
            graphs.append(G)
        G = nx.compose_all(graphs)
        return G

    def bps_bls(self):
        '''numpy arrays of boundary points, boundary lines

        Returns
        -------
        tuple of list of np.ndarray
            (boundary points, boundary lines)
            boundary points is a list of np.ndarrays, sizes [B, 2] for points B in each boundary
            boundary lines is a list of np.ndarrays, sizes [B, 2, 2] for lines B in each boundary
        '''
        bps, bls = [], []
        for bound, _, _ in self.blist:
            pts = list(bound.keys())
            pts.append(next(iter(bound.keys())))
            lines = []
            for k, v in bound.items():
                lines.append([v[0], k])
            bps.append(np.array(pts))
            bls.append(np.array(lines))
        return bps, bls
            
    def count_tris(self):
        '''Count the triangles in the delaunay triangulation

        Returns
        -------
        Tuple of int
            (0 edge, single edge, double edge)
        '''
        edge0 = 0
        edge1 = 1
        edge2 = 2
        for s in self.dt.neighbors:
            c = (s == -1).sum()
            if c == 0:
                edge0 += 1
            elif c == 1:
                edge1 += 1
            elif c == 2:
                edge2 += 1
        return (edge0, edge1, edge2)
    
    def count_outer_edges(self):
        e0, e1, e2 = self.count_tris()
        return e1 + 2*e2

    def id_bps(self):
        '''Identify points within 2 of a boundary.

        Returns
        -------
        tuple of set
            (safe (interior) points, unsafe (exterior) points)
        '''
        unsafe = set()
        for neigh, tri in zip(self.dt.neighbors, self.dt.simplices):
            if -1 in neigh:
                # all points in this tri
                unsafe.update(set([x for x in tri]))
                for neigh2 in [x for x in neigh if x != -1]:
                    # all points in neighboring tris also
                    unsafe.update(set(self.dt.simplices[neigh2].flatten()))
        return set(list(range(len(self.dt.points)))) - unsafe, unsafe
    
    def id_tris(self):
        unsafe = set()
        for i, neigh in enumerate(self.dt.neighbors):
            if (neigh == -1).sum() > 0:
                unsafe.add(i)
        return set(list(range(len(self.dt.simplices)))) - unsafe, unsafe

    def id_tris2(self):
        '''Identify tris that have points within a single point of an edge.

        Returns
        -------
        tuple of set
            ('safe' tris, 'unsafe' tris)
            Unsafe are within a single point, safe are not within a single point of an edge.
        '''
        unsafe_tris, safe_tris = set(), set()
        safe_pts, unsafe_pts = self.id_bps()
        for i, tri in enumerate(self.dt.simplices):
            if set(tri).issubset(safe_pts):
                safe_tris.add(i)
            else:
                unsafe_tris.add(i)
        return safe_tris, unsafe_tris

    def centroid(self, pts):
        k = np.shape(pts)[0]
        return np.array(np.sum(pts, axis=0)/k, dtype='float64')

    def plot_tri(self, ax):
        '''Plot self.dt on mpl axes `ax`

        Parameters
        ----------
        ax : matplotlib `ax` object
            The axis on which to plot
        '''
        centers = np.sum(self.dt.points[self.dt.simplices], axis=1, dtype='int')/3.0
        centr = self.centroid(centers)
        colors = np.array([ (x - centr[0])**2 + (y - centr[1])**2 for x, y in centers])
        ax.tripcolor(self.dt.points[:,0], self.dt.points[:,1], self.dt.simplices, facecolors=colors, cmap='YlGn', edgecolors='darkgrey')
        
        
        ax.set_aspect('equal')
        ax.set_facecolor('lightblue')
    
    def first(self, s):
        '''
        get first item of collection
        '''
        return next(iter(s))

    def chart(self, ax, pltcolors=('lightblue', 'darkgreen', 'lightblue'), legend=False):
        '''
        Draw a chart of the figure on the Axes `ax`

        Returns
        -------
        `matplotlib.pyplot.Axes`
            Axis object containing a plot of the polygon.
        '''
        centers = np.sum(self.dt.points[self.dt.simplices], axis=1, dtype='int')/3.0
        centr = self.centroid(centers)
        colors = np.array([ (x - centr[0])**2 + (y - centr[1])**2 for x, y in centers])
        # draw colored tris
        # ax.tripcolor(self.points[:,0], self.points[:,1], self.dt.simplices, facecolors=colors, cmap='YlGn', edgecolors='darkgrey')

        # plt.triplot(self.dt.points[:,0], self.dt.points[:,1], self.dt.simplices, color='silver')
        # plt.fill(self.dt.points[:,0], self.dt.points[:,1])

        # points
        safe_pt, unsafe_pt = self.id_bps()
        safe_pt, unsafe_pt = list(safe_pt), list(unsafe_pt)
        # ax.plot(self.dt.points[safe_pt][:,0], self.dt.points[safe_pt][:,1], 'b.')
        # ax.plot(self.dt.points[unsafe_pt][:,0], self.dt.points[unsafe_pt][:,1], 'm.')

        # draw boundary
        for boundary, is_outer in self.blist:
            pts = []
            for k, v in boundary.items():
                pts.append(k)
            pts.append(self.first(boundary))
            if is_outer:
                ax.fill(self.dt.points[pts,0], self.dt.points[pts,1], pltcolors[1], zorder=1)
            else:
                ax.fill(self.dt.points[pts,0], self.dt.points[pts,1], pltcolors[2], zorder=2)

        # preserve aspect
        ax.set_aspect('equal')
        ax.set_facecolor(pltcolors[0])
        if legend:
            ax.legend()

    def aspect(self, tri):
        '''Given idx to a tri, find its aspect ratio

        Parameters
        ----------
        tri : int
            idx of tri

        Returns
        -------
        float 
            aspect ratio
        '''
        pts = self.dt.points[self.dt.simplices[tri]]
        v1 = pts[0,:] - pts[2,:]
        v2 = pts[1,:] - pts[0,:]
        v3 = pts[2,:] - pts[1,:]
        a = np.sqrt(v1.dot(v1.T))
        b = np.sqrt(v2.dot(v2.T))
        c = np.sqrt(v3.dot(v3.T))
        return a * b * c / ( (b + c - a) * (c + a - b) * (a + b - c) )

    def order_boundaries(self):
        is_outer = np.asarray(self.dt.neighbors == -1, dtype='bool')
        tris = [list(x.compressed()) for x in np.ma.MaskedArray(self.dt.simplices, is_outer)]
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
        minpt = min(b, key=lambda i: self.dt.points[i, 0])
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

    def make_boundaries(self):
        boundaries = []
        # find point with min x val
        min_p = np.argmax(self.dt.points, axis=0)
        for boundary, is_outer in self.order_boundaries():
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

    def update_plot(self, ax):
        ax.clear()
        self.plot_tri(ax)
        plt.pause(.1)

    def check_edgesafety(self, rm_candidate, edge_tris):
        critpts = self.dt.simplices[rm_candidate, (self.dt.neighbors[rm_candidate,:] == -1)]
        unsafe = self.id_tris()[1]
        check = True
        # check all unsafe triangles for that critical point
        for utri in unsafe:
            for pt in critpts:
                if utri != rm_candidate and pt in self.dt.simplices[utri]:
                    check = False
        return check

    def gen_poly(self, jaggedness, holes=0):
        tic = datetime.now()
        self.dt = spatial.Delaunay(self.points)
        n_edges = self.count_outer_edges()
        # holes
        for h in range(holes):
            safe, unsafe = self.id_tris2()
            if not safe:
                break
            rm = np.random.choice(list(safe), replace=False)
            self.del_tri(rm)
        # jaggedness
        edges_desired = int(n_edges * (jaggedness + 1))
        
        while n_edges < edges_desired:
            rm = None
            # get 1-edge tris by aspect ratio
            etris = sorted(list(self.id_tris()[1]), key=lambda tri: self.aspect(tri))
            for etri in etris:
                if self.check_edgesafety(etri, etris) == True:
                    rm = etri
                    break
            if rm is not None:
                self.del_tri(rm)
            else:
                print('None viable to remove.')
                break
            n_edges = self.count_outer_edges()

        toc = datetime.now()
        print('Generated a polygon with {} points and {} edges in {}'.format(self.dt.points.shape[0], n_edges, toc - tic))

    def del_tri(self, rm):
        '''Delete a tri

        Parameters
        ----------
        rm : int
            idx of the tri to delete
        '''
        # each neighbor in dt will have -1 where the neighboring simplex used to be.
        for i, nb in enumerate(self.dt.neighbors):
            for j, s in enumerate(nb):
                if rm == s:
                    self.dt.neighbors[i,j] = -1
        # we have to decrement all references to simplexes above rm because we're going to remove that simplex
        decrement_idx = np.zeros(np.shape(self.dt.neighbors), dtype='int32')
        for i, nb in enumerate(self.dt.neighbors):
            for j, s in enumerate(nb):
                if self.dt.neighbors[i,j] > rm:
                    decrement_idx[i,j] = -1
        self.dt.neighbors += decrement_idx
        # perform the deletion
        self.dt.simplices = np.delete(self.dt.simplices, rm, axis=0)
        self.dt.neighbors = np.delete(self.dt.neighbors, rm, axis=0)

def iterate_neighbors(idx, container):
    '''Helper to iterate idx through container

    Parameters
    ----------
    idx : int
        index
    container : list of instance
        the container containing items

    Returns
    -------
    tuple of instance of container
        i-1th, ith, i+1th instance of the container.
        if 
    
    Examples
    --------
    Use as
    `for v, item in enumerate(container):
        u, v, w = iterate_neighbors(v, container)
    `
    u, v, w stores the neighbors of v, wrapped around the first
    and last item of the list.

    '''
    if idx == len(container) - 1:
        a, b, c = container[idx-1], container[idx], container[0]
    elif idx == 0:
        a, b, c = container[-1], container[idx], container[idx+1]
    else:
        a, b, c = container[idx-1], container[idx], container[idx+1]
    return a, b, c




###################################################################################


def find_lower_upper(pts, v, e1, e2):
    o1 = (set(e1) - set([v])).pop()
    o2 = (set(e2) - set([v])).pop()

    # see which is lower/upper
    if pts[o1][1] < pts[o2][1]:
        lower,  upper, lower_e,upper_e = o1, o2, e1, e2
    else:
        lower,  upper, lower_e,upper_e = o2, o1, e2, e1
    # return lower then upper
    return lower, upper, lower_e, upper_e


def intersect_line(x, a, b):
    '''
    Check if the vertical line x=x intersects with the line
    formed by the points a, c and return intersection point
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

def line_sweep(poly, ax):
    # List of events (vertices/nodes)
    
    L = sorted(poly.G.nodes, key=lambda t: poly.points[t][0])
    # List of open cells
    O = []
    # List of closed cells
    C = []
    for i, v in enumerate(L):
        E = [L[i]]
        E, O, C, poly.points, poly.G = process_events(E, O, C, poly.points, poly.G)
    



    ### Plotting stuff
    # get vertices and sort by x position
    pos = {}
    for n in poly.G.nodes:
        pos[n] = poly.points[n]
    pos_higher = {}
    offmax = np.max(poly.points[:,1])
    offmin = np.min(poly.points[:,1])
    y_off = (offmax - offmin) * 0.04
    # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0], v[1]-y_off)
    edges, weights = zip(*nx.get_edge_attributes(poly.G,'weight').items())
    nx.draw(poly.G, pos, node_size=10, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.tab10)
    nx.draw_networkx_labels(poly.G, pos_higher, ax=ax, font_size=10)
    plt.show()
    



def check_edge(x, edge, points):
    if points[edge[0]][0] > x and x > points[edge[1]][0]:
        return True
    elif points[edge[1]][0] > x and x > points[edge[0]][0]:
        return True
    else:
        return False

def get_intersects(event, G, points):
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

def inflection(event, G, points):
    a, b = get_intersects(event, G, points)
    add = []
    if a:
        # add point to points list
        points = np.concatenate([points, np.array([a['pt']])])
        # index is the last member of new points array
        a_i = points.shape[0] - 1
        # add the new edge to G
        # G.add_edge(event, a_i, weight=3)
        G.add_edge(a_i, a['edge'][0], weight=a['edgew'])
        G.add_edge(a_i, a['edge'][1], weight=a['edgew'])
        G.remove_edge(a['edge'][0], a['edge'][1])
        add.append(a_i)
    if b:
        points = np.concatenate([points, np.array([b['pt']])])
        b_i = points.shape[0] - 1
        # G.add_edge(event, b_i, weight=3)
        G.add_edge(b_i, b['edge'][0], weight=b['edgew'])
        G.add_edge(b_i, b['edge'][1], weight=b['edgew'])
        G.remove_edge(b['edge'][0], b['edge'][1])
        add.append(b_i)
    return add, G, points

def lower_upper(points, event, G):
    vA, vB = tuple(G.adj[event])
    above = False
    if points[vA][1] > points[vB][1]:
        lower, upper = vA, vB
        above = False
    else:
        lower, upper = vB, vA
        above = True
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
            return Event.MERGE
        # entering below
        else:
            return Event.CLOSE
    # lower right, upper left
    elif points[lower][0] > points[event][0] and points[upper][0] < points[event][0]:
        return Event.INFLECTION
    # lower left, upper right
    elif points[lower][0] < points[event][0] and points[upper][0] > points[event][0]:
        return Event.INFLECTION


def process_events(E, O, C, points, G):
    split_merge = False
    for event in E:
        E_add = []
        event_type = check_lu(points, event, G)
        if event_type == Event.SPLIT:
            split_merge = True
            add, G, points = inflection(event, G, points)
            E_add.extend(add)
        elif event_type == Event.MERGE:
            split_merge = True
            add, G, points = inflection(event, G, points)
            E_add.extend(add)
    if E_add:
        # add new points to E
        E.extend(E_add)
        # sort by y-position        
        E = sorted(E, key=lambda e: points[e][1])

    print('---')
    for event in E:
        event_type = check_lu(points, event, G)
        print('vert={}, type={}'.format(event, event_type))
        if event_type == Event.OPEN:
            O.append([event])
            continue
        elif event_type == Event.CLOSE:
            for cell in O:
                if cell[-1] in G.adj[event]:
                    cell.append(event)
                    C.append(cell)
        elif event_type == Event.INFLECTION:
            for cell in O:
                for adj in G.adj[event]:
                    if adj in cell:
                        print('\tAppending...{}-->{}'.format(cell[-1], event))
                        cell.append(event)
    print('\tOpen:')
    for i in O:
        print('\t\t{}'.format(i))
    print('\tClosed:')
    for i in C:
        print('\t\t{}'.format(i))

    return E, O, C, points, G

            
    



def bcd(poly, ax):
    tic = datetime.now()
    vlist, elist = [], []
    newpoints = poly.points.tolist()
    vlist += [v for v in poly.G.nodes]
    elist += [e for e in poly.G.edges()]

    '''
    ### Plotting stuff
    # get vertices and sort by x position
    pos = {}
    for n in G.nodes:
        pos[n] = poly.points[n]
    pos_higher = {}
    y_off = .2  
    # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0]+y_off, v[1])
    nx.draw(G, pos, node_size=10, ax=ax)
    nx.draw_networkx_labels(G, pos_higher, ax=ax)
    '''

    # sort vlist by x val
    vlist = sorted(vlist, key=lambda v: poly.points[v][0])
    # stateful edge list
    L = set()
    # edge list, only we don't remove edges
    cellL = set()

    clist = []
    # boustrophedon cell points list

    new_graph_nlist = []
    new_graph_elist = []

    for v in vlist:
        cellL = L     
        es = [e for e in elist if v in e]
        l, u, le, ue = find_lower_upper(poly.points, v, es[0], es[1])
        assert(poly.points[l][1] < poly.points[u][1])
        # both right
        # IN
        if poly.points[l][0] > poly.points[v][0] and poly.points[u][0] > poly.points[v][0]:
            L |= set([ue, le])
            cellL |= set([ue, le])
            crit = True
        # both left
        # OUT
        elif poly.points[l][0] < poly.points[v][0] and poly.points[u][0] < poly.points[v][0]:
            L -= set([ue, le])
            crit = True
        # middle (lower is left)
        elif poly.points[l][0] < poly.points[v][0] and poly.points[u][0] > poly.points[v][0]:
            L -= set([le])
            L.add(ue)
            cellL.add(ue)
            crit = False
        # middle (upper is left)
        elif poly.points[u][0] < poly.points[v][0] and poly.points[l][0] > poly.points[v][0]:
            L -= set([ue])
            L.add(le)
            cellL.add(le)
            crit = False
        else:
            raise(Exception('lower/upper edge comparison failed!'))
        
        # print("v, {} crit: {}-->{}".format(crit, v, [e for e in L if v not in e]))

        # create new cell
        if crit:
            line_x = poly.points[v][0]
            ipts = []
            print('\tnew cell at {}'.format(line_x))
            # find all intersection points of x=v_x
            for edge in cellL:
                if v not in edge:
                    ipts.append( (intersect_line(line_x, poly.points[edge[0]], poly.points[edge[1]]), edge))

            # iterate through the list and find the y-neighbors of v
            # b corresponds to v, the current vertex
            new_v = (poly.points[v], v)
            ysorted_ipts = sorted(ipts, key=lambda t: t[0][1])

            a, a_i, c, c_i, a_iw, c_iw = -1, -1, -1, -1, -1, -1

            # add these intersection points to the list of points
            # and store their indices into a_i, c_i
            for i, pt in enumerate(ysorted_ipts):
                if pt[0][1] > poly.points[v][1]:
                    print(pt[1])
                    # a_i, c_i are new indices in newpoints
                    a = ysorted_ipts[i-1]
                    c = ysorted_ipts[i]
                    newpoints.append(a[0])
                    a_i = len(newpoints) - 1 
                    newpoints.append(c[0])
                    c_i = len(newpoints) - 1
                    break
            
            if (a, c, a_i, c_i) != (-1, -1, -1, -1):
                assert(newpoints[a_i] == a[0] and newpoints[c_i] == c[0])
                
                # create new graph with e[0] --> a_i, e[1] --> a_i, v --> a_i mappings
                new_graph_elist.extend([
                    (a[1][0], a_i, 1), 
                    (a[1][1], a_i, 1), 
                    (c[1][0], c_i, 1), 
                    (c[1][1], c_i, 1), 
                    (a_i, v, 3), 
                    (c_i, v, 3)
                ])
                new_graph_nlist.extend([a[1][0], a[1][1], c[1][0], c[1][1], a_i, c_i, v])

    P = nx.Graph()
    P.add_nodes_from(new_graph_nlist)
    P.add_weighted_edges_from(new_graph_elist)


    F = nx.compose(poly.G, P)
    ### Plotting stuff
    # get vertices and sort by x position
    pos = {}
    for n in F.nodes:
        pos[n] = newpoints[n]
    pos_higher = {}
    y_off = .08  
    # offset on the y axis
    for k, v in pos.items():
        pos_higher[k] = (v[0]+y_off, v[1])
    edges, weights = zip(*nx.get_edge_attributes(F,'weight').items())
    nx.draw(F, pos, node_size=10, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=plt.cm.tab10)
    # nx.draw_networkx_labels(F, pos_higher, ax=pax)


    toc = datetime.now()
    print('Generated BCD in {}'.format(toc - tic))



if __name__ == '__main__':
    poly = ConvPolygon(points=(2, 13, 1, 1), jaggedness=9, holes=1)
    fig = plt.figure()
    # ax1 = fig.add_subplot(121)
    # poly.chart(ax1)
    ax = fig.add_subplot()
    #ax.set_aspect('equal')
    bcd = line_sweep(poly, ax)
    # plt.show()