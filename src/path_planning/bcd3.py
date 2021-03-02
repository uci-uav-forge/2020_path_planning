import numpy as np
from scipy import spatial
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import LineCollection, PatchCollection
from datetime import datetime
from collections import OrderedDict, deque
from scipy import linalg
import networkx as nx

class ConvPolygon(object):
    def __init__(self, points=(2, 40, 10, 40), jaggedness=2, holes=0):
        self.points = self.gen_cluster_points(*points)
        self.gen_poly(jaggedness=jaggedness, holes=holes)
        self.blist = self.make_boundaries()
        self.graphs = self.shape_graph()

        
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
                elist.append( (v[0], k) )
            G.add_nodes_from(nlist)
            G.add_edges_from(elist)
            graphs.append((G, outer))
        return graphs

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


def bcd(poly, ax):
    tic = datetime.now()

    critpts = []
    intersects = []
    intersectpts = []

    for G, outer in poly.graphs:
        start_node = list(G.nodes)[0]

    
        # PLOT POSITIONS FOR NODES
        # get vertices and sort by x position
        pos = {}
        for n in G.nodes:
            pos[n] = poly.points[n]
        pos_higher = {}
        y_off = 15  
        # offset on the y axis
        for k, v in pos.items():
            pos_higher[k] = (v[0]+y_off, v[1])
        

        # Iterate through nodes and check if they are critical
        ordered_nodes = list(nx.dfs_preorder_nodes(G, source=start_node))
        for j, _ in enumerate(ordered_nodes):
            i, j, k = iterate_neighbors(j, ordered_nodes)
            crit, crit_type = check_pt(poly.points, outer, i, j, k)
            if crit:
                ips = []
                for H, h_outer in poly.graphs:
                    h_ordered_nodes = list(nx.dfs_preorder_nodes(G, source=start_node))
                    for m, _ in enumerate(h_ordered_nodes):
                        l, m, _ = iterate_neighbors(m, h_ordered_nodes)
                        ip = intersect_line(poly.points[j][0], poly.points[l], poly.points[m])
                        if ip and ip not in ips:
                            ips.append(ip)

                intersects.append( (j, ips) )

        nx.draw(G, pos, node_size=10, ax=ax)
        # nx.draw_networkx_labels(G, pos_higher, ax=ax)
        
    for c in intersects:
        print('point: {} --> {}'.format(c[0], poly.points[c[0]]))
        for k in c[1]:
            print('\tpoints: {}'.format(k))
            intersectpts.append(k)
        critpts.append(c[0])

    ax.plot(poly.points[critpts, 0], poly.points[critpts, 1], 'rx')
    ax.plot([i[0] for i in intersectpts], [i[1] for i in intersectpts], 'y.')
            
    toc = datetime.now()
    print('Generated BCD in {}'.format(toc - tic))


def check_pt(points, outer, i, j, k):
    '''
    check the points i, j, k in the points list 
    '''
    is_crit = False
    crit_type = None
    if points[i][0] > points[j][0] and points[k][0] > points[j][0]:
        is_crit = True
        if outer:
            crit_type = 'opening'
        else:
            crit_type = 'split'
    elif points[i][0] < points[j][0] and points[k][0] < points[j][0]:
        is_crit = True
        if outer:
            crit_type = 'closing'
        else:
            crit_type = 'merge'
    return is_crit, crit_type

def intersectv(Q, x):
    '''Check if a line segment Q intersects with a vertical line
    on the point x.

    Parameters
    ----------
    Q : np.ndarray of shape (2 x 2)
        line: p1, p2
    x : float64
        location of vertical line segment

    Returns
    -------
    None or float64
        The intersection y- point or None if no intersection
    '''
    x1, x2, y1, y2 = Q[0,0], Q[1,0], Q[0, 1], Q[1,1]
    m = (y2 - y1) / (x2 - x1)
    y = y1 + m * (x - x1)
    if y1 <= y <= y2:
        return y
    else:
        return None

def intersect(M, N):
    '''
    check _if_ two lines, M & N intersect (faster)
    '''
    ccw = lambda a, b, c : (c[1] - a[1])*(b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    # points on lines
    P, Q, R, S = M[0,:], M[1,:], N[0,:], N[1,:]
    # compute intersections
    return ccw(P, R, S) != ccw(Q, R, S) and ccw(P, Q, R) != ccw(P, Q, S)


def intersect_line(x, a, c):
    '''
    Check if the vertical line x=x intersects with the line
    formed by the points a, c and return intersection point
    '''
    is_intersect = False

    if a[0] < x and x < c[0]:
        is_intersect = True
        p1 = a
        p2 = c
    elif a[0] > x and x > c[0]:
        is_intersect = True
        p1 = c
        p2 = a

    if is_intersect:
        px = x
        m = abs(x - p1[0]) / abs(p2[0] - p1[0])
        py = p1[1] + m * (p2[1] - p1[1])

        return [px, py]
    else:
        return None

def intersectpt(P, Q):
    '''
    Find the intersection point between two line segments, P and Q.
    Returns a length 2 array of x, y vals, 
    or None if they don't intersect.
    '''
    # solve system
    # x = x1 + t * (x2 - x1) = x3 + u * (x4-x3)
    # we solve for [t, u] first 
    # then solve for [x, y]
    x1, x2, x3, x4 = P[0,0], P[1,0], Q[0,0], Q[1,0]
    y1, y2, y3, y4 = P[0,1], P[1,1], Q[0,1], Q[1,1]
    A = np.array([ [x2-x1, x3-x4], [y2-y1, y3-y4] ])
    b = np.array([ [x3-x1], [y3-y1] ])
    t = linalg.solve(A, b)
    m = np.array([ [x2-x1], [y2-y1] ])
    n = np.array([ [x1], [y1] ])
    if 1 >= t[0] >= 0 and 1 >= t[1] >= 0:
        return t[0] * m + n
    else:
        return None

def crit_check(u, v, w, outer):
    '''
    Check if the vertex `v` is a critical vertex
    '''
    cross = (u[0] - v[0]) * (w[1] - v[1]) -  (u[1] - v[1]) * (w[0] - v[0])
    if outer:
        if cross > 0:
            return True
        else:
            return False
    else:
        if cross < 0:
            return True
        else:
            return False
    
def cross(M, N, sign=False, lines=False):
    '''
    2d cross product of two lines M & N

    Parameters
    ----------
    L1, L2 : np.ndarray, dtype=float64
        2 x 2 ndarray: rows are start point end point, cols are x, y
    sign: bool
        return -1 or 1 corresponding to the sign of M x N
        faster because it skips expensive sqrt operation
        defaults to False
    lines: 
        whether we take x product of lines or points. Default M, N
        are length 2 arrays or tuples of points; if true, M, N are
        2x2 arrays of lines.
    '''
    if lines:
        u = M[1,:] - M[0,:]
        v = N[1,:] - N[0,:]
    else:
        u = M
        v = N

    if sign:
        return int(np.sign(u[0]*v[1] - u[1]*v[0]))
    else:
        u = u / np.sqrt(u.dot(u.T))
        v = v / np.sqrt(v.dot(v.T))
        return u[0]*v[1] - u[1]*v[0]

def conv_vert(pts):
    '''
    Takes a list of coordinates, assumes they connect in the sequence they are 
    listed to form a polygon, and returns the indices of coordinates that are 
    convex vertices of that polygon

    Parameters
    ----------
    pts : ndarray, dtype=float64
        2d M x 2 array of `M` points. NOTE: First and last M must be identical
        and are the opening/closing point.
    
    Returns
    -------
    tuple of set
        convex pt indices, concave pt indices
    '''
    convex = set()
    concave = set()
    for i, p in enumerate(pts[:-1]):
        if cross(pts[i], pts[i+1], sign=True) == -1:
            convex.add(i)
        else:
            concave.add(i)
    return convex, concave

def check_inside(p1, p2, p3, L):
    '''
    check if a line L is locally contained within the boundary formed by p1, p2, p3.
    
    Parameters
    ----------
    p1, p2, p3 : np.ndarray, dtype=float64
        length 2 array of points
    L : np.ndarray, dtype=float64
        2x2 array which form a line segment from two points
    
    Returns
    -------
    bool
        True if locally contained
        False if not contained
    '''
    M = np.array([
        p1,
        p2
    ])
    N = np.array([
        p2,
        p3
    ])
    if cross(M, L, sign=True, lines=True) ^ cross(L, N, sign=True, lines=True):
        return True
    else:
        return False

if __name__ == '__main__':
    poly = ConvPolygon(points=(2, 20, 40, 90), jaggedness=8, holes=2)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)

    poly.chart(ax1)
    ax = fig.add_subplot(122)
    ax.set_aspect('equal')
    bcd = bcd(poly, ax)
    plt.show()