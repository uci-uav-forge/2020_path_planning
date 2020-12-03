import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from matplotlib.collections import LineCollection
from scipy import linalg
import gen_polygon

def boustrophedon_decomp(poly, alpha, plot=False):
    '''
    returns list of (L, intersectpts)
    '''
    toc = datetime.now()

    print(poly.boundarypts)

    lines = []
    points = []
    points_neighbors = []
    for b, boundary in enumerate(poly.boundarypts):
        # check if we're inside a hole
        for i, v in enumerate(poly.boundarypts[b]):
            if i == 0:
                u = poly.boundarypts[b][-1,  :]
                w = poly.boundarypts[b][i + 1,  :]
            elif i == len(poly.boundarypts[b]) - 1:
                u = poly.boundarypts[b][i - 1,  :]
                w = poly.boundarypts[b][0,  :]
            else:
                u = poly.boundarypts[b][i - 1,  :]
                w = poly.boundarypts[b][i + 1,  :]
            # vertical sweep
            L = np.array([
                [poly.min_bounds[0], v[1]],
                [poly.max_bounds[0], v[1]]
            ])
        
            # number of intersections to find critical points
            intersects = 1
            for j, bl in enumerate(poly.boundarylines[b]):
                if intersect(L, bl):
                    intersects += 1

            intersectpts = []
            intersectptsneighbors = []

            # critical points
            if crit_check(u, v, w, L) and (intersects - 1) % 2 == 0:
                # find intersect points
                for closedshape in range(len(poly.boundarylines)):
                    for bl in poly.boundarylines[closedshape]:
                        x, y = intersectpt(bl, L)
                        if x and y:
                            print('intersect at <{}, {}>'.format(x[0], y[0]))
                            intersectpts.append([x[0],y[0]])
                            intersectptsneighbors.append(([bl[0,:], bl[1,:]]))

                    lines.append(L)
                    points.append(intersectpts)
                    points_neighbors.append(intersectptsneighbors)

    tic = datetime.now()
    print('BDC completed in {}'.format(tic - toc))
    return lines, points

def connectivity(regions):
    '''
    Parameters
    ----------
    regions : np.ndarray
        1 dimensional collection of regions
    
    Returns
    -------
    tuple (int, list of (int, int))
        the no. of connections and the connectivity points 
        of the 1d collection of `regions`
    '''
    r_prev = 0
    r_open = False
    r_connected = []
    connections, connectivity = 0, []
    for i, r in enumerate(regions):
        if r_prev == 0 and r == 1:
            r_open = True
            start = i
        if r_prev == 1 and r == 0 and r_open:
            r_open = False
            connections += 1
            end = i
            connectivity.append(start, end)
    return connections, connectivity

def intersect(M, N):
    '''
    check if two line segments, M and N, intersect, by checking
    orientation of their points
    (faster)
    '''
    ccw = lambda a, b, c : (c[1] - a[1])*(b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])
    # points on lines
    P = M[0,:]
    Q = M[1,:]
    R = N[0,:]
    S = N[1,:]
    # compute intersections
    return ccw(P, R, S) != ccw(Q, R, S) and ccw(P, Q, R) != ccw(P, Q, S)

def intersectpt(P, Q):
    '''
    Check intersectpt

    NOTE: the div/0 check is on Q...
    '''
    # solve system
    # x = x1 + t * (x2 - x1) = x3 + u * (x4-x3)
    # we solve for [t, u] first 
    # then solve for [x, y]

    x1, x3 = P[0,0], Q[0,0]
    y1, y3 = P[0,1], Q[0,1]
    x2, x4 = P[1,0], Q[1,0]
    y2, y4 = P[1,1], Q[1,1]

    A = np.array([
        [x2-x1, x3-x4],
        [y2-y1, y3-y4]
    ])
    b = np.array([
        [x3-x1],
        [y3-y1]
    ])

    t = linalg.solve(A, b)
    
    m = np.array([
        [x2-x1],
        [y2-y1]
    ])
    n = np.array([
        [x1],
        [y1]
    ])
    if 1 >= t[0] >= 0 and 1 >= t[1] >= 0:
        return (t[0] * m + n)[0], (t[0] * m + n)[1]
    else:
        return None, None

def crit_check(u, v, w, L, plot=False):
    '''
    Check if the vertex `v` is a critical vertex
    '''
    M, N = np.empty((2,2)), np.empty((2,2))
    M[0] = u
    M[1] = v
    N[0] = v
    N[1] = w


    if cross(M, L, sign=True, lines=True) > 0:
        s = True
    else:
        s = False
    if cross(N, L, sign=True, lines=True) > 0:
        t = True
    else:
        t = False
    
    if plot:
        ax = plt.gca()
        lines = np.asarray([M, N])
        test = LineCollection(lines, zorder=4, colors='r', linestyles=':')
        ax.add_collection(test)

    return (s ^ t)
    
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

    ax = plt.axes()
    pts = gen_polygon.gen_cluster_points(no_clusters=3, cluster_n=20, cluster_size=80, cluster_dist=120)
    poly = gen_polygon.NonConvexPolygon(pts, 4, 2, plot=True)
    poly.chart(ax)
    lines, points = boustrophedon_decomp(poly, 1)
    
    p = []
    for pset in points:
        p.extend(pset)
    p = np.asarray(p)
    xs = p[:,0]
    ys = p[:,1]

    ax.scatter(xs, ys, c='r', marker='.')
    segs = LineCollection(lines, colors='grey')
    ax.add_collection(segs)
    plt.show()