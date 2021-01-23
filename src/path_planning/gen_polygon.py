from matplotlib import pyplot as plt
import matplotlib
import numpy as np
from scipy import spatial
import logging
from datetime import datetime
from collections import defaultdict

def gen_cluster_points(no_clusters=4, cluster_n=40, cluster_size=80, cluster_dist=200):
    '''
    Generate points in one or more normally distributed clusters

    Parameters
    ----------
    `no_clusters`: int
        number of clusters to generate
    `cluster_n`: int
        number of points in each cluster
    `cluster_size`: float
        the variance of each cluster (smaller `cluster size` means tighter clusters)
    `cluster_dist`: float
        the mean distance between each cluster. Higher means clusters are further
        away from each other on average

    Returns
    -------
    array of float, shape (`no_clusters` * `cluster_n`, 2)
        The points
    '''
    pts = np.zeros((no_clusters * cluster_n, 2))
    loc = np.array([0,0], dtype='float64')
    for c in range(no_clusters):
        pts[c * cluster_n:(c+1)* cluster_n, :] = np.random.normal(loc=loc, scale=cluster_size, size=(cluster_n, 2))
        loc += np.random.normal(loc=0.0, scale=cluster_dist, size=np.shape(loc))
    return pts

class NonConvexPolygon(object):
    """
    goddamn this is insanely slow

    i am a genius programmer

    Attributes
    ----------
    dt : scipy.spatial.Delaunay
        the delaunay triangle object
    outside_pts: np.ndarray, float64
        n x 2 unordered ndarray of outside n points
    inside_pts: np.ndarray, float64
        n x 2 unordered ndarray of inside n points
    boundarypts: np.ndarray, float64
        n x 2 ordered ndarray of outside n points
    boundarylines: np.ndarray, float64
        n x 2 x 2 ordered ndarray of polygon lines
        
    """
    def __init__(self, points, jaggedness, holes=0, plot=False):
        '''
        Create a non-concave polygon.

        Parameters
        ----------
        points: `numpy.ndarray`
            2d array, which is an M x 2 matrix indicating M (x, y) points
        jaggedness: float
            0 jaggedness produces a convex polygon. 1 jaggedness removes 1x the number of edges of
            the convex polygon to produce a non-convex polygon, 2x removes 2x the edges, etc..
        plot: bool
            draws an interactive plot of the polygon creation process.
        '''
        self.points = points
        self.dt = self._gen_polygon(points, jaggedness, holes=holes, plot=plot)
        outside_pts_idx, inside_pts_idx = self._boundary_pts(self.dt)
        self.outside_pts, self.inside_pts = self.dt.points[outside_pts_idx], self.dt.points[inside_pts_idx]
        self.boundarypts = []
        self.boundarylines = []
        self.max_bounds = np.max(points, axis=0)
        self.min_bounds = np.min(points, axis=0)
        for l in self._order_boundary_pts(self.dt):
            self.boundarypts.append(self.dt.points[l])
            self.boundarylines.append(self._draw_boundarylines(self.dt.points[l]))

        self.neighbor_nodes = self._neighbor_nodes(self.dt)

        self.interiorpts = []
    
    def chart(self, ax, legend=False):
        '''
        Draw a chart of the figure on the Axes `ax`

        Returns
        -------
        `matplotlib.pyplot.Axes`
            Axis object containing a plot of the polygon.
        '''
        # can't figure out how to just make all tris the same color...
        centers = np.sum(self.points[self.dt.simplices], axis=1, dtype='int')/3.0
        centr = self._centroid(centers)
        colors = np.array([ (x - centr[0])**2 + (y - centr[1])**2 for x, y in centers])
        # draw colored tris
        # ax.tripcolor(self.points[:,0], self.points[:,1], self.dt.simplices, facecolors=colors, cmap='YlGn', edgecolors='darkgrey')
        # draw boundary
        for bps in self.boundarypts:
            ax.plot(bps[:,0], bps[:,1], 'k', label='boundary', zorder=2)
        # preserve aspect
        ax.set_aspect('equal')
        ax.set_facecolor('lightblue')
        if legend:
            ax.legend()

    def _draw_boundarylines(self, boundarypts):
        '''
        from an ordered sequence of boundary points, draw boundary lines

        zeroth pt is the connecting line between last pt and 0th pt.
        '''
        lines = np.empty(shape=(np.shape(boundarypts)[0] - 1, 2, 2), dtype='float64')
        for i, p in enumerate(boundarypts[:-1,:]):
            lines[i,0,:] = boundarypts[i]
            lines[i,1,:] = boundarypts[i+1]
        return lines
            
    def _centroid(self, pts):
        '''
        Find the centroid of a group of points

        Parameters
        ----------
        `pts`: array of float, shape (n,2)
            n Input points

        Returns
        -------
        array of float, shape (1,2)
            The point of the hull
        '''
        k = np.shape(pts)[0]
        return np.array(np.sum(pts, axis=0)/k, dtype='float64')

    def _gen_polygon(self, points, jaggedness, holes=0, plot=False):
        tic = datetime.now()
        # if we're plotting, init figure
        
        if plot:
            def plot_update(ax, dt):
                ax.clear()
                centers = np.sum(dt.points[dt.simplices], axis=1, dtype='int')/3.0
                centr = self._centroid(centers)
                o = list(self._2boundary_pts(dt))
                outers = dt.points[o]
                # might as well make it cool
                colors = np.array([ (x - centr[0])**2 + (y - centr[1])**2 for x, y in centers])
                # draw colored tris
                ax.tripcolor(dt.points[:,0], dt.points[:,1], dt.simplices, facecolors=colors, cmap='YlGn', edgecolors='darkgrey')
                # preserve aspect
                ax.scatter(outers[:,0], outers[:,1], c='brown', marker='.', label='outer')
                ax.set_aspect('equal')
                ax.set_facecolor('lightblue')
                plt.pause(.001)
            plt.ion()
            fig = plt.figure('Creating Polygon Interactive Plot Window')
            ax = plt.axes()
    
        # delaunay tri
        dt = spatial.Delaunay(points)
        # no of edges initialized to start dt
        edges = len(self._edge_tris_n(dt, 1)) + 2 * len(self._edge_tris_n(dt, 2))
        # desired no of edges comes from jaggedness multiplier
        edges_desired = int(edges * (jaggedness + 1))
        print('generating polygon with {} desired edges (from {}) and {} holes'.format(edges_desired, edges, holes))

        if plot:
            plot_update(ax, dt)

        ######## HOLES ########
        # outer, inner simplices
        for h in range(holes):
            edge, non_edge = self._edge_tris(dt)
            deleted_simplex = np.random.choice(list(non_edge), replace=False)
            self.interiorpts.append(self._centroid(dt.points[dt.simplices[deleted_simplex]]))
            dt = self._del_simplex(dt, deleted_simplex)
            if plot:
                plot_update(ax, dt)

        ######## REMOVE EDGES ONE BY ONE TO CREATE NON-CONVEX POLYGON ########
        while edges != edges_desired:
            logging.debug('edges: {}, desired: {}'.format(edges, edges_desired))
            rm = None

            # if we're plotting, update the plot
            if plot:
                plot_update(ax, dt)

            # we assume that we always have fewer edges than desired.
            # we can't have more edges than desired, because the desired no
            # of edges is set by `jaggedness` which cannot be less than 0
            # which means that a jaggedness 0 would theoretically never
            # enter this loop.
            if edges < edges_desired:
                # remove a random triangle with 1 outer edge
                single_edges = self._edge_tris_n(dt, 1)
                # sort by aspect ratio
                ars = [self._aspectratio(dt, s) for s in dt.simplices[single_edges]]
                sort_idx = np.argsort(ars)[::-1]
                single_edges = np.array(single_edges)[sort_idx]
                ars = np.array(ars)[sort_idx]

                # assume a split
                i = 0
                no_viable = False
                while True:
                    rm = single_edges[i]
                    not_rm = np.delete(single_edges, i, axis=0)

                    edgepts = dt.simplices[not_rm,:].flatten()
                    crit_edgepts = dt.simplices[rm, (dt.neighbors[rm,:] == -1)][0]
                    if not (edgepts == crit_edgepts).any():
                        break
                    i += 1
                    if i >= len(single_edges):
                        logging.debug('\tno viable simplex to remove!')
                        no_viable = True
                        break
                
                if no_viable == True:
                    break
                
                # perform the removal
                logging.debug('\tremoving 1-edge at idx {}: {}'.format(rm, dt.simplices[rm, :]))
                dt = self._del_simplex(dt, rm)
        
            # new count of edges
            edges = 1 * len(self._edge_tris_n(dt, 1)) + 2 * len(self._edge_tris_n(dt, 2))
            logging.debug('\tedges: {}: {} 1-edges and {} 2-edges'.format(edges, len(self._edge_tris_n(dt, 1)), len(self._edge_tris_n(dt, 2))))
        
        # turn off interactive plot
        if plot:
            plt.ioff()

        # log the time msg
        toc = datetime.now()
        time_msg = "Took {} to generate a polygon with {} points and {} edges.".format(toc-tic, np.shape(points)[0], edges_desired)
        print(time_msg)
        logging.debug(time_msg)

        return dt

    def _neighbor_nodes(self, dt):
        neigh = defaultdict(set)
        for s in dt.simplices:
            for i in s:
                other = set(s)
                other.remove(i)
                neigh[i] = neigh[i].union(other)
        return neigh

    def _aspectratio(self, dt, s):
        '''
        given a simplex s
        find aspect ratio of its triangle.
        '''
        p1 = dt.points[s][0,:] - dt.points[s][2,:]
        p2 = dt.points[s][1,:] - dt.points[s][0,:]
        p3 = dt.points[s][2,:] - dt.points[s][1,:]
        a = np.sqrt(p1.dot(p1.T))
        b = np.sqrt(p2.dot(p2.T))
        c = np.sqrt(p3.dot(p3.T))

        return a * b * c / ( (b + c - a)*(c + a - b)*(a + b - c) )

    def _2boundary_pts(self, dt):
        '''
        Find points within 2 nodes of a boundary
        '''
        out, inn = self._boundary_pts(dt)
        neigh = self._neighbor_nodes(dt)
        neighs = set()
        for o in out:
            neighs |= neigh[o]
        return neighs
        
    def _edge_tris(self, dt):
        all_tris = set(range(len(dt.simplices)))
        edge_tris = set()
        bps = self._2boundary_pts(dt)
        for i, s in enumerate(dt.simplices):
            if set(s) <= bps:
                edge_tris.add(i)
        return set(edge_tris), set(all_tris - edge_tris)


    def _edge_tris_n(self, dt, n):
        '''
        Find triangles within dt with `n` outer edges 

        Parameters
        ----------
        `dt`: `scipy.spatial.Delaunay`
            The triangulated polygon to check
        `n`: int [0,2]
            The no. of exterior edges. 0 is an interior triangle (no boundaries on external edge)
            1 is a triangle with 1 boundary on external edge, 2 is a triangle with 2 boundaries on
            external edge.

        Returns
        -------
        list of int
            The indexes to the simplices containing `n` outer boundaries. 
        
        Examples
        --------
        >>>s = dt.simplices[_edge_tris(dt, 1)]
        >>>print(s)

        [[2 5 -1], [4 -1 9], [9 -1 3], ...]

        `s` contains 1-edge boundary simplices of dt.
        '''
        edge_tris = []
        for i, s in enumerate(dt.neighbors):
            outers = (s == -1).sum()
            if outers == n:
                edge_tris.append(i)
        return edge_tris


    def _boundary_pts(self, dt):
        '''
        Find the points on the boundary. 
        NOTE: ordering of these points MATTERS!

        Returns
        -------
        tuple of `numpy.ndarray` of int
            (outer, inner) indices of points of the object
        '''
        # get boundary tris
        outer_simplices = self._edge_tris_n(dt, 1)
        if self._edge_tris_n(dt, 2):
            outer_simplices.extend(self._edge_tris_n(dt, 2))

        # select only points on boundary tris which are opposite
        # real tris (not empty space) and create mask
        s_outerpts_mask = dt.neighbors[outer_simplices,:] != -1
        # outer points are the unique set of the points selected by that mask
        outerpts = np.unique(dt.simplices[outer_simplices,:][s_outerpts_mask].flatten())
        innerpts = np.setdiff1d(np.unique(dt.simplices.flatten()), outerpts)
        return outerpts, innerpts

    def _order_boundary_pts(self, dt):
        '''
        Put the boundary points in order.

        Returns
        -------
        list of list of int
            list of indices to dt.points (one for each obj)
        '''
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
            t.append(t[0])
            ordered_objs.append(t)
            t = set(t)
            b = b - t
        return ordered_objs

    def _del_simplex(self, dt, rm):
        '''
        Delete a simplex from a dt object.
        Updates dt.neighbors and dt.simplices
        NOTE: Breaks qhull functionality...

        Parameters
        ----------
        dt: scipy.spatial.Delaunay
            the delaunay triangulation
        rm: int
            the index (in dt.simplices) of the simplex to remove

        Returns
        -------
        scipy.spatial.Delaunay
            updated delaunay tri with simplex `rm` gone
        '''
        # each neighbor in dt will have -1 where the neighboring simplex used to be.
        for i, nb in enumerate(dt.neighbors):
            for j, s in enumerate(nb):
                if rm == s:
                    logging.debug('\t\tfound ref to {} in simplex {}, neighbors: {}'.format(s, i, nb))
                    dt.neighbors[i,j] = -1
        
        # we have to decrement all references to simplexes above rm because we're going to remove that simplex
        decrement_idx = np.zeros(np.shape(dt.neighbors), dtype='int32')
        for i, nb in enumerate(dt.neighbors):
            for j, s in enumerate(nb):
                if dt.neighbors[i,j] > rm:
                    decrement_idx[i,j] = -1

        dt.neighbors += decrement_idx

        # remove entry for simplex and neighbor
        # TODO: refactor this whole thing
        for i, s in enumerate(dt.simplices):
            if rm == i:
                dt.simplices = np.delete(dt.simplices, i, axis=0)
                dt.neighbors = np.delete(dt.neighbors, i, axis=0)

        return dt

if __name__ == '__main__':
    single=True
    if single==True:
        ax = plt.subplot()
        xy = gen_cluster_points(no_clusters=10)
        newpoly = NonConvexPolygon(xy, jaggedness=13, holes=5, plot=False)
        newpoly.chart(ax)
        plt.show()
    else:
        r, c = 3, 4
        fig, axs = plt.subplots(r, c, figsize=(c*5, r*5))
        tic = datetime.now()
        for i in range(r):
            for j in range(c):
                print('generating {} out of {} polygons'.format(r*i + j, r * c))
                ax = axs[i,j]
                xy = gen_cluster_points(no_clusters=10)
                newpoly = NonConvexPolygon(xy, jaggedness=13, holes=5, plot=False)
                newpoly.chart(ax)
        toc = datetime.now()
        print('Finished! Took {}.'.format(toc - tic))
        plt.savefig('test.png', dpi=450)