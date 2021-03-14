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

Polygons contain:

    - a "master" list of points which includes interior points
    - a networkX graph of points organized by index
    - a list of list of boundaries (lines) for each edge 
    - a list of list of points which are outer bounds and holes

Written by Mike Sutherland
'''
import numpy as np
import networkx as nx
from datetime import datetime
from scipy import spatial
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.transforms import offset_copy

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
        Mx2 np.array
            list of M xy points --> M x 2
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
            G = nx.DiGraph()
            elist, nlist = [], []
            
            # check if we flip 
            cw = 0.0
            for k, v in boundary.items():
                x1, y1 = tuple(self.points[k])
                x2, y2 = tuple(self.points[v[0]])
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

    def chart(self, ax, G, cm=plt.cm.tab10):
        '''
        Plot the graph `G` on `ax`
        '''
        pos = {}
        for n in self.G.nodes:
            pos[n] = self.points[n]
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G, pos, node_size=16, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=cm)
        ax.set_aspect('equal')
        def offset(ax, x, y):
            return offset_copy(ax.transData, x=x, y=y, units='dots')
        for n in G.nodes:
            x, y = self.points[n][0], self.points[n][1]
            ax.text(x, y, str(n), fontsize=9, transform=offset(ax, 0, 5))

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
        '''
        Order the boundaries by their topological structure
        '''
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
        '''=create boundary dict structure
        keys are boundary point idx
        values are tuple (prev idx, next idx, bool outer)
        '''
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
        '''
        Generate the polgyon. This algorithm is _very_ slow, it probably has n^3
        time complexity or something like that. It works up to polygons of about 3
        or 400 points or less.
        '''
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

if __name__ == '__main__':
    poly = ConvPolygon()