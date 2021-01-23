import numpy as np

class Boundary(object):
    def __init__(self, xy: np.ndarray, outer: bool):
        self.xy = xy        # ordered xy points of the boundary
        self.outer = outer  # inner vs outer boundary
    
class Graph(object):
    def __init__(self, )
    
class Polygon(object):
    def __init__(self)


    def edges(self, dt, tribounds):
        '''
        dt: delaunay triangulation object
        tribounds: integer no of triangles bounding that 
        '''
        print(dt.vertex_neighbor_vertices[0].shape)
        print(dt.vertex_neighbor_vertices[1].shape)
        for vertex, nvertices in dt.vertex_neighbor_vertices:
            for nv in nvertices:
                (vertex, nv)

    def edges(dt, n):
        edges = set() # internal tris

        for tri_idx, nb_idx in enumerate(dt.neighbors):
            for e in range(3):
                n_outer = (e == -1).sum() # no of "empty" neighbors
                if n_outer == n:
                    edge_tris.append(i)
        return edge_tris

    def gen_polygon(self, points, jaggedness, holes=0):
        toc = datetime.now()
        dt = spatial.Delaunay(points)

        n_edges = 
        n_desired = int(n_edges * jaggedness + n_edges)



    def _gen_polygon(self, points, jaggedness, holes=0, plot=False):
        # start timer
        toc = datetime.now()
        dt = spatial.Delaunay(points)
        edges = len(self._edge_tris_n(dt, 1)) + 2 * len(self._edge_tris_n(dt, 2))
        # desired no of edges comes from jaggedness multiplier
        edges_desired = int(edges * (jaggedness + 1))
        print('generating polygon with {} desired edges (from {})'.format(edges_desired, edges))

        if plot:
            plot_update(ax, dt)

        ######## HOLES ########
        # outer, inner simplices
        for h in range(holes):
            edge, non_edge = self._edge_tris(dt)
            dt = self._del_simplex(dt, np.random.choice(list(non_edge), replace=False))
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
