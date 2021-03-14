import numpy as np
from enum import Enum
from gen_polygon import ConvPolygon
import matplotlib.pyplot as plt
import networkx as nx
import math, itertools, string, copy
from matplotlib.transforms import offset_copy
from matplotlib.collections import LineCollection

plt.rcParams['figure.facecolor'] = 'grey'

SMALL_SIZE = 9
MEDIUM_SIZE = 11
BIGGER_SIZE = 13
plt.rcParams['figure.facecolor'] = 'grey'
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = SMALL_SIZE
plt.rcParams['axes.titlesize'] = MEDIUM_SIZE
plt.rcParams['axes.labelsize'] = SMALL_SIZE
plt.rcParams['xtick.labelsize'] = SMALL_SIZE
plt.rcParams['ytick.labelsize'] = SMALL_SIZE
plt.rcParams['legend.fontsize'] = SMALL_SIZE
plt.rcParams['figure.titlesize'] = BIGGER_SIZE

class Event(Enum):
    CLOSE=1
    OPEN=2
    SPLIT=3
    MERGE=4
    INFLECTION=5
    INTERSECT=6

class World(object):
    def __init__(self, poly=ConvPolygon(), theta=0):
        # the input polygon
        self.poly = poly
        # the points in 2d space which form the polygon
        self.points = copy.deepcopy(poly.points)
        # the structural graph of the world
        self.G = copy.deepcopy(poly.G)
        # list of sets() which represent a cell
        self.cells = self._line_sweep(theta)
        # reebgraph object
        self.Rg = self._build_reebgraph()
        # sweep line direction
        self.theta = theta
        self.cell_bboxes = self.cell_bbox()

        # scalar qualities are qualities by which we evaluate the decomposition...
        # some mean something, others... do not. lol
        self.scalar_qualities = {
            'avg_cell_width' : sum([d['w'] for d in [*self.cell_bboxes.values()]]) / len(self.cells),
            'min_cell_width' : min([d['w'] for d in [*self.cell_bboxes.values()]]),
            'avg_cell_aspect': sum([d['w']/d['h'] for d in [*self.cell_bboxes.values()]]) / len(self.cells),
            'min_cell_aspect': min([d['w']/d['h'] for d in [*self.cell_bboxes.values()]]),
            'estrada' : nx.estrada_index(self.Rg),
            'no_cells' : len(self.cells),
            'weiner' : nx.wiener_index(self.Rg),
            'assortivity_coeff' : nx.degree_assortativity_coefficient(self.Rg),
            'degrees' : sum([self.Rg.degree[n] for n in self.Rg.nodes]),
            'area_variance' : np.var(np.array(self.cell_areas()))
        }

    def cell_bbox(self):
        rotpts = np.array(self.points @ self._make_rot_matrix(-self.theta))
        bboxes = {}
        for c, cell in enumerate(self.cells):
            bboxes[c] = {
                'right' : np.max(rotpts[list(cell), 0]),
                'left' : np.min(rotpts[list(cell), 0]),
                'top' : np.max(rotpts[list(cell), 1]),
                'bot' : np.min(rotpts[list(cell), 1]),
                'w' : abs(np.max(rotpts[list(cell), 0]) - np.min(rotpts[list(cell), 0])),
                'h' : abs(np.max(rotpts[list(cell), 1]) - np.min(rotpts[list(cell), 1]))
            }
        return bboxes

    def cell_areas(self):
        areas = []
        for c in self.cells:
            x, y = np.squeeze(np.asarray(self.points[list(c), 0])), np.squeeze(np.asarray(self.points[list(c), 1]))
            # shoelace formula
            correction = x[-1] * y[0] - y[-1]* x[0]
            main_area = np.dot(x[:-1], y[1:]) - np.dot(y[:-1], x[1:])
            area = 0.5*np.abs(main_area + correction)
            areas.append(area)
        return areas
    
    def _intersect_line(self, x, a, b):
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

    def _check_edge(self, x, edge):
        if self.points[edge[0],0] > x and x > self.points[edge[1],0]:
            return True
        elif self.points[edge[1],0] > x and x > self.points[edge[0],0]:
            return True
        else:
            return False

    def _get_intersects(self, event):
        x, y = self.points[event,0], self.points[event,1]
        collisions = []
        # get all intersects
        for edge in self.G.edges():
            if self._check_edge(x, edge):
                # get the point of intersection
                ipt = self._intersect_line(x, self.points[edge[0]], self.points[edge[1]])
                # store its x, y, edge, edgew
                collision = {
                    'vx' : event, # the vertex associated with the collision
                    'pt' : ipt, # the point of the collision
                    'edge' : edge, # the edge with which the line collided
                    'edgew' : self.G[edge[0]][edge[1]]['weight'] # the weight of that edge
                }
                collisions.append(collision)
        above, below = None, None
        if collisions:
            above = min([c for c in collisions if c['pt'][1] > y], key=lambda x: abs(x['pt'][1]-y), default=None)
            below = min([c for c in collisions if c['pt'][1] < y], key=lambda x: abs(x['pt'][1]-y), default=None)
        return above, below

    def _qcross(self, u, v, w):
        a = self.points[v] - self.points[u]
        b = self.points[v] - self.points[w]
        if a[0] * b[1] - b[0] * a[1] >= 0:
            return True
        else:
            return False

    def _lower_upper(self, event):
        # we only consider the non-added edges
        # weight 3 means that edges were added by
        # prior split/merge sequences...
        for p in self.G.predecessors(event):
            if self.G[p][event]['weight'] != 3:
                vA = p
        for s in self.G.successors(event):
            if self.G[event][s]['weight'] != 3:
                vB = s

        above = False
        if self._qcross(vA, event, vB):
            above = True
            lower, upper = vB, vA
        else:        
            lower, upper = vA, vB
        return lower, upper, above

    def _check_lu(self, event):
        lower, upper, above = self._lower_upper(event)
        # both right
        if self.points[lower,0] > self.points[event,0] and self.points[upper,0] > self.points[event,0]:
            # entering above
            if above:
                return Event.OPEN
            # entering below
            else:
                return Event.SPLIT
        # both left
        elif self.points[lower,0] < self.points[event,0] and self.points[upper,0] < self.points[event,0]:
            # entering above
            if above:
                return Event.CLOSE
            # entering below
            else:
                return Event.MERGE
        # lower right, upper left
        elif self.points[lower,0] > self.points[event,0] and self.points[upper,0] < self.points[event,0]:
            return Event.INFLECTION
        # lower left, upper right
        elif self.points[lower,0] < self.points[event,0] and self.points[upper,0] > self.points[event,0]:
            return Event.INFLECTION

    def _node_classify(self, v, A, crits, vtypes):
        etype = self._check_lu(v)
        add = (None, None)
        if etype == Event.SPLIT or etype == Event.MERGE:
            add = self._make_splitmerge_points(v, etype)
            A[v] = add
            # print('\tv: {}, type: {}, new points: {}'.format(v, etype, add))
        if etype in [Event.OPEN, Event.SPLIT, Event.MERGE, Event.CLOSE]:
            crits.append((v, etype))
        vtypes[v] = etype
        for a in add:
            if a is not None:
                vtypes[a] = Event.INTERSECT
        return A, crits, vtypes

    def _get_addpt_neighbors(self, a, want3=False):
        neigh = set()
        if not want3:
            for p in self.G.predecessors(a):
                if self.G[p][a]['weight'] != 3:
                    neigh.add(p)
            for s in self.G.successors(a):
                if self.G[a][s]['weight'] != 3:
                    neigh.add(s)
        else:
            for p in self.G.predecessors(a):
                if self.G[p][a]['weight'] == 3:
                    neigh.add(p)
            for s in self.G.successors(a):
                if self.G[a][s]['weight'] == 3:
                    neigh.add(s)
        return neigh

    def _make_splitmerge_points(self, event, event_type):
        '''returns add list, updated G, updated points'''
        a, b = self._get_intersects(event)
        a_i, b_i = None, None
        if a:
            # add point to points list
            self.points = np.concatenate([self.points, np.array([a['pt']])])
            # index is the last member of new points array
            a_i = self.points.shape[0] - 1
            # add the new edge to G
            if event_type == Event.SPLIT:
                self.G.add_edge(a_i, event, weight=3) # close
                self.G.add_edge(event, a_i, weight=4) # open
            elif event_type == Event.MERGE:
                self.G.add_edge(a_i, event, weight=3) # open
                self.G.add_edge(event, a_i, weight=4) # close
            self.G.add_edge(a['edge'][0], a_i, weight=a['edgew'])
            self.G.add_edge(a_i, a['edge'][1], weight=a['edgew'])
            self.G.remove_edge(a['edge'][0], a['edge'][1])
        if b:
            self.points = np.concatenate([self.points, np.array([b['pt']])])
            b_i = self.points.shape[0] - 1
            if event_type == Event.SPLIT:
                self.G.add_edge(event, b_i, weight=3) # open
                self.G.add_edge(b_i, event, weight=4) # close
            elif event_type == Event.MERGE:
                self.G.add_edge(event, b_i, weight=3) # open
                self.G.add_edge(b_i, event, weight=4) # close
            self.G.add_edge(b['edge'][0], b_i, weight=b['edgew'])
            self.G.add_edge(b_i, b['edge'][1], weight=b['edgew'])
            self.G.remove_edge(b['edge'][0], b['edge'][1])
        return (a_i, b_i)

    def _neigh(self, v):
        neigh = set()
        if v:
            neigh |= set(self.G.succ[v])
            neigh |= set(self.G.pred[v])
        return neigh

    @staticmethod
    def _make_rot_matrix(theta):
        return np.matrix([
            [np.cos(theta), -1*np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])

    def _line_sweep(self, theta):
        # first, rotate points
        self.points = np.array(self.points @ self._make_rot_matrix(theta))
        # List of events (vertices/nodes)
        L = sorted(self.G.nodes, key=lambda t: self.points[t,0])
        # List of closed cells
        C = []
        # Additional points found in splits & merges
        A = {}
        crits, vtypes = [], {}
        for v in L:
            A, crits, vtypes = self._node_classify(v, A, crits, vtypes)        
        for v in crits:
            C = self._process_events(v, vtypes, C)
        # rotate back
        self.points = np.dot(self.points, self._make_rot_matrix(-theta))
        return C

    def _right_turn(self, u, v, w):
        a = self.points[v] - self.points[u]
        b = self.points[w] - self.points[v]
        a = a/np.linalg.norm(a)
        b = b/np.linalg.norm(b)
        return a[0] * b[1] - b[0] * a[1]

    def _process_events(self, v, vtypes, C):
        prev_node = v[0] # inititate prior for c product
        for path_start in self.G.adj[v[0]]:
            path_end = False
            path = []
            # start the path
            node = path_start
            n = 0
            while path_end == False:
                cvals = []
                for possible_node in self.G.adj[node]:
                    if possible_node != prev_node:
                        # calculate cross product and append
                        cval = self._right_turn(prev_node, node, possible_node)
                        cvals.append( (possible_node, cval) )
                
                cvals = sorted([cval for cval in cvals], key=lambda t: t[1])
                # choose node with most CW pointing cval
                best = cvals[0][0]
                # replace previous with current
                prev_node = node
                # replace node with the one we chose
                node = best
                # append to list
                path.append(best)
                # if we're back at the original node, then we know we have formed a loop
                # and therefore have formed a cell!
                if best == path_start:
                    path_end = True
                elif n >= 1e5:
                    raise(Exception('Path Not Closed --> Exceeded Max Path Length!'))
                n += 1
            if set(path) not in C:
                C.append(set(path))
        return C

    def cell_chart(self):
        sq = math.ceil(np.sqrt(len(self.cells)))
        fig, ax = plt.subplots(nrows=sq, ncols=sq, facecolor='gray')
        fig.suptitle('Cells')
        for i in range(sq):
            for j in range(sq):
                ci = i * sq + j
                if ci < len(self.cells):
                    cell = self.cells[ci]
                    centr = self._cell_centroid(cell)
                    H = nx.subgraph(self.G, cell)
                    self.graph_chart(H, ax[i,j])
                    ax[i,j].text(centr[0], centr[1], str(ci), fontsize=16)
    
    def graph_chart(self, G, ax):
        cm = plt.get_cmap('viridis')
        pos = {}
        for n in G.nodes:
            pos[n] = [self.points[n,0], self.points[n,1]]
        edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
        nx.draw(G, pos, node_size=16, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=cm)
        ax.set_aspect('equal')
        def offset(ax, x, y):
            return offset_copy(ax.transData, x=x, y=y, units='dots')
        for n in G.nodes:
            x, y = self.points[n,0], self.points[n,1]
            ax.text(x, y, str(n), fontsize=9, transform=offset(ax, 0, 5), ha='center', va='center')
    
    def world_chart(self, ax):
        cm = plt.get_cmap('viridis')
        pos = {}
        H = nx.to_undirected(self.G)
        for n in H.nodes:
            pos[n] = [self.points[n,0], self.points[n,1]]
        edges, weights = zip(*nx.get_edge_attributes(H, 'weight').items())
        nx.draw(H, pos, node_size=16, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=cm)
        ax.set_aspect('equal')
        def offset(ax, x, y):
            return offset_copy(ax.transData, x=x, y=y, units='dots')
        for n in H.nodes:
            x, y = self.points[n,0], self.points[n,1]
            ax.text(x, y, str(n), fontsize=8, transform=offset(ax, 0, 5), ha='center', va='center')
        for i, cell in enumerate(self.cells):
            centr = self._cell_centroid(cell)
            ax.text(centr[0], centr[1], str(self._int_to_alph(i+1)), fontsize=14, ha='center', va='center')
        
    @staticmethod
    def _int_to_alph(x):
        result = []
        while x:
            x, r = divmod(x-1, 26)
            result[:0] = string.ascii_uppercase[r]
        return ''.join(result)
    
    def chart_reebgraph(self, ax, draw_chart_behind=False, point_text=False):
        pos = {}
        for n in self.Rg.nodes:
            pos[n] = self.Rg.nodes[n]['centroid']
        nx.draw(self.Rg, pos, ax=ax, node_size=14, node_color='red', width=2)
        if draw_chart_behind:
            H = nx.to_undirected(self.G)
            for n in H.nodes:
                pos[n] = [self.points[n,0], self.points[n,1]]
            edges, weights = zip(*nx.get_edge_attributes(H, 'weight').items())
            nx.draw(H, pos, node_size=11, ax=ax, edgelist=edges, edge_color=weights, edge_cmap=plt.get_cmap('copper'))            
            ax.set_aspect('equal')
            def offset(ax, x, y):
                return offset_copy(ax.transData, x=x, y=y, units='dots')
            if point_text:
                for n in H.nodes:
                    x, y = self.points[n,0], self.points[n,1]
                    ax.text(x, y, str(n), fontsize=8, transform=offset(ax, 0, 5), ha='center', va='center')

            for i, cell in enumerate(self.cells):
                centr = self._cell_centroid(cell)
                ax.text(
                    centr[0], 
                    centr[1], 
                    str(self._int_to_alph(i+1)), 
                    fontsize=10, 
                    ha='center', va='center', 
                    transform=offset(ax, 0,-10),
                    color='k'
                )

        
    def _build_reebgraph(self):
        Rg = nx.Graph()
        subgraph_list = []
        for c in self.cells:
            subgraph_list.append(nx.subgraph(self.G, c))
        
        node_data, edges = {}, []
        for i, ci in enumerate(self.cells):
            for j, cj in enumerate(self.cells):
                common_edge = ci & cj
                if len(common_edge) >= 2:
                    e = (i, j)
                    node_data[i] = {
                        'cell' : ci,
                        'centroid' : self._cell_centroid(ci),
                        'name' : self._int_to_alph(i+1)
                        }
                    node_data[j] = {
                        'cell' : cj,
                        'centroid' : self._cell_centroid(cj),
                        'name' : self._int_to_alph(j + 1)
                        }
                    edges.append(e)
        # build RG
        Rg.add_edges_from(edges)
        nx.set_node_attributes(Rg, node_data)
        return Rg
    
    def _cell_centroid(self, cell):
        ctr = np.sum(self.points[list(cell)], axis=0) / len(list(cell))
        return [ctr[0,0], ctr[0,1]]
                    

if __name__ == '__main__':
    print('nothing')