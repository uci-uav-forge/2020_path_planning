from bcd import World
import bcd
from tqdm import tqdm
from gen_polygon import ConvPolygon
from sub_sweep import PathAnimator, Sweep
from multiprocessing.pool import Pool
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime
import copy
import math


class PathGenerator(object):
    def __init__(self, poly):
        self.SCALAR_QUALITIES = {
            'avg_cell_width',
            'min_cell_width',
            'avg_cell_aspect',
            'min_cell_aspect',
            'no_cells',
            'degrees',
            'area_variance',
        }
        self.polygon = poly
        self.worlds = self.sort_worlds(self.create_worlds(6), 'no_cells')
        # either choose 0 or -1 depending on if you want to minimize or maximize
        # scalar quality...
        self.world = self.worlds[0]

    
    def create_worlds(self, no_worlds=180):
        angles = np.linspace(0, np.pi, no_worlds)
        worlds = []
        print('creating Worlds...')
        for angle in tqdm(angles):
            addworld = lambda angle: World(poly=self.polygon, theta=angle)
            worlds.append(addworld(angle))
        print('done!')
        return worlds
    
    def sort_worlds(self, worlds, sortby):
        if sortby not in self.SCALAR_QUALITIES:
            raise ValueError('sortby must be a key of worlds.scalar_qualities. (sortby=\'{}\')'.format(sortby))
        q_function = lambda x: x.scalar_qualities[sortby]
        return sorted([ w for w in worlds], key=q_function)

    def centroid(self, ci):
        cell = self.world.cells[ci]
        return np.sum(self.world.points[list(cell)], axis=0) / len(list(cell))

    def cell_sweep(self, cellidx, sweep_w):
        '''given a world, a cell in that world's Reeb Graph, and a sweep width, sweep the cell

        Parameters
        ----------
        world : World
            the world
        cellidx : int
            index of the cell being swept
        sweep_w : float
            world's sweep width

        Returns
        -------
        tuple of Mx2 array
            the points of the sweep
        '''
        if cellidx not in list(range(len(self.world.cells_list))):
            raise ValueError('Cell index passed is not in cells list. cell_idx={}, world.cells_list={}'.format(cellidx, world.cells_list))
        # get boundaries of cell
        cell_pts = np.array(self.world.points[self.world.cells_list[cellidx]])
        # turn x, y to list
        cell_x = np.squeeze(cell_pts[:,0]).tolist()
        cell_y = np.squeeze(cell_pts[:,1]).tolist()
        # create sweeper
        sweeper = Sweep(use_theta=True, theta=-self.world.theta)
        px, py = sweeper.planning(cell_x, cell_y, sweep_w)
        return (np.array([px, py]).T)

    def get_path(self, sweep_w):
        # eulerian path through Rg
        rgpath = [(a, b) for (a, b) in list(nx.eulerian_circuit(nx.eulerize(self.world.Rg))) if a != b]
        entire = []
        visited = set()
        for i, (u, v) in enumerate(rgpath[:-1]):
            if i == 0:
                entire.append( (np.asarray(self.cell_sweep(u, sweep_w)), u) )
                visited.add(u)
            # go in to center of cell
            entire.append( (self.to_center_of_edge(u, v, v, to_center=True), v) )
            # lawnmower if not already scanned
            if v not in visited:
                entire.append( (np.asarray(self.cell_sweep(v, sweep_w)), v) )
            # go out to outside of cell
            w, x = rgpath[i+1]
            entire.append( (self.to_center_of_edge(w, x, v, to_center=False), v) )
            visited.add(v)
        return entire

    def to_center_of_edge(self, a, b, v, to_center=True):
        A = self.world.points[self.world.Rg[a][b]['common'][0]]
        B = self.world.points[self.world.Rg[a][b]['common'][1]]
        midpoint = B + 0.5 * (A - B)
        distance = np.linalg.norm(np.array(self.centroid(v) - midpoint)).sum()    
        if to_center:
            return np.array(np.linspace(self.centroid(v), midpoint, math.ceil(distance)))
        else:
            return np.array(np.linspace(midpoint, self.centroid(v), math.ceil(distance)))

    def show_path_animation(self, save=False):
        '''Generate and show a path animation in the plotting window.

        Parameters
        ----------
        save : bool, optional
            save animation to mp4, by default False
        '''
        fig = plt.figure('Path Planner', figsize=(24, 13.5), facecolor='lightgrey')
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax1 = bcd.draw_graph(
            ax1, 
            self.world.G,
            self.world.points,
            node_marker='.',
            edge_colors=('black','black','silver','silver'),
            node_text=False, 
            chart_name='World',
            cell_text=True)

        cell_centroids = np.squeeze(np.array([c for c in self.world.cell_centroids.values()]))
        ax2 = bcd.draw_graph(
            ax2,
            self.world.Rg,
            np.array(cell_centroids)
        )
        ax2 = bcd.draw_graph(
            ax2, 
            self.world.G, 
            self.world.points,
            node_color='lightgrey',
            edge_colors=('silver','silver','silver','silver'),
            chart_name='Reeb Graph'
        )

        path_animator = PathAnimator()
        path_animator.animate(path, self.world, fig, ax1, ax2, save=save, savepath='./planner_alpha.mp4')
        plt.show()


poly = ConvPolygon(points=(6, 15, 10, 10), jaggedness=12, holes=3)
pathgen = PathGenerator(poly)
path = pathgen.get_path(0.8)
pathgen.show_path_animation()