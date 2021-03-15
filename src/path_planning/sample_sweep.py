'''
Sample sweep a random polygon
'''
from bcd import World
from gen_polygon import ConvPolygon
from sub_sweep import PathAnimator, Sweep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

poly = ConvPolygon(points=(3, 24, 3, 10), jaggedness=2, holes=1)
angles = np.linspace(0, np.pi, 30)
worlds = []
print('building worlds...')
for angle in angles:
    print('{}'.format(str(round(angle, 2)).ljust(7)), end='', flush=True)
    print(angle * 180/np.pi, end='\r', flush=True)
    worlds.append( World(poly=poly, theta=angle) )
print('done!')
q_function = lambda x: x[1].scalar_qualities['min_cell_width']
cwsorted = sorted([ (idx, world) for (idx, world) in enumerate(worlds) ], key=q_function)
# metric highest
best = cwsorted[-1][1]
# metric lowest
worst = cwsorted[0][1]
world = worst
allpx, allpy = [], []

allcell = []
ordered_cell_idx = nx.dfs_preorder_nodes(world.Rg, min([n for n in world.Rg.nodes], key=lambda n: world.Rg.nodes[n]['centroid'][0]))


cell_visited = []

def centroid(ci):
    cell = world.cells[ci]
    return np.sum(world.points[list(cell)], axis=0) / len(list(cell))

celldata = {}

def get_path(world):
    cellidx = list(nx.edge_dfs(world.Rg))
    cell_path_data = []
    entire = []
    for i, edge in enumerate(cellidx):
        u, v = edge
        # common edges between u and v
        commonA, commonB = tuple(world.Rg[u][v]['common'])
        # visit cell border
        cellbord = world.points[commonB] + 0.5 * (world.points[commonA] - world.points[commonB])
        if v not in cell_visited:
            # go to center first
            centry = centroid(v)
            cell_pts = np.array(world.points[world.cells_list[v]])
            cell_x = np.squeeze(cell_pts[:,0]).tolist()
            cell_y = np.squeeze(cell_pts[:,1]).tolist()
            # path planner
            sweeper = Sweep(use_theta=True, theta=world.theta)
            px, py = sweeper.planning(cell_x, cell_y, 0.14)
            inside_cell = np.array([px, py]).T
            # go to center
            cexit = centroid(v)
            path = np.concatenate((cellbord, centry, inside_cell, cexit), axis=0)
        # we've already been there, don't scan
        else:
            # if we're just passing through, pass through center
            path = np.concatenate((cellbord, centroid(v)), axis=0)

        cell_visited.append(v)
        # store data
        cell_path_data.append({
            'no_points' : path.shape[0],
            'length' : np.sum(np.linalg.norm(path[1:] - path[:1], axis=1)),
        })
        entire.append(path)
    entire = np.concatenate(entire, axis=0)

    for i, celldata in enumerate(cell_path_data):
        print('\tCell: {}'.format(i))
        for k, v  in celldata.items():
            print('\t\t{} = {}'.format(k,v))

    return entire

fig = plt.figure('Path Planner', figsize=(16, 9), facecolor='lightgrey')
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1 = world.world_chart(ax1, node_text=True, cell_text=True)
ax2 = world.world_chart(ax2)
path = get_path(world)
path_animator = PathAnimator()
path_animator.animate(path, world, fig, ax2, save=False, savepath='./planner_v1.mp4')
plt.show()