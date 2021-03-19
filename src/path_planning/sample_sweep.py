'''
Sample sweep a random polygon
'''
from bcd import World
import bcd
from gen_polygon import ConvPolygon
from sub_sweep import PathAnimator, Sweep
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import copy

poly = ConvPolygon(points=(4, 14, 1, 1), jaggedness=2, holes=2)
angles = np.linspace(0, np.pi, 180)
worlds = []
print('building worlds...')
for angle in angles:
    print('{}'.format(str(round(angle, 2)).ljust(7)), end='', flush=True)
    print(angle * 180/np.pi, end='\r', flush=True)
    w = World(poly=poly, theta=angle)
    worlds.append( w )
    print(w.scalar_qualities['avg_cell_aspect'])
print('done!')


q_function = lambda x: x.scalar_qualities['min_cell_width']
cwsorted = sorted([ w for w in worlds], key=q_function)

# metric highest
best = cwsorted[-1]
# metric lowest
worst = cwsorted[0]
world = best


print('World with angle={} chosen.'.format(world.theta))
allpx, allpy = [], []

allcell = []
ordered_cell_idx = nx.dfs_preorder_nodes(world.Rg, min([n for n in world.Rg.nodes], key=lambda n: world.Rg.nodes[n]['center'][0]))


cell_visited = []

def centroid(ci):
    cell = world.cells[ci]
    return np.sum(world.points[list(cell)], axis=0) / len(list(cell))

celldata = {}

def get_path(world):
    cellidx = list(nx.edge_dfs(world.Rg))
    print(cellidx)
    cell_path_data = []
    entire = []
    for i, edge in enumerate(cellidx):
        u, v = edge
        # common edges between u and v
        commonA, commonB = tuple(world.Rg[u][v]['common'])
        # visit cell border
        cellbord = world.points[commonB] + 0.5 * (world.points[commonA] - world.points[commonB])
        # go to center
        centry = np.linspace(cellbord, centroid(v), 12)
        if v not in cell_visited:            
            cell_pts = np.array(world.points[world.cells_list[v]])
            cell_x = np.squeeze(cell_pts[:,0]).tolist()
            cell_y = np.squeeze(cell_pts[:,1]).tolist()
            # path planner
            sweeper = Sweep(use_theta=False, theta=world.theta)
            px, py = sweeper.planning(cell_x, cell_y, 0.05)
            inside_cell = np.array([px, py]).T
            # go to center
            if inside_cell.shape[0] == 0:
                cexit = centroid(v)
            else:
                cexit = np.linspace(inside_cell[-1], centroid(v), 12)
            path = np.concatenate((centry, inside_cell, cexit), axis=0)
        else:
            # if we're just passing through, pass through center
            path = centry = np.linspace(cellbord, centroid(v), 12)
        cell_visited.append(v)
        # store data
        cell_path_data.append({
            'no_points' : path.shape[0],
            'length' : np.sum(np.linalg.norm(path[1:] - path[:1], axis=1)),
        })
        entire.append((path, v))

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

ax1 = bcd.draw_graph(
    ax1, 
    world.G,
    world.points,
    node_marker='.',
    node_text=False, 
    chart_name='World',
    cell_text=True)

cc = np.squeeze(np.array([c for c in world.cell_centroids.values()]))

ax2 = bcd.draw_graph(
    ax2,
    world.Rg,
    np.array(cc)
)


ax2 = bcd.draw_graph(
    ax2, 
    world.G, 
    world.points,
    node_color='lightgrey',
    edge_colors=('silver','silver','silver','silver'),
    chart_name='Reeb Graph'
    )


path = get_path(world)
path_animator = PathAnimator()
path_animator.animate(path, world, fig, ax1, ax2, save=True, savepath='./planner_v1.mp4')
plt.show()
