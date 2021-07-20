from pathplanner import bdc, polygon
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

seed = np.random.randint(1e7)
print(seed)
np.random.seed(seed)

points = np.random.beta(4.0,4.0, (30, 2))
G, _, _ = polygon.polygon(points, holes=2, removals=50)
theta = bdc.degree2rad(70)
H, J, R, S = bdc.line_sweep(G, theta=theta)
P = bdc.discretize_entire(J, R, 0.007)

fig, ax = plt.subplots()
lines = []
for e1, e2 in nx.eulerian_path(nx.eulerize(P)):
   lines.append(P.nodes[e1]['points'])
lines = np.array(lines)

path_line = matplotlib.lines.Line2D(lines[0:1,0], lines[0:1,1],animated=True,antialiased=True)

def init():
   ax.add_artist(path_line)
   return path_line,

def animate(i):
    ax.draw_artist(path_line)
    path_line.set_xdata([lines[:i,0]])
    path_line.set_ydata([lines[:i,1]])
    return path_line,
ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=list(range(lines.shape[0])), interval=5, blit=True)



polygon.draw_G(J, ax=ax, node_text=False, draw_nodes=False, ecolor='silver')


plt.show()