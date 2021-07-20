from pathplanner import bdc, polygon, astar
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib

seed = np.random.randint(1e7)
print(seed)
np.random.seed(seed)

points = polygon.beta_clusters(2, ppc=220)

G, _, _ = polygon.polygon(points, holes=1, removals=150)
theta = bdc.degree2rad(70)
H, J, R, S = bdc.line_sweep(G, theta=theta)
P = bdc.discretize_entire(J, R, 0.01)
sp = sorted([n for n in P], key=lambda n: P.nodes[n]['points'][0])
path = astar.astar(P, sp[0], sp[-1])

fig, ax = plt.subplots()
lines = []
for n in path:
   lines.append(P.nodes[n]['points'])
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
ani = matplotlib.animation.FuncAnimation(fig, animate, init_func=init, frames=list(range(lines.shape[0])), interval=30, blit=True)
plt.show()
polygon.draw_G(J, ax=ax, node_text=False, draw_nodes=False, ecolor='silver')

ani.save('astar.mp4')



plt.show()