from pathplanner import bdc, polygon
import numpy as np
import matplotlib.pyplot as plt

points = np.random.beta(4.0,4.0, (30, 2))
G, _, _ = polygon.polygon(points, holes=2, removals=50)

theta = bdc.degree2rad(70)

H, J, R, S = bdc.line_sweep(G, theta=theta)


fig, ax = plt.subplots(ncols=2)



polygon.draw_G(J, ax=ax[0])
polygon.draw_G(J, ax=ax[1], ecolor='silver', node_text=False, draw_nodes=False)

M = bdc.discretize(J, R, 0.012, theta=theta)
pts = np.array([M.nodes[n]['points'] for n in M])
polygon.draw_G(M, ax[1], node_text=False)


plt.show()