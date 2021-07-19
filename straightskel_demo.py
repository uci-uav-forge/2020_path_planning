from networkx.classes.function import frozen
from pathplanner import polygon, bdc, lawnmower
import matplotlib.pyplot as plt
import numpy as np
from scipy import spatial
import os

if __name__ == '__main__':
   seed = np.random.randint(0,1e5)
   print(seed)
   np.random.seed(87941)
   
   save=False
   bgcolor = 'silver'

   fig1, ax1 = plt.subplots()   
   ### Points
   points = polygon.beta_clusters(clusters=2, ppc=25)
   ax1.scatter(points[:,0], points[:,1], c='k', marker='.')
   ax1.set_title('1. 2D Distribution of Points')
   ax1.set_aspect('equal')

   fig2, ax2 = plt.subplots()
   points = polygon.remove_close_points(points)
   ax2.set_title('2. Remove Close Points')
   ax2.scatter(points[:,0], points[:,1], c='k', marker='.')
   ax2.set_aspect('equal')


   ### Polygon Creation
   G, dt, dt_orig = polygon.polygon(points, holes=2, removals=100)

   fig3, ax3 = plt.subplots()
   ax3.set_title('3. Delaunay Triangulation')
   spatial.delaunay_plot_2d(dt_orig, ax3)
   ax3.set_aspect('equal')

   fig4, ax4 = plt.subplots()
   ax4.set_title('4. Delaunay Triangulation After Removals')
   spatial.delaunay_plot_2d(dt, ax4)
   ax4.set_aspect('equal')

   fig5, ax5 = plt.subplots()
   ax5.set_title('5. Boundaries (CW) and Holes (CCW)')
   polygon.draw_G(G, ax=ax5, arrows=True, node_text=False)
   ax5.set_aspect('equal')

   ### BDC
   H, J, R, S = bdc.line_sweep(G, theta=bdc.degree2rad(-40))

   fig6, ax6 = plt.subplots()
   ax6.set_title('6. Rotated graph with Vertical Line Sweep')
   polygon.draw_G(H, ax=ax6, arrows=True, node_text=False)
   ax6.set_aspect('equal')
   
   fig7, ax7 = plt.subplots()
   ax7.set_title('7. Rotate Back to Original')
   polygon.draw_G(J, ax=ax7, arrows=True, node_text=False)
   ax7.set_aspect('equal')

   fig8, ax8 = plt.subplots()
   ax8.set_title('8. Reeb Graph of Joined Cells')
   polygon.draw_G(R, ax=ax8, posattr='centroid', ecolor='red')
   polygon.draw_G(J, ax=ax8, node_text=False, ecolor=bgcolor, draw_nodes=False)
   ax8.set_aspect('equal')

   fig9, ax9 = plt.subplots()
   ax9.set_title('9. Straight Skeleton over Each Cell')
   interior_nodes = [n for n in S.nodes if S.nodes[n]['interior'] == True]
   polygon.draw_G(S.subgraph(interior_nodes), ax=ax9, node_text=False)
   polygon.draw_G(J, ax=ax9, node_text=False, ecolor=bgcolor, draw_nodes=False)
   
   fig10, ax10 = plt.subplots()
   ax10.set_title('10. Joined Straight Skeletons')
   polygon.draw_G(S, ax=ax10, node_text=False, draw_nodes=False)
   polygon.draw_G(J, ax=ax10, node_text=False, ecolor=bgcolor, draw_nodes=False)

   fig11, ax11 = plt.subplots()
   ax11.set_title('11. Shortest Eulerian Over Straight Skeleton')
   polygon.draw_G(S, ax=ax11, arrows=True, node_text=False, ecolorattr='distance', ecmap='gist_earth')

   titles = [
      '1. 2D Distribution of Points',
      '2. Remove Close Points',
      '3. Delaunay Triangulation',
      '4. Delaunay Triangulation After Removals',
      '5. Boundaries (CW) and Holes (CCW)',
      '6. Rotated graph with Vertical Line Sweep',
      '7. Rotate Back to Original',
      '8. Reeb Graph of Joined Cells',
      '9. Straight Skeleton over Each Cell',
      '10. Joined Straight Skeletons',
      '11. Shortest Eulerian Over Straight Skeleton',
   ]
   if save:
      os.makedirs(name = './figures/', exist_ok=True)
      
      for fig, title in zip([fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9, fig10, fig11], titles):
         fig.set_size_inches((8,8))
         fig.savefig('./figures/' + title + '.png', dpi=200, facecolor='gray')
   else:
      plt.show()