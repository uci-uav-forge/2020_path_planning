from pathplanner import polygon
import matplotlib.pyplot as plt

if __name__ == "__main__":
   fig1, ax1 = plt.subplots()
   ax1.set_title('n=10')
   radpoints, G = polygon.stupid_spiky_polygon(1, 5)
   polygon.draw_G(G, ax=ax1)


   fig2, ax2 = plt.subplots()
   ax2.set_title('n=20')
   radpoints2, G2 = polygon.stupid_spiky_polygon(1, 5, n=20)
   polygon.draw_G(G2, ax=ax2)


   fig3, ax3 = plt.subplots()
   ax3.set_title('n=80')
   radpoints3, G3 = polygon.stupid_spiky_polygon(1, 5, n=80)
   polygon.draw_G(G3, ax3)

   titles = ['n=10', 'n=20', 'n=80']

   for fig, title in zip([fig1, fig2, fig3], titles):
      fig.set_size_inches((8,8))
      fig.savefig('./figures/' + title + '.png', dpi=200, facecolor='gray')

   plt.show()