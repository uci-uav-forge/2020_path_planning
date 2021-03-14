import numpy as np
from bcd import ConvPolygon, World
import matplotlib.pyplot as plt

# create a sample polygon
NO_CLUSTERS = 5
PTS_PER_CLUSTER = 15
CLUSTER_VAR = 1
CLUSTER_DIST = 4
poly = ConvPolygon(points=(NO_CLUSTERS, PTS_PER_CLUSTER, CLUSTER_VAR, CLUSTER_DIST), jaggedness=9, holes=3)
# define our quality metrics here. They must be keys of World.scalar_qualities.
metrics = [
    'avg_cell_width',
    'min_cell_width',
    'avg_cell_aspect',
    'min_cell_aspect',
    'no_cells',
    'estrada',
    'weiner',
    'degrees',
    'area_variance'
]
# build the list of worlds, one at each angle
angles = np.linspace(0, np.pi, 180)
worlds = []
print('building worlds...')
for angle in angles:
    print(angle * 180/np.pi, end='\r', flush=True)    
    worlds.append( (World(poly=poly, theta=angle), round(angle * 180/np.pi, 2) ) )
print('done!')

# create a list of lambdas to test those metrics on the list of worlds
qualityfns = []
for m in metrics:
    qualityfns.append((
        lambda w: w[1][0].scalar_qualities[m],
        m
    ))

# build a list of best and worst worlds assessed on each quality metric in `qualityfns`
best_worst = []
print('Assessing worlds...')
for qualfn, name in qualityfns:
    print(name + str('...'))
    # clear worlds list
    cwsorted = []
    # sort worlds list by metric
    cwsorted = sorted([ (idx, world) for (idx, world) in enumerate(worlds) ], key=qualfn)
    # best one = last in list (metric highest)
    best = cwsorted[-1][1]
    # worst one = first in list (metric lowest)
    worst = cwsorted[0][1]
    best_worst.append( (best, worst, name) )
    print('done!')
    print('best: world {}, worst: world {}'.format( cwsorted[-1][0], cwsorted[0][0] ) )

# plot every world
for i, (b, w, name) in enumerate(best_worst):
    fig, ax = plt.subplots(nrows=1, ncols=2, sharex=True, sharey=True)
    ax[0].set_title('Worst by Metric ' + str(name))
    ax[1].set_title('Best by Metric ' + str(name))
    b[0].chart_reebgraph(ax=ax[0], draw_chart_behind=True)
    w[0].chart_reebgraph(ax=ax[1], draw_chart_behind=True)
    captionb = name + '=' + str(round(b[0].scalar_qualities[name], 2)) + '; angle=' + str(b[1])
    captionw = name + '=' + str(round(w[0].scalar_qualities[name], 2)) + '; angle=' + str(w[1])
    ax[0].text(0.1, -0.1, captionb, transform=ax[0].transAxes)
    ax[1].text(0.1, -0.1, captionw, transform=ax[1].transAxes)
plt.show()