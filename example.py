from path_planning import path_gen
from path_planning.gen_polygon import gen_poly

points, G = gen_poly()
pgen = path_gen.PathGenerator(points, G)
path = pgen.get_path(0.05)
pgen.show_path_animation()