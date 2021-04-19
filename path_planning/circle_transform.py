import numpy as np


def circ_transform(p, r, points = 10):
	'''
	turns circle with radius r and center point p into a concentric polygon of points many sides
	'''
	step_angle = 360/points*(np.pi/180) # radians
	edge_length = r*np.tan(step_angle/2)
	perimeter = []
	polygon = []
	poly = {}
	for x in range(points):
		polygon.append([p[0]+r*np.cos(x*step_angle)+edge_length*np.cos(x*step_angle+np.pi/2), p[1]+r*np.sin(x*step_angle)+edge_length*np.sin(x*step_angle+np.pi/2)])
	return polygon


if __name__ == '__main__':
	print(circ_transform([120, 120], 20, 5))