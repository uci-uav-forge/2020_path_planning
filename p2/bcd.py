from p2 import polygon


if __name__ == '__main__':
    poly = polygon.polygon(polygon.cluster_points(), holes=4, removals=40)
