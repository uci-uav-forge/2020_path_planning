import json
import gmplot
import pyproj
from math import *
from geopy.distance import distance
from geopy import Point

with open('data.json') as json_file:
    data = json.load(json_file)

latitudes_wp = []  #lat waypoints
longitudes_wp = []  #long waypoints
altitudes_wp = []  #altitudes waypoints
longitudes_b = []  #lat boundaries
latitudes_b = []  #long boundaries
waypoints_x_y = []  #x/y converted waypoints
waypoints_x = []
waypoints_y = []
latitudes_ob = []  #lat obstacles
longitudes_ob = []  #long obstacles
heights_ob = []   #heights obstacles
radii_ob = []   #radii obstacles
obstacles_x_y = []  #x/y converted obstacles
obstacles_x = []
obstacles_y = []
finalWaypoints_x = []
finalWaypoints_y = []
finalWaypoints_long = []
finalWaypoints_lat = []
finalWaypoints_long_lat = []



def get_latitudes_wp():
    coord = data['waypoints']
    for coords in coord:
        lat = coords['latitude']
        latitudes_wp.append(lat)


def get_longitudes_wp():
    coord = data['waypoints']
    for coords in coord:
        long = coords['longitude']
        longitudes_wp.append(long)


def get_altitudes_wp():
    altitude = data['waypoints']
    for altitudes in altitude:
        alt = altitudes['altitude']
        altitudes_wp.append(alt)


def distance_between_points():
    get_latitudes_wp()
    get_longitudes_wp()
    p1 = Point(latitudes_wp[0], longitudes_wp[0])
    p2 = Point(latitudes_wp[1], longitudes_wp[1])
    print(p1)
    print(p2)

    result = distance(p1, p2).meters
    print(result)


def get_latitudes_b():
    bound = data['boundaryPoints']
    for bounds in bound:
        lat = bounds['latitude']
        latitudes_b.append(lat)


def get_longitudes_b():
    bound = data['boundaryPoints']
    for bounds in bound:
        long = bounds['longitude']
        longitudes_b.append(long)


def get_latitudes_ob():
    obst = data['stationaryObstacles']
    for obstacle in obst:
        lat = obstacle['latitude']
        latitudes_ob.append(lat)


def get_longitudes_ob():
    obst = data['stationaryObstacles']
    for obstacle in obst:
        long = obstacle['longitude']
        longitudes_ob.append(long)


def get_heights_ob():
    height = data['stationaryObstacles']
    for heights in height:
        h = heights['height']
        heights_ob.append(h)


def get_radii_ob():
    radius = data['stationaryObstacles']
    for radii in radius:
        r = radii['radius']
        radii_ob.append(r)


def mapping():
    gmap1 = gmplot.GoogleMapPlotter(38.145103, -76.427856, 16)
    get_latitudes_b()
    get_longitudes_b()
    get_longitudes_wp()
    get_latitudes_wp()
    get_altitudes_wp()
    get_latitudes_ob()
    get_longitudes_ob()
    get_heights_ob()
    get_radii_ob()


    # trace boundaries
    gmap1.scatter(latitudes_b, longitudes_b, 'white', size=30, marker=False)
    gmap1.polygon(latitudes_b, longitudes_b, color='cornflowerblue')

    # trace waypoints
    gmap1.scatter(latitudes_wp, longitudes_wp, 'green', size=20, marker=False)
    gmap1.plot(latitudes_wp, longitudes_wp, 'black', edge_width=2.5)

    # Pass the absolute path
    gmap1.draw("C:\\Users\\Adrien\\Desktop\\map11.html")


def joinWaypoints():
    waypoints = [[i, j] for i, j in zip(latitudes_wp, longitudes_wp)]
    print("Original waypoints: " + str(waypoints))


def joinObstacles():
    obstacles = [[i, j] for i, j in zip(latitudes_ob, longitudes_ob)]
    print("Original obstacles: " + str(obstacles))


def x_y_conversion_waypoints():
    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_gk4 = pyproj.Proj(init="epsg:3857")
    x, y = pyproj.transform(proj_wgs84, proj_gk4, longitudes_wp, latitudes_wp)
    #print(x)
    #print(y)
    for i, j in zip(y, x):
        waypoints_x_y.append([i, j])
        waypoints_x.append(i)
        waypoints_y.append(j)
    print("Converted waypoints: " + str(waypoints_x_y))


def x_y_conversion_obstacles():
    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_gk4 = pyproj.Proj(init="epsg:3857")
    x, y = pyproj.transform(proj_wgs84, proj_gk4, longitudes_ob, latitudes_ob)
    #print(x)
    #print(y)
    for i, j in zip(y, x):
        obstacles_x_y.append([i, j])
        obstacles_x.append(i)
        obstacles_y.append(j)
    print("Converted obstacles: " + str(obstacles_x_y))


def test():
    mapping()
    joinWaypoints()
    joinObstacles()
    x_y_conversion_waypoints()
    x_y_conversion_obstacles()

test()


class Point:
    def __init__(self, xval=0.0, yval=0.0):
        self.x = xval
        self.y = yval

    def PrintMe(self):
        print("x = " + str(self.x) + " y = " + str(self.y))


class Waypoint(Point):
    def __init__(self, aname, xval=0.0, yval=0.0, altitude=0):
        self.name = aname
        self.x = xval
        self.y = yval
        self.altitude = altitude

    def PrintMe(self):
        print(self.name + "   x = " + str(self.x) + "   y = " + str(self.y) + "   alt = " + str(self.altitude))


class Circle:
    def __init__(self, pt: Point, rad):  # pt:Point
        self.center = pt
        self.radius = rad

    def PrintMe(self):
        print("x=" + str(self.center.x) + " y=" + str(self.center.y) + " r=" + str(self.radius))


class Obstacle(Circle):
    def __init__(self, aname, pt: Point, rad, height):  # pt:Point
        self.name = aname
        self.center = pt
        self.radius = rad
        self.height = height

    def PrintMe(self):
        print(self.name + "   x = " + str(self.center.x) + "   y = " + str(self.center.y) + "   r = " + str(self.radius) + "   h = " + str(self.height))


class Line:
    def __init__(self, m, yint):
        self.slope = m
        self.yInt = yint

    def PrintMe(self):
        print("m=" + str(self.slope) + " b=" + str(self.yInt))


def GetLinePts(pt1, pt2):
    m = (pt2.y - pt1.y) / (pt2.x - pt1.x)
    b = pt1.y - (m * pt1.x)
    return (Line(m, b))


def GetLineSlope(pt, m):
    b = pt.y - (m * pt.x)
    return (Line(m, b))


# Solve Quadratic returns a list of solutions to the quadratic formula
def SolveQuadratic(a, b, c):
    d = b ** 2 - 4 * a * c  # discriminant
    if d < 0:
        return ([])
    elif d == 0:
        s1 = (-b) / (2 * a)
        return ([s1])
    else:
        s1 = (-b + sqrt(d)) / (2 * a)
        s2 = (-b - sqrt(d)) / (2 * a)
        return ([s1, s2])


def GetIntersectLineCirc(aline, circ):
    # Need to solve quadratic formula
    # First, define some shorthand
    m = aline.slope
    bi = aline.yInt
    x = circ.center.x
    y = circ.center.y
    r = circ.radius

    #    print("m=" + str(m) + " bi=" + str(bi) + " x=" + str(x) + " y=" + str(y) + " r=" + str(r)) # debug

    # Next, compute a, b, and c
    a = m ** 2 + 1
    b = 2 * (bi * m - y * m - x)
    c = x ** 2 + y ** 2 + bi ** 2 - r ** 2 - 2 * bi * y

    #    print("a=" + str(a) + " b=" + str(b) + " c=" + str(c)) # debug

    # Now, apply the quadratic formula to get the 2 solutions
    solns = SolveQuadratic(a, b, c)

    # Now generate the points and return them
    if len(solns) == 0:
        return ([])
    elif len(solns) == 1:
        return ([Point(solns[0], m * solns[0] + bi)])
    elif len(solns) == 2:
        return ([Point(solns[0], m * solns[0] + bi), Point(solns[1], m * solns[1] + bi)])
    else:
        return (-1)  # This should never happen


def Midpoint(pt1, pt2):
    return (Point((pt1.x + pt2.x) / 2, (pt1.y + pt2.y) / 2))


def GetAvoidPoints(w1, w2, o1):

    altitudeOne = w1.altitude
    altitudeTwo = w2.altitude
    altitudeObs = o1.height

    if altitudeOne > altitudeObs and altitudeTwo > altitudeObs:
        return []

    # Step 1: Find intersecting points between waypoint line and buffer circle
    wline = GetLinePts(w1, w2)

    #    print("Waypoint line") # debug
    #    wline.PrintMe() # debug

    SafetyMargin = o1.radius * 0.2
    bcirc = Circle(o1.center, o1.radius + SafetyMargin)

    #    print("Buffer circle") # debug
    #    bcirc.PrintMe() # debug

    iPts = GetIntersectLineCirc(wline, bcirc)
    # Important! Check that intersecting points not between the two waypoints.
    minx = min(w1.x, w2.x)
    maxx = max(w1.x, w2.x)
    miny = min(w1.y, w2.y)
    maxy = max(w1.y, w2.y)
    for pt in iPts:
        if pt.x > maxx or pt.x < minx or pt.y > maxy or pt.y < miny:
            return ([])

    #    print("Intersecting points") # debug
    #    PrintPointList(iPts) # debug

    # Step 2: Check how many intersections there are
    if len(iPts) > 2 or len(iPts) < 0:
        print("Error")
        return (-1)
    if len(iPts) == 0:
        return ([])
    if len(iPts) > 0:
        # Step 3: Compute the midpoint of the secant line
        if len(iPts) == 1:
            midPt = iPts[0]
        else:  # Two intersection points are found
            midPt = Midpoint(iPts[0], iPts[1])
        # Step 4: Get slope of perpendicular line
        if wline.slope != 0:
            pSlope = -1 / wline.slope
        else:
            pSlope = 1000.0
        # Step 5: Generate perpendicular line and double safety circle
        pline = GetLineSlope(midPt, pSlope)
        SafetyMargin = o1.radius * 0.2
        bcirc2 = Circle(o1.center, o1.radius + 2 * SafetyMargin)
        # Step 6: Find the intersection points and return them
        return (GetIntersectLineCirc(pline, bcirc2))


def checkSafe(pt, o):
    # check if the points in the range of the obstacle
    margin = o.radius * 0.2
    return not (o.center.x - o.radius - margin < pt.x and pt.x < o.center.x + o.radius + margin and \
                o.center.y - o.radius - margin < pt.y and pt.y < o.center.y + o.radius + margin)


def getSafePts(pts):
    safePts = []
    for pt in pts:
        if all(checkSafe(pt, o) for o in ObstacleList):
            safePts.append(pt)
    if len(safePts) == 0:
        # reduce the margin but for now just return pts
        return pts
    return safePts


def fixSingleSegment():
    global WaypointSeq
    prevPt = WaypointSeq[0]
    for i in range(1, len(WaypointSeq)):
        for ob in ObstacleList:
            aPts = GetAvoidPoints(prevPt, WaypointSeq[i], ob)
            if len(aPts) > 0:  # Crossing
                #check aPts position
                safePts = getSafePts(aPts)
                WaypointSeq.insert(i, safePts[0])
                return False
        prevPt = WaypointSeq[i]
    return True


WaypointSeq = []
ObstacleList = []


def testInitProblem():
    global WaypointSeq
    global ObstacleList
    for i in range(0, len(waypoints_x)):
        WaypointSeq += [Waypoint('waypoint_' + str(i+1), waypoints_y[i], waypoints_x[i], altitudes_wp[i])]
    for j in range(0, len(obstacles_x)):
        ObstacleList += [Obstacle('obstacle_' + str(j+1), Point(obstacles_y[j], obstacles_x[j]), radii_ob[j], heights_ob[j])]


def printWaypointSeq(wseq):
    print("\nWaypoint Sequence:")
    for w in wseq:
        w.PrintMe()


def jObstacleList(oseq):
    print("\nObstacle List:")
    for o1 in oseq:
        o1.PrintMe()


testInitProblem()
fixSingleSegment()
printWaypointSeq(WaypointSeq)
jObstacleList(ObstacleList)


def getFinalWaypoints(WaypointSeq):
    for i in WaypointSeq:
        finalWaypoints_x.append(i.x)
        finalWaypoints_y.append(i.y)


getFinalWaypoints(WaypointSeq)


def lat_long_conversion_waypoints():
    proj_wgs84 = pyproj.Proj(init="epsg:4326")
    proj_gk4 = pyproj.Proj(init="epsg:3857")
    long, lat = pyproj.transform(proj_gk4, proj_wgs84, finalWaypoints_x, finalWaypoints_y)

    for i, j in zip(long, lat):
        finalWaypoints_long.append(i)
        finalWaypoints_lat.append(j)
        finalWaypoints_long_lat.append([j, i])

    print("Converted final waypoints: " + str(finalWaypoints_long_lat))


lat_long_conversion_waypoints()


def writingToJson():
    dict = {}
    dict['waypoints'] = []
    for i, j in zip(finalWaypoints_lat, finalWaypoints_long):
        dict['waypoints'].append({
            'latitude': i,
            'longitude': j})

    with open('finalOutput.json', 'w') as json_file:
        json.dump(dict, json_file, indent=3)

writingToJson()
