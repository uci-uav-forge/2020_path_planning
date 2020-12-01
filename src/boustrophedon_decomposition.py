from tkinter import *
from math import *


# CODE IS DEPENDENT ON WHICH "DIRECTION" COORDINATES ARE ENTERED IN BOUNDARY COORDINATE TEST VARIABLE
# CROSS PRODUCT IS USED TO DETERMINE CONVEX VS CONCAVE VERTICES, THIS WILL BE POSITIVE OR NEGATIVE DEPENDING
# ON THE ORDER THE COORDINATES ARE INPUTTED IN THE LIST "BOUNDARY_COORDINATES"
# code was tested inputting points clockwise


class Line:
    '''
    straight line created by connecting [x1,y1] and [x2,y2]
    tkinter returns a unique identifier when create_line is called
    identifier can be used to delete the line later with canvas.delete(...)

    Parameters:
    x1: float
    y1: float
    x2: float
    y2: float
    identifier: int

    Returns:
    none
    '''

    def __init__(self, x1, y1, x2, y2, identifier=0):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.identifier = identifier

    def slope(self):
        '''
        uses rise/run formula to calculate slope
        returns string "undefined" and prints an error if the line is vertical

        Parameters:
        none

        Returns:
        none
        '''
        try:
            return (self.y2 - self.y1) / (self.x2 - self.x1)
        except ZeroDivisionError:
            #print("Divide by zero error")
            return "undefined"

    def intercept(self):
        '''
        uses y=mx+b -> b=y-mx to calculate y-intercept
        returns string "does not exist" and prints an error if the line is vertical
        '''
        try:
            return self.y2 - self.slope() * self.x2
        except TypeError:
            print("Y-intercept does not exist due to undefined slope")
            return "does not exist"


def intersects(l1, l2):
    '''
    Tests if two line segments intersect
    Also returns true if line segments touch but do not pass through

    Parameters:
    l1: Line object
    l2: Line object

    Returns:
    True if line segments intersect or touch each other
    False if line segments do not touch each other
    '''
    # check if domains share values
    if max(min(l1.x1, l1.x2), min(l2.x1, l2.x2)) > min(max(l1.x1, l1.x2), max(l2.x1, l2.x2)):
        return False

    # check if ranges share values
    if max(min(l1.y1, l1.y2), min(l2.y1, l2.y2)) > min(max(l1.y1, l1.y2), max(l2.y1, l2.y2)):
        return False

    # if slopes are equal, check if collinear
    # accounts for collinear vertical lines: l1.slope() == l2.slope() == "undefined"
    if l1.slope() == l2.slope():
        if l1.intercept() == l2.intercept():
            return True
        else:
            return False

    # calculate x-coordinate of intersection point, if l1 or l2 is vertical, sets x-coordinate as x-intersection
    if l1.slope() == "undefined" or l2.slope() == "undefined":
        if l1.slope() == "undefined" and min(l1.y1,l1.y2)<l2.slope()*l1.x1+l2.intercept()<max(l1.y1,l1.y2):
            x_intersection = l1.x1
        elif l2.slope() == "undefined" and min(l2.y1,l2.y2)<l1.slope()*l2.x1+l1.intercept()<max(l2.y1,l2.y2):
            x_intersection = l2.x1
        else:
            return False
    else:

        # round due to floating point arithmetic yielding incorrect results when the end of one line is equal to the
        # start of another line
        x_intersection = round((l2.intercept() - l1.intercept()) / (l1.slope() - l2.slope()), 5)

    # if x_intersection is within shared domain, return true
    if x_intersection < max(min(l1.x1, l1.x2), min(l2.x1, l2.x2)) or x_intersection > min(max(l1.x1, l1.x2), max(l2.x1, l2.x2)):
        return False
    else:
        return True


def init_gui(w, h):
    '''
    Initializes a canvas with specified width and height

    Parameters:
    w: int
        width in pixels of the canvas
    h: int
        height in pixels of the canvas

    Returns:
    canv: Canvas object
        canvas window with specified width and height
    '''
    master = Tk()
    canv = Canvas(master, width=w, height=h)
    canv.pack()
    return canv


def start_gui():
    '''
    Opens a canvas window

    Parameters:
    none

    Returns:
    none
    '''
    mainloop()


def draw_boundary(canv, coordinates):
    """
    Draws the outer boundary of the zone to be covered
    Writes coordinates in [x,y] on every point in the boundary

    Parameters:
    canv: Canvas object
        canvas on which to draw boundary lines
    coordinates: list of [int,int]
        list of coordinate pairs making up the boundary

    Returns:
    none
    """
    boundary = coordinates_to_lines(coordinates)
    for line in boundary:
        #canv.create_text(line.x1, line.y1, text="[" + str(line.x1) + ',' + str(line.y1) + "]")
        canv.create_line(line.x1, line.y1, line.x2, line.y2)


def coordinates_to_lines(coordinates):
    '''
    Takes a list of coordinates and returns a list of lines connecting the points in sequence
    In addition, connects the first point to the last point

    Parameters:
    coordinates: list of [int, int]
        list of coordinates to convert to lines [[x1,y1], [x2,y2], ... , [xn,yn]]

    Returns:
    lines: list of Line
        list of lines created by connecting coordinate points in order
    '''
    lines = []
    for index, coordinate in enumerate(coordinates):
        lines.append(Line(coordinates[index - 1][0], coordinates[index - 1][1], coordinates[index][0], coordinates[index][1]))
    return lines


# should change so that this function can create lines tangent to circles, not just lines intersecting with points
def boustrophedon_line_sweep(canv, angle, lines, orig_points, concave=False):
    '''
    Performs a boustrophedon decomposition with a line sweep perpendicular to the angle specified in the function call
    Draws lines that satisfy the requirements of boustrophedon decomposition
    Currently must be run on the boundary and every obstacle individually (improvements can be made here)
    Concave parameter indicates whether to keep lines that occur at concave vertices or convex vertices

    Parameters:
    canv: Canvas object
        canvas window on which to perform the line sweep
    angle: Float
        angle in degrees counterclockwise from +x-axis
    orig_points: list of [int, int]
        list of coordinates making up the boundary or the obstacle
    concave: boolean
        True if performing this function on a boundary
        False if performing this function on an obstacle

    Returns:
    none
    '''
    slope = -tan(angle * pi / 180)
    points = orig_points[:]
    points.append(points[0])
    if concave:
        indices = extract_concave_vertices(orig_points)
    else:
        indices = extract_convex_vertices(orig_points)
        print(str(indices))
    if slope == 0:
        for num in indices:
            y1 = points[num][1]
            y2 = points[num][1]
            x1 = 0
            x2 = canvas_width
            if check_line_inside_boundary(points[num - 1], points[num], points[num + 1], x1, y1):
                canv.create_line(x1, y1, x2, y2)
    else:
        for num in indices:
            y1 = 0
            y2 = canvas_height
            x1 = (0 - points[num][1] + slope * points[num][0]) / slope
            x2 = (y2 - points[num][1] + slope * points[num][0]) / slope
            print("Coordinate: " + str(points[num]) + "P1: " + str(points[num-1]) + "P2: " + str(points[num+1]))
            if check_line_inside_boundary(points[num - 1], points[num], points[num + 1], x1, y1):
                if(num==0) and not check_line_inside_boundary(points[len(points)-2],points[0], points[1], x1, y1):
                    continue
                long_line = Line(x1, y1, x2, y2)
                x_intercepts = []
                for l in lines:
                    if intersects(long_line, l):
                        if long_line.slope() == "undefined" or l.slope() == "undefined":
                            if long_line.slope() == "undefined":
                                x_intersection = long_line.x1
                            if l.slope() == "undefined":
                                x_intersection = l.x1
                        else:
                            x_intersection = round((l.intercept() - long_line.intercept()) / (long_line.slope() - l.slope()), 5)
                        if x_intersection != points[num][0]:
                            x_intercepts.append(x_intersection)
                x_lower = min(x1, x2)
                x_upper = max(x1, x2)
                for x_int in x_intercepts:
                    if x_int < points[num][0]:
                        if x_int > x_lower:
                            x_lower = x_int
                    if x_int > points[num][0]:
                        if x_int < x_upper:
                            x_upper = x_int
                y1 = slope*(x_lower-points[num][0])+points[num][1]
                y2 = slope*(x_upper-points[num][0])+points[num][1]
                canv.create_line(x_lower, y1, x_upper, y2)


# cross product with p1 = [x1,y1]
def cross_product(p1, p2, p3):
    '''
    Takes the cross product between vectors <v21> and <v23>

    Parameters:
    p1: list of int, len(list)=2
        first set of coordinates [x1,y1]
    p1: list of int, len(list)=2
        second set of coordinates [x2,y2]
    p1: list of int, len(list)=2
        third set of coordinates [x3,y3]

    Returns:
    int
        computed cross product
        most use cases within this file look at sign of cross-product, not necessarily the numerical value
    '''
    dx1 = p1[0] - p2[0]
    dy1 = p1[1] - p2[1]
    dx2 = p3[0] - p2[0]
    dy2 = p3[1] - p2[1]
    return dx1 * dy2 - dy1 * dx2


def extract_concave_vertices(orig_coordinates):
    '''
    Takes a list of coordinates, assumes they connect in the sequence they are listed to form a polygon, and returns
    the indices of coordinates that are concave vertices of that polygon

    Parameters:
    orig_coordinates: list of [int, int]
        list of coordinates making up the polygon

    Returns:
    list of int
        list containing the index of every coordinate that is a concave vertex
    '''
    convex_vertices = []
    coordinates = orig_coordinates[:]
    if cross_product(coordinates[-1], coordinates[0], coordinates[1]) > 0:
        convex_vertices.append(0)
    coordinates.append(coordinates[0])
    for index, point in enumerate(coordinates):
        if cross_product(coordinates[index-2], coordinates[index-1], coordinates[index]) > 0:
            convex_vertices.append(index-1)
    return convex_vertices


def extract_convex_vertices(orig_coordinates):
    '''
    Takes a list of coordinates, assumes they connect in the sequence they are listed to form a polygon, and returns
    the indices of coordinates that are convex vertices of that polygon

    Parameters:
    orig_coordinates: list of [int, int]
        list of coordinates making up the polygon

    Returns:
    list of int
        list containing the index of every coordinate that is a convex vertex
    '''
    concave_vertices = []
    coordinates = orig_coordinates[:]
    if cross_product(coordinates[-1], coordinates[0], coordinates[1]) < 0:
        concave_vertices.append(0)
    coordinates.append(coordinates[0])
    for index, point in enumerate(coordinates):
        if cross_product(coordinates[index-2], coordinates[index-1], coordinates[index]) < 0:
            concave_vertices.append(index-1)
    return concave_vertices


def check_line_inside_boundary(p1, p2, p3, x1, y1):
    '''
    Uses cross product to check if a line is locally contained within the boundary or outside of an obstacle
    Checks if line passes between <v21> and <v23> or if it passes 'tangent' to p2

    Parameters:
    p1: list of int, len(list)=2
        first set of coordinates [x1,y1]
    p1: list of int, len(list)=2
        second set of coordinates [x2,y2]
    p1: list of int, len(list)=2
        third set of coordinates [x3,y3]
    x1: int
        select any point on the line [x1,y1] OTHER than [p2[0],p2[1]]
    y1: int
        select any point on the line [x1,y1] OTHER than [p2[0],p2[1]]

    Returns:
    boolean
        True if one cross-product is positive and one is negative
        False if both cross-products have the same sign
    '''
    if (cross_product(p1, p2, [x1, y1]) > 0) ^ (cross_product([x1, y1], p2, p3) > 0):
        return True
    else:
        return False


############################
# INITIALIZE TEST CONDITIONS
############################
boundary_coordinates = [[150, 280], [200, 200], [450, 260], [470, 220], [400, 100], [585, 82.5], [770, 65], [830, 300],
                        [760, 500], [520, 520], [450, 440], [350, 400], [210, 420]]
test_obstacle_coordinates = [[600, 200], [675, 275], [600, 350], [525, 275]]
test_obstacle_coordinates_2 = [[650, 120], [700, 120], [700, 170], [650, 170]]
all_lines = []
for line in coordinates_to_lines(boundary_coordinates,) + coordinates_to_lines(test_obstacle_coordinates) + coordinates_to_lines(test_obstacle_coordinates_2):
    all_lines.append(line)


canvas_width = 1280
canvas_height = 720
theta = 57

canvas = init_gui(canvas_width, canvas_height)
draw_boundary(canvas, boundary_coordinates)
draw_boundary(canvas, test_obstacle_coordinates)
draw_boundary(canvas, test_obstacle_coordinates_2)
boustrophedon_line_sweep(canvas, theta, all_lines, boundary_coordinates, concave=True)
boustrophedon_line_sweep(canvas, theta, all_lines, test_obstacle_coordinates, concave=False)
boustrophedon_line_sweep(canvas, theta, all_lines, test_obstacle_coordinates_2, concave=False)
start_gui()
