"""
Grid based sweep planner

original author: Atsushi Sakai
modifications made by: Rick Meade

This code is used and modified under MIT license.
"""
import math, os, sys
from enum import IntEnum
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from grid_map_lib import GridMap
import matplotlib.pyplot as plt
from matplotlib.patches import PathPatch, Patch, Circle
import matplotlib

do_animation = True

class SweepSearcher:
    class SweepDirection(IntEnum):
        UP = 1
        DOWN = -1

    class MovingDirection(IntEnum):
        RIGHT = 1
        LEFT = -1

    def __init__(self, moving_direction, sweep_direction, x_inds_goal_y, goal_y):
        self.moving_direction = moving_direction
        self.sweep_direction = sweep_direction
        self.turing_window = []
        self.update_turning_window()
        self.x_indexes_goal_y = x_inds_goal_y
        self.goal_y = goal_y

    def move_target_grid(self, c_x_index, c_y_index, grid_map):
        n_x_index = self.moving_direction + c_x_index
        n_y_index = c_y_index
        # found safe grid
        if not grid_map.check_occupied_from_xy_index(n_x_index, n_y_index, occupied_val=0.5):
            return n_x_index, n_y_index
        else:  # occupied
            next_c_x_index, next_c_y_index = self.find_safe_turning_grid(
                c_x_index, c_y_index, grid_map)
            if (next_c_x_index is None) and (next_c_y_index is None):
                # moving backward
                next_c_x_index = -self.moving_direction + c_x_index
                next_c_y_index = c_y_index
                if grid_map.check_occupied_from_xy_index(next_c_x_index, next_c_y_index):
                    # moved backward, but the grid is occupied by obstacle
                    return None, None
            else:
                # keep moving until end
                while not grid_map.check_occupied_from_xy_index(
                        next_c_x_index + self.moving_direction,
                        next_c_y_index, occupied_val=0.5):
                    next_c_x_index += self.moving_direction
                self.swap_moving_direction()
            return next_c_x_index, next_c_y_index

    def find_safe_turning_grid(self, c_x_index, c_y_index, grid_map):
        for (d_x_ind, d_y_ind) in self.turing_window:
            next_x_ind = d_x_ind + c_x_index
            next_y_ind = d_y_ind + c_y_index
            # found safe grid
            if not grid_map.check_occupied_from_xy_index(next_x_ind, next_y_ind, occupied_val=0.5):
                return next_x_ind, next_y_ind
        return None, None

    def is_search_done(self, grid_map):
        for ix in self.x_indexes_goal_y:
            if not grid_map.check_occupied_from_xy_index(ix, self.goal_y, occupied_val=0.5):
                return False

        # all lower grid is occupied
        return True

    def update_turning_window(self):
        # turning window definition
        # robot can move grid based on it.
        self.turing_window = [
            (self.moving_direction, 0.0),
            (self.moving_direction, self.sweep_direction),
            (0, self.sweep_direction),
            (-self.moving_direction, self.sweep_direction),
        ]

    def swap_moving_direction(self):
        self.moving_direction *= -1
        self.update_turning_window()

    def search_start_grid(self, grid_map):
        x_inds = []
        y_ind = 0
        sweeper = Sweep()
        if self.sweep_direction == self.SweepDirection.DOWN:
            x_inds, y_ind = sweeper.search_free_grid_index_at_edge_y(grid_map, from_upper=True)
        elif self.sweep_direction == self.SweepDirection.UP:
            x_inds, y_ind = sweeper.search_free_grid_index_at_edge_y(grid_map, from_upper=False)
        if self.moving_direction == self.MovingDirection.RIGHT:
            if x_inds:
                return min(x_inds), y_ind
            else:
                return None
        elif self.moving_direction == self.MovingDirection.LEFT:
            if x_inds:
                return max(x_inds), y_ind
            else:
                return None
        raise ValueError("self.moving direction is invalid ")

class Sweep(object):
    def __init__(self, use_theta=False, theta=0):
        self.use_theta = use_theta
        self.theta = theta

    def find_sweep_direction_and_start_position(self, ox, oy):
        # sweep direction and position. 
        # position is the xy pair of the terminal node on cell border
        # direction is the dx/dy of the last (largest) cell direction
        max_dist = 0.0
        vec = [0.0, 0.0]
        sweep_start_pos = [0.0, 0.0]
        for i in range(len(ox) - 1):
            dx = ox[i + 1] - ox[i]
            dy = oy[i + 1] - oy[i]
            d = np.hypot(dx, dy)
            if d > max_dist:
                max_dist = d
                if self.use_theta:
                    vec = [np.sin(self.theta), np.cos(self.theta)]
                else:
                    vec = [dx, dy]
                sweep_start_pos = [ox[i], oy[i]]
        return vec, sweep_start_pos


    def convert_grid_coordinate(self, ox, oy, sweep_vec, sweep_start_position):
        tx = [ix - sweep_start_position[0] for ix in ox]
        ty = [iy - sweep_start_position[1] for iy in oy]
        th = math.atan2(sweep_vec[1], sweep_vec[0])
        rot = Rot.from_euler('z', th).as_matrix()[0:2, 0:2]
        converted_xy = np.stack([tx, ty]).T @ rot
        return converted_xy[:, 0], converted_xy[:, 1]


    def convert_global_coordinate(self, x, y, sweep_vec, sweep_start_position):
        th = math.atan2(sweep_vec[1], sweep_vec[0])
        rot = Rot.from_euler('z', -th).as_matrix()[0:2, 0:2]
        converted_xy = np.stack([x, y]).T @ rot
        rx = [ix + sweep_start_position[0] for ix in converted_xy[:, 0]]
        ry = [iy + sweep_start_position[1] for iy in converted_xy[:, 1]]
        return rx, ry

    def search_free_grid_index_at_edge_y(self, grid_map, from_upper=False):
        y_index = None
        x_indexes = []

        if from_upper:
            x_range = range(grid_map.height)[::-1]
            y_range = range(grid_map.width)[::-1]
        else:
            x_range = range(grid_map.height)
            y_range = range(grid_map.width)

        for iy in x_range:
            for ix in y_range:
                if not grid_map.check_occupied_from_xy_index(ix, iy):
                    y_index = iy
                    x_indexes.append(ix)
            if y_index:
                break

        return x_indexes, y_index

    def setup_grid_map(self, ox, oy, resolution, sweep_direction, offset_grid=10):
        width = math.ceil((max(ox) - min(ox)) / resolution) + offset_grid
        height = math.ceil((max(oy) - min(oy)) / resolution) + offset_grid
        center_x = (np.max(ox) + np.min(ox)) / 2.0
        center_y = (np.max(oy) + np.min(oy)) / 2.0

        grid_map = GridMap(width, height, resolution, center_x, center_y)
        grid_map.set_value_from_polygon(ox.tolist(), oy.tolist(), 1.0, inside=False)
        grid_map.expand_grid()

        x_inds_goal_y = []
        goal_y = 0
        if sweep_direction == SweepSearcher.SweepDirection.UP:
            x_inds_goal_y, goal_y = self.search_free_grid_index_at_edge_y(grid_map, from_upper=True)
        elif sweep_direction == SweepSearcher.SweepDirection.DOWN:
            x_inds_goal_y, goal_y = self.search_free_grid_index_at_edge_y(grid_map, from_upper=False)
        return grid_map, x_inds_goal_y, goal_y


    def sweep_path_search(self, sweep_searcher, grid_map, grid_search_animation=False, **kwargs):
        # search start grid
        idxs = sweep_searcher.search_start_grid(grid_map)
        if idxs:
            c_x_index, c_y_index = idxs
            if not grid_map.set_value_from_xy_index(c_x_index, c_y_index, 0.5):
                return [], []

            x, y = grid_map.calc_grid_central_xy_position_from_xy_index(c_x_index, c_y_index)
            px, py = [x], [y]
        else:
            px = []
            py = []
        if idxs:
            while True:
                c_x_index, c_y_index = sweep_searcher.move_target_grid(c_x_index, c_y_index, grid_map)
                if sweep_searcher.is_search_done(grid_map) or (
                        c_x_index is None or c_y_index is None):
                    break
                x, y = grid_map.calc_grid_central_xy_position_from_xy_index(
                    c_x_index, c_y_index)
                px.append(x)
                py.append(y)
                grid_map.set_value_from_xy_index(c_x_index, c_y_index, 0.5)
        return px, py

    def planning(self, ox, oy, resolution, moving_direction=SweepSearcher.MovingDirection.RIGHT, sweeping_direction=SweepSearcher.SweepDirection.UP):
        sweep_vec, sweep_start_position = self.find_sweep_direction_and_start_position(ox, oy)
        rox, roy = self.convert_grid_coordinate(ox, oy, sweep_vec, sweep_start_position)
        grid_map, x_inds_goal_y, goal_y = self.setup_grid_map(rox, roy, resolution, sweeping_direction)
        sweep_searcher = SweepSearcher(moving_direction, sweeping_direction, x_inds_goal_y, goal_y)
        px, py = self.sweep_path_search(sweep_searcher, grid_map)
        rx, ry = self.convert_global_coordinate(px, py, sweep_vec, sweep_start_position)
        return rx, ry

    def sweep(self, coords, width, flip_path = False):
        ox = []
        oy = []
        waypoints = []

        for x in coords:
            ox.append(x[0])
            oy.append(x[1])
        ox.append(coords[0][0])
        oy.append(coords[0][1])

        px, py = planning(ox, oy, width)
        for rx, ry in zip(px, py):
            waypoints.append([rx, ry])
        if flip_path:
            return waypoints.reverse()
        else:
            return waypoints

class PathAnimator(object):
    def __init__(self):
        pass

    def animate(self, path, world, fig, ax1, ax2, save=False, **kwargs):
        tour = []
        rg_node = []
        for cell, i in path:
            rg_node.extend([i] * cell.shape[0])
        path = np.concatenate([p[0] for p in path], axis=0)
        print(len(rg_node))
        print(path.shape)

        cmap = plt.get_cmap('viridis')
        plt.gcf()
        path_linecoll = matplotlib.collections.LineCollection( 
            path, 
            linestyle='-', 
            color='red',
            linewidth=0.5,
            capstyle='round',
            joinstyle='round',
            animated=True,
            )
        scale = (ax2.get_xlim()[1] - ax2.get_xlim()[0]) * 0.02
        
        rg_circle = matplotlib.patches.Ellipse(
            (1e5,1e5),
            width=scale,
            height=scale,
            animated=True,
            fill=True,
            hatch='+',
            zorder=4,
            linewidth=
            None,
            color='red'
        )


        path_linecoll.set_animated(True)
        frames = path.shape[0]
        def initframe():
            ax1.add_artist(path_linecoll)
            ax2.add_patch(rg_circle)
            return path_linecoll, rg_circle
            
        def drawframe(i):
            ax1.draw_artist(path_linecoll)
            ax2.draw_artist(rg_circle)
            pts = np.array(world.points)
            path_linecoll.set_segments( [path[:i,:]] )
            rg_center = world.Rg.nodes[rg_node[i]]['center']
            rg_circle.set_center((rg_center[0], rg_center[1]))
            return path_linecoll, rg_circle


        if save:
            # for some reason, calling plt.show() makes saving the animation possible... I don't get it but whatever
            plt.show()
            plt.close()
            ani = matplotlib.animation.FuncAnimation(fig, drawframe, frames=frames, init_func=initframe, interval=1, blit=True)
        else:
            ani = matplotlib.animation.FuncAnimation(fig, drawframe, frames=frames, init_func=initframe, interval=1, blit=True)
            
        if save:
            if 'savepath' not in kwargs:
                raise(Exception('Must include a save path, pass a str `path` in.'))
            ani.save(kwargs['savepath'], writer='ffmpeg', fps=30)
        plt.show()