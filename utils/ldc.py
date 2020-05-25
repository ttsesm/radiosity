import numpy as np
import sys
import os
import multiprocessing
from scipy.spatial.transform import Rotation as R

import copy

import pyvista as pv

import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Qt5Agg')

class LightDistributionCurve(object):
    """docstring for Isocell"""
    #TODO: Modify and update this accordingly

    def __init__(self, ldc=np.array([])):
        self.properties = {}

        if ldc.any():
            self.properties['ldc'] = ldc
        else:
            # Use ldc sample
            self.properties['ldc'] = np.array([  [426.0060,  426.0060,  426.0060,  426.0060,  426.0060,  426.0060,  426.0060],
                                   [424.7540,  425.0980,  425.5810,  425.9940,  425.6490,  425.1670,  424.7540],
                                   [421.8600,  422.2040,  422.7550,  423.1690,  422.6860,  422.2040,  421.8600],
                                   [415.5200,  416.0020,  416.1400,  416.8980,  416.6910,  416.2780,  415.7960],
                                   [408.6290,  406.9060,  407.0440,  408.0090,  409.0420,  407.3890,  406.5620],
                                   [394.1580,  394.2960,  394.5030,  395.1920,  395.0540,  394.5720,  394.5720],
                                   [374.5880,  375.8290,  376.9310,  376.6550,  374.6570,  376.3800,  377.4820],
                                   [349.7810,  351.3660,  352.8130,  351.9170,  350.7460,  352.1930,  353.9150],
                                   [317.8070,  316.2910,  318.2210,  316.2910,  316.8420,  317.1870,  319.8750],
                                   [267.2280,  269.4330,  264.4720,  267.9170,  267.9170,  269.6400,  266.1260],
                                   [200.4280,  162.1010,  174.1260,  163.2380,  199.5320,  163.8650,  174.8770],
                                   [111.4940,  118.7160,  150.0280,  118.3780,  112.6450,  116.1870,  151.6540],
                                    [73.5810,   78.4390,   99.5180,   80.3960,   75.7450,   78.4250,  100.4280],
                                    [49.7660,   69.2120,   54.2240,   71.2930,   52.1980,   71.8860,   56.8360],
                                    [35.5290,   49.5180,   46.4930,   48.3810,   35.8390,   47.2780,   48.2220],
                                    [34.3720,   37.9410,   35.3290,   38.5750,   33.9930,   39.9950,   35.4050],
                                    [24.1730,   24.9380,   28.9690,   25.8750,   24.7800,   24.6350,   28.5140],
                                    [14.3050,   15.9870,   19.6670,   16.4830,   15.1460,   15.9450,   19.7770],
                                     [6.0920,    6.5390,    7.0420,    7.2350,    6.9940,    6.8220,    6.6840],
                                     [4.7550,    4.9550,    5.2780,    5.2160,    5.2090,    5.4780,    5.6640],
                                     [3.8590,    3.8520,    3.8930,    3.9000,    3.8870,    4.3210,    4.4380],
                                     [3.1150,    3.1420,    2.9350,    2.8530,    3.0870,    3.4660,    3.5970],
                                     [2.8670,    2.6670,    2.4740,    2.3500,    2.5700,    2.9290,    3.1700],
                                     [2.4120,    2.4050,    2.3360,    2.0330,    2.3500,    2.6050,    2.7840],
                                     [1.6540,    1.6670,    1.9090,    1.8880,    1.7300,    1.5920,    1.6810],
                                     [1.1300,    1.1990,    1.2680,    1.4880,    1.2270,    1.1580,    1.1300],
                                     [1.0470,    1.0400,    1.0400,    1.1440,    1.0750,    1.0540,    1.0470],
                                     [1.1300,    1.1640,    1.2060,    1.1850,    1.2130,    1.1850,    1.1710],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0],
                                          [0,         0,         0,         0,         0,         0,         0]])


        self.properties['normalized_ldc'] = self.__normalize_ldc(self.properties['ldc'])
        # self.normalized_ldc = self.__normalize_ldc()
        # self.symmetric_ldc = np.pad(self.properties['ldc'], (0, self.propertis['ldc'].shape(1)), 'symmetric')
        # self.properties['symmetric_ldc'] = np.pad(self.properties['ldc'], ((0,0), (0,self.properties['ldc'].shape[1])), 'symmetric')
        self.properties['symmetric_ldc'] = self.__get_symmetric_ldc(self.properties['ldc'], axis=1)
        self.properties['normalized_symmetric_ldc'] = self.__normalize_ldc(self.properties['symmetric_ldc'])

        self.plot3D()

        print()

    def __plot2D(self, ldc):

        print()

    def plot2D(self, color='red', legend='LDC', inline=True, type='default'): # type could be normalized/symmetric or default

        try:
            if type not in ('default', 'normalized', 'symmetric'):
                raise ValueError('\'{}\' is not a valid type.'.format(type))
        except ValueError:
            exit('\'{}\' is not a valid type.'.format(type))

        if type == 'normalized':
            # ldc = self.normalized_ldc
            ldc = self.properties['normalized_ldc']
        elif type == 'symmetric':
            # ldc = self.symmetric_ldc
            ldc = self.properties['symmetric_ldc']
        else:
            # ldc = self.ldc
            ldc = self.properties['ldc']

        # step = np.ceil(180/ldc.shape[0])
        middle_index = int(np.ceil(ldc.shape[1] // 2))
        # self.ldc[:, self.ldc.shape[1] // 2]
        # theta = np.arange(0, 180+1, step)
        theta = np.linspace(0, 180, ldc.shape[0])

        ax = plt.subplot(111, projection='polar')
        ax.plot(np.radians(theta), ldc[:,middle_index], color=color, label=legend)
        ax.plot(np.radians(-theta), ldc[:, middle_index], color=color)
        ax.set_theta_zero_location("S")
        ax.set_rlabel_position(0)
        # fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
        ax.set_xticks(np.pi / 180. * np.linspace(180, -180, 8, endpoint=False))
        ax.set_xticklabels(list(abs(np.linspace(180, -180, 8, endpoint=False))))
        ax.set_thetalim(-np.pi, np.pi)

        plt.legend()
        if inline:
            plt.show()

    def plot3D(self, pos=np.array([0, 0, 0]), rot=np.array([]), color='blue', legend='3D LDC', inline=True, type='default'): # type could be normalized/symmetric or default

        try:
            if type not in ('default', 'normalized', 'symmetric'):
                raise ValueError('\'{}\' is not a valid type.'.format(type))
        except ValueError:
            exit('\'{}\' is not a valid type.'.format(type))

        if type == 'normalized':
            # ldc = self.normalized_ldc
            ldc = self.properties['normalized_ldc']
        elif type == 'symmetric':
            # ldc = self.symmetric_ldc
            ldc = self.properties['symmetric_ldc']
        else:
            # ldc = self.ldc
            ldc = self.properties['ldc']

        # ldc_planes = np.pad(self.properties['ldc'], ((0,self.properties['ldc'].shape[0]),(0,0)), 'symmetric')
        # middle_index = int(np.ceil(ldc_planes.shape[0] // 2))
        # ldc_planes = np.delete(ldc_planes, middle_index, 0)
        ldc_planes = self.__get_symmetric_ldc(self.properties['ldc'])

        # step = np.ceil(360 / ldc_planes.shape[0])
        # angles around x-axis, need to turn by 90 degree right pol2cart output
        # anglesX = np.arange(0, 360 + 1, step) / 180 * np.pi + np.pi / 2
        anglesX = np.linspace(0, 360, ldc_planes.shape[0]) / 180 * np.pi + np.pi / 2

        # angles around z-axis
        anglesZ = np.linspace(0, 90, ldc_planes.shape[1])

        p1, p2 = self.pol2cart(anglesX.reshape(-1,1), ldc_planes)

        X = np.ones(np.shape(p1)) * pos[0];
        Y = p1 + pos[1];
        Z = p2 + pos[2];

        # points = np.column_stack([X.flatten(order='F'), Y.flatten(order='F'), Z.flatten(order='F')])
        # poly = pv.PolyData(points)
        # poly.rotate_x()
        #
        plotter = pv.BackgroundPlotter()
        # # plotter = pv.Plotter()
        # plotter.add_mesh(poly, color="b", point_size=1)
        # lines = pv.lines_from_points(points)
        # lines.rotate_x()
        # plotter.add_mesh(lines, color="r")
        # plotter.add_axes()
        # plotter.show()

        for i in range(ldc_planes.shape[1]):
            points = np.column_stack([X[:,i], Y[:,i], Z[:,i]])
            poly = pv.PolyData(copy.deepcopy(points))
            poly.rotate_z(anglesZ[i])
            plotter.add_mesh(poly, color="b", point_size=2)

            # lines = pv.lines_from_points(poly.points)
            # # lines.rotate_z(anglesZ[i])
            # plotter.add_mesh(lines, color="r")

            poly2 = pv.PolyData(copy.deepcopy(points))
            poly2.rotate_z(-anglesZ[i])
            plotter.add_mesh(poly2, color="r", point_size=2)
            # lines = pv.lines_from_points(poly.points)
            # # lines.rotate_z(180+anglesZ[i])
            # plotter.add_mesh(lines, color="r")

        # for i in range(ldc_planes.shape[1]):
        #     points = np.column_stack([X[:,i], Y[:,i], Z[:,i]])
        #     # poly = pv.PolyData(points)
        #     # poly.rotate_z(anglesZ[i])
        #     # plotter.add_mesh(poly, color="b", point_size=3)
        #     #
        #     # # lines = pv.lines_from_points(poly.points)
        #     # # # lines.rotate_z(anglesZ[i])
        #     # # plotter.add_mesh(lines, color="r")
        #
        #     poly = pv.PolyData(points)
        #     poly.rotate_z(-anglesZ[i])
        #     plotter.add_mesh(poly, color="r", point_size=2)
        #     # lines = pv.lines_from_points(poly.points)
        #     # # lines.rotate_z(180+anglesZ[i])
        #     # plotter.add_mesh(lines, color="r")

        plotter.add_axes()
        plotter.show()

        print()


    def __normalize_ldc(self, ldc, low_bound=0, upper_bound=1):
        # Normalize to [0, 1] or any other bounds
        m = np.amin(ldc)
        range = np.amax(ldc) - m
        normalized_ldc = (ldc - m) / range

        # Then scale to[x, y]
        range2 = upper_bound - low_bound
        normalized_ldc = (normalized_ldc * range2) + low_bound

        return  normalized_ldc

    def __get_symmetric_ldc(self, ldc, axis=0):

        if axis == 0:
            ldc = np.pad(ldc, ((0, ldc.shape[axis]), (0, 0)), 'symmetric')
            # middle_index = int(np.ceil(ldc.shape[0] // 2))
            # ldc = np.delete(ldc, middle_index, 0)
        else:
            ldc = np.pad(ldc, ((0, 0), (0, ldc.shape[axis])), 'symmetric')
            # middle_index = int(np.ceil(ldc.shape[1] // 2))

        middle_index = int(np.ceil(ldc.shape[axis] // 2))
        ldc = np.delete(ldc, middle_index, axis)

        return ldc

    def pol2cart(self, theta, rho):
        x = rho * np.cos(theta)
        y = rho * np.sin(theta)
        return x, y
