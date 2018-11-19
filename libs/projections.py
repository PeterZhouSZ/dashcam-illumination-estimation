import numpy as np
from math import *

def sphere2world(theta, phi):
    x = cos(radians(phi)) * sin(radians(theta))
    y = sin(radians(phi))
    z = cos(radians(phi)) * cos(radians(theta))
    return np.asarray([x, y, z])

def sphere2uv(theta, phi, width=128, height=64):
    x = theta/180
    y = phi/90
    i = floor((x + 1.0) * width/2)
    j = floor((1 - (y + 1.0)/2)*height)
    return np.array([i,j])

def world2sphere(x, y, z):
    u = (1 / np.pi) * np.arctan2(x, z)
    v = (1 / np.pi) * np.arcsin(y)
    u = u / 2
    return np.array([u * 360.0, v * 180.0])

'''
    Total 256 bins in hemisphere,
    8x32: 8 bins for 90 degrees elevation, 32 bins for -180 to 180 (360 degrees) azimuth
'''
def bin2Sphere(i):
    phi = (floor(i/32)) * (90/8.0) + (90/16.0)
    theta = ((i+1) - floor(i/32) * 32 - 1) * (360.0/32.0) + (360.0/64.0) - 180.0
    return np.array([theta, phi])

def sphere2Bin(theta, phi):
    return int(round((phi - (90/16.0))/(90/8.0))*32) + floor((theta+180-(360/64.0))/(360/32.0))

def getRotationX(angle):
    return np.asarray([[1, 0, 0],
                       [0, cos(radians(angle)), sin(radians(angle))],
                       [0, -sin(radians(angle)), cos(radians(angle))]])

def getRotationY(angle):
    return np.asarray([[cos(radians(angle)), 0, sin(radians(angle))],
                       [0, 1, 0],
                       [-sin(radians(angle)), 0, cos(radians(angle))]])

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counter-clockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / sqrt(np.dot(axis, axis))
    a = cos(theta / 2.0)
    b, c, d = -axis * sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def rotateSP(sp, x, y):
    '''
        input is spherical coordinate
        x: positive theta,  y: negative phi
    '''
    sp_v = sphere2world(sp[0], sp[1])
    sp_v = np.dot(getRotationY(x), sp_v)
    temp_sc = world2sphere(sp_v[0], sp_v[1], sp_v[2])
    temp_v = sphere2world(temp_sc[0], 0)
    axis = np.dot(getRotationY(90), temp_v)
    sp_v = np.dot(rotation_matrix(axis, radians(y)), sp_v)
    return world2sphere(sp_v[0], sp_v[1], sp_v[2])