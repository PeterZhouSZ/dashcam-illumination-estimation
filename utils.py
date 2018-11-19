# Math
import numpy as np
from math import *
# Custom libs
from libs.projections import sphere2world, bin2Sphere

def vMF(SP, kappa=80.0):	
	'''
		discrete the sky into 256 bins and model the sky probability distirbution. (von Mises-Fisher)
	'''
	sp_vec = sphere2world(SP[0], SP[1])
	pdf = np.zeros(256)
	for i in range(256):
		sp = bin2Sphere(i)
		vec = sphere2world(sp[0], sp[1])
		pdf[i] = exp(np.dot(vec, sp_vec) * kappa)
	return pdf/np.sum(pdf)

def getAngle(c1, c2):
    '''
    inputs are spherical coordinate format [theta, phi]
    Compute the angle between two given vectors
    '''
    v1, v2 = sphere2world(c1[0], c1[1]), sphere2world(c2[0], c2[1])
    norm_v1, norm_v2 = v1/np.linalg.norm(v1), v2/np.linalg.norm(v2)
    return degrees(np.arccos(np.clip(np.dot(norm_v1, norm_v2), -1.0, 1.0)))