"""
.. module:: pykonal.transformations
   A module to facilitate coordinate-system transformations.
"""


import numpy as np


def sph2sph(nodes, origin):
    '''
    Transform input spherical coordinates to new spherical coordinate
    system.
    :param nodes: Grid-node coordinates (spherical) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param origin: Coordinates (spherical) of the origin of the primed
        coordinate system w.r.t. the unprimed coordinate system.
    :type origin: 3-tuple, list, np.ndarray
    :return: Coordinates of input nodes in new (spherical) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    xx = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    zz = nodes[...,0] * np.cos(nodes[...,1])
    x0 = origin[0] * np.sin(origin[1]) * np.cos(origin[2])
    y0 = origin[0] * np.sin(origin[1]) * np.sin(origin[2])
    z0 = origin[0] * np.cos(origin[1])
    xx += x0
    yy += y0
    zz += z0
    xyz = np.moveaxis(np.stack([xx, yy, zz]), 0, -1)
    rr  = np.sqrt(np.sum(np.square(xyz), axis=-1))
    old = np.seterr(divide='ignore', invalid='ignore')
    tt  = np.arccos(xyz[...,2] / rr)
    np.seterr(**old)
    pp  = np.arctan2(xyz[...,1], xyz[...,0])
    rtp = np.moveaxis(np.stack([rr, tt, pp]), 0, -1)
    return (rtp)


def xyz2sph(nodes, translation):
    '''
    Transform input Cartesian coodinates to new spherical coordinate
    system.
    :param nodes: Grid-node coordinates (Cartesian) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param translation: Coordinates (Cartesian) of the origin of the primed
        coordinate system w.r.t. the umprimed coordinate system.
    :type translation: 3-tuple, list, np.ndarray
    :return: Coordinates of input nodes in new (spherical) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    # TODO: update arguments to take origin instead of translation
    xyz = nodes - translation
    rr  = np.sqrt(np.sum(np.square(xyz), axis=-1))
    old = np.seterr(divide='ignore', invalid='ignore')
    tt  = np.arccos(xyz[...,2] / rr)
    np.seterr(**old)
    pp  = np.arctan2(xyz[...,1], xyz[...,0])
    rtp = np.moveaxis(np.stack([rr, tt, pp]), 0, -1)
    return (rtp)


def sph2xyz(nodes, translation):
    '''
    Transform input spherical coodinates to new Cartesian coordinate
    system.
    :param nodes: Grid-node coordinates (spherical) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param translation: Coordinates (spherical) of the translation of the primed
        coordinate system w.r.t. the unprimed coordinate system.
    :type translation: 3-tuple, list, np.ndarray
    :return: Coordinates of input nodes in new (Cartesian) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    # TODO: update arguments to take origin instead of translation
    xx  = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy  = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    zz  = nodes[...,0] * np.cos(nodes[...,1])
    translation = [
        translation[0] * np.sin(translation[1]) * np.cos(translation[2]),
        translation[0] * np.sin(translation[1]) * np.sin(translation[2]),
        translation[0] * np.cos(translation[1])
    ]
    xx -= translation[0]
    yy -= translation[1]
    zz -= translation[2]
    xyz = np.moveaxis(np.stack([xx, yy, zz]), 0, -1)
    return (xyz)


def xyz2xyz(nodes, translation):
    '''
    Transform input Cartesian coodinates to new (primed) Cartesian coordinate
    system.
    :param nodes: Grid-node coordinates (Cartesian) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param translation: Coordinates (Cartesian) of the origin of the primed
        coordinate system w.r.t. the unprimed coordinate system.
    :type translation: 3-tuple, list, np.ndarray
    :return: Coordinates of nodes in primed (Cartesian) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    # TODO: update arguments to take origin instead of translation
    return (nodes - translation)


def rotation_matrix(alpha, beta, gamma):
    '''
    Return the rotation matrix used to rotate a set of cartesian
    coordinates by alpha radians about the z-axis, then beta radians
    about the y'-axis and then gamma radians about the z''-axis.
    :param alpha: Angle to rotate about the z-axis.
    :type alpha: float
    :param beta: Angle to rotate about the y'-axis.
    :type beta: float
    :param gamma: Angle to rotate about the z''-axis.
    :type gamma: float
    :return:
    :rtype: (3x3) np.ndarray
    '''
    aa = np.array(
        [
            [np.cos(alpha), -np.sin(alpha), 0           ],
            [np.sin(alpha),  np.cos(alpha), 0           ],
            [0,              0,             1           ]
        ]
    )
    bb = np.array(
        [
            [ np.cos(beta), 0,              np.sin(beta)],
            [ 0,            1,              0           ],
            [-np.sin(beta), 0,              np.cos(beta)]
        ]
    )
    cc = np.array(
        [
            [np.cos(gamma), -np.sin(gamma), 0           ],
            [np.sin(gamma),  np.cos(gamma), 0           ],
            [0,              0,             1           ]
        ]
    )
    return (aa.dot(bb).dot(cc))
