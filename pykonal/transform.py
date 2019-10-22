'''
.. module:: pykonal.transform
   A module to facilitate coordinate-system transformations.
'''


import numpy as np


def sph2sph(nodes, origin):
    '''
    Transform input spherical coordinates to new spherical coordinate
    system.
    :param nodes: Grid-node coordinates (spherical) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param origin: Coordinates (spherical) of the origin of the input
        coordinate system w.r.t. the new coordinate system.
    :type origin: 3-tuple, list, np.ndarray
    :return: Coordinates of input nodes in new (spherical) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    xx_in = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy_in = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    zz_in = nodes[...,0] * np.cos(nodes[...,1])
    origin_xyz = [
        origin[0] * np.sin(origin[1]) * np.cos(origin[2]),
        origin[0] * np.sin(origin[1]) * np.sin(origin[2]),
        origin[0] * np.cos(origin[1])
    ]
    xx_new  = xx_in + origin_xyz[0]
    yy_new  = yy_in + origin_xyz[1]
    zz_new  = zz_in + origin_xyz[2]
    xyz_new = np.moveaxis(np.stack([xx_new,yy_new,zz_new]), 0, -1)
    rr_new  = np.sqrt(np.sum(np.square(xyz_new), axis=-1))
    old     = np.seterr(divide='ignore', invalid='ignore')
    tt_new  = np.arccos(xyz_new[...,2] / rr_new)
    np.seterr(**old)
    pp_new  = np.arctan2(xyz_new[...,1], xyz_new[...,0])
    rtp_new = np.moveaxis(np.stack([rr_new, tt_new, pp_new]), 0, -1)
    return (rtp_new)


def xyz2sph(nodes, origin, rotate=False):
    '''
    Transform input Cartesian coodinates to new spherical coordinate
    system.
    :param nodes: Grid-node coordinates (Cartesian) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param origin: Coordinates (Cartesian) of the origin of the input
        coordinate system w.r.t. the new coordinate system.
    :type origin: 3-tuple, list, np.ndarray
    :param rotate: Rotate the output coordinates so the coordinate axes
        align with geographic coordinate axes (latitude, longitude, and
        depth)
    :type rotate: boolean, optional, default=False
    :return: Coordinates of input nodes in new (spherical) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    origin_xyz = [
        origin[0] * np.sin(origin[1]) * np.cos(origin[2]),
        origin[0] * np.sin(origin[1]) * np.sin(origin[2]),
        origin[0] * np.cos(origin[1])
    ]
    if rotate is True:
        nodes = nodes.dot(
            rotation_matrix(np.pi/2-origin[2], 0, np.pi/2-origin[1])
        )
    else:
        nodes = nodes
    xyz_new = nodes + origin_xyz
    rr_new  = np.sqrt(np.sum(np.square(xyz_new), axis=-1))
    old     = np.seterr(divide='ignore', invalid='ignore')
    tt_new  = np.arccos(xyz_new[...,2] / rr_new)
    np.seterr(**old)
    pp_new  = np.arctan2(xyz_new[...,1], xyz_new[...,0])
    rtp_new = np.moveaxis(np.stack([rr_new,tt_new, pp_new]), 0, -1)
    return (rtp_new)


def sph2xyz(nodes, origin):
    '''
    Transform input spherical coodinates to new Cartesian coordinate
    system.
    :param nodes: Grid-node coordinates (spherical) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param origin: Coordinates (Cartesian) of the origin of the input
        coordinate system w.r.t. the new coordinate system.
    :type origin: 3-tuple, list, np.ndarray
    :return: Coordinates of input nodes in new (Cartesian) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    xx_in   = nodes[...,0] * np.sin(nodes[...,1]) * np.cos(nodes[...,2])
    yy_in   = nodes[...,0] * np.sin(nodes[...,1]) * np.sin(nodes[...,2])
    zz_in   = nodes[...,0] * np.cos(nodes[...,1])
    xx_new  = xx_in + origin[0]
    yy_new  = yy_in + origin[1]
    zz_new  = zz_in + origin[2]
    xyz_new = np.moveaxis(np.stack([xx_new,yy_new,zz_new]), 0, -1)
    return (xyz_new)


def xyz2xyz(nodes, origin):
    '''
    Transform input Cartesian coodinates to new Cartesian coordinate
    system.
    :param nodes: Grid-node coordinates (Cartesian) to transform.
    :type nodes: (MxNxPx3) np.ndarray
    :param origin: Coordinates (Cartesian) of the origin of the input
        coordinate system w.r.t. the new coordinate system.
    :type origin: 3-tuple, list, np.ndarray
    :return: Coordinates of input nodes in new (Cartesian) coordinate
        system.
    :rtype: (MxNxPx3) np.ndarray
    '''
    return (nodes + origin)


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
