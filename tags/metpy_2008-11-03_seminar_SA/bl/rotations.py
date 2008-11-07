#!/usr/bin/python

import numpy as N
from metobs.generic import get_dims

def two_three_rot(ui,vi,wi,common=None,return_basis=False,unrotate = False,use_basis=None):
    '''
        Perform 2-3 rotation (only apply first two rotations)

        ui,vi,wi: input u,v, and w compoenents of velocity

        common: use `common` basis to perform rotation/unrotation (int)

        return_basis: Return the new basis vectors (True/False)

        unrotate: undo rotation (True/False)

        use_basis: list of basis vectors to use in the rotation [i(nx3),j(nx3),k(nx3)]

    '''
    sensors,obs,blks = get_dims(ui,vi,wi)

#
#   initilize arrays to force three dimensions
#
    u=N.reshape(ui,(sensors,obs,blks))
    v=N.reshape(vi,(sensors,obs,blks))
    w=N.reshape(wi,(sensors,obs,blks))

    meanu1=N.ones((sensors,obs,blks))
    meanu2=N.ones((sensors,obs,blks))
    u1=N.ones((sensors,blks))
    v1=N.ones((sensors,blks))
    w1=N.ones((sensors,blks))

    data_rot_u = N.ones((sensors,obs,blks))
    data_rot_v = N.ones((sensors,obs,blks))
    data_rot_w = N.ones((sensors,obs,blks))

    avg_axis = 1
    zeros = N.zeros((sensors,blks))
    sindex = range(0,sensors)

#
#   define cartesian unit vectors
#
    i1t = N.array([1,0,0])
    j1t = N.array([0,1,0])
    k1t = N.array([0,0,1])
    i1=i1t
    j1=j1t
    k1=k1t
    if sensors != 1:
        for i in range(0,sensors-1):
            i1=N.c_[i1,i1t]
            j1=N.c_[j1,j1t]
            k1=N.c_[k1,k1t]

    i1=i1.transpose()
    j1=j1.transpose()
    k1=k1.transpose()
#
#   Compute mean 2d and 3d wind vectors
#
    u1 = N.average(u,axis=avg_axis)  # mean u comp
    v1 = N.average(v,axis=avg_axis)  # mean v comp
    w1 = N.average(w,axis=avg_axis)  # mean w comp

    meanu1 = N.array([u1,v1,zeros])  # mean 2D wind vector
    meanu2 = N.array([u1,v1,w1])     # mean 3D wind vector

    for blk in range(0,blks):
#
#       define 2-3 rotation unit vectors
#
        if use_basis is None:
#         first rotation
#         i2 is aligned with mean horizontal wind
            i2_temp = meanu1[:,:,blk]/N.power(meanu1[0,:,blk]*meanu1[0,:,blk]+\
                                              meanu1[1,:,blk]*meanu1[1,:,blk]+\
                                              meanu1[2,:,blk]*meanu1[2,:,blk],0.5)
#         k2 is the same as k1 in original cartesian coordinate system
#         j2 is the cross product between k1 and the new i (i2)
            j2 = N.cross(k1[sindex,:],i2_temp.transpose()[sindex,:])
#
#         second rotation
#
#         now define i2 to be aligned with the mean vector wind
            i2 = N.transpose(meanu2[:,:,blk]/N.power(meanu2[0,:,blk]*meanu2[0,:,blk]+\
                                                     meanu2[1,:,blk]*meanu2[1,:,blk]+\
                                                     meanu2[2,:,blk]*meanu2[2,:,blk],0.5))
#         j2 is the result of the first rotation
#         k2 is the cross product between i2 and j2
            k2 = N.cross(i2[sindex,:],j2[sindex,:])
#         correct alg...need to store results in data_rot_u,v and w correctly.
        else:
            i2=use_basis[0]
            j2=use_basis[1]
            k2=use_basis[2]

        if not unrotate:
            i_new = i2
            j_new = j2
            k_new = k2
            i_old = i1
            j_old = j1
            k_old = k1
        else:
            i_new = i1
            j_new = j1
            k_new = k1
            i_old = i2
            j_old = j2
            k_old = k2

        if common == None:
            for i in range(0,sensors):
                data_rot_u[i,:,blk] = u[i,:,blk]*N.dot(i_old[i,:],i_new[i,:])+\
                             v[i,:,blk]*N.dot(j_old[i,:],i_new[i,:])+\
                             w[i,:,blk]*N.dot(k_old[i,:],i_new[i,:])

                data_rot_v[i,:,blk] = u[i,:,blk]*N.dot(i_old[i,:],j_new[i,:])+\
                             v[i,:,blk]*N.dot(j_old[i,:],j_new[i,:])+\
                             w[i,:,blk]*N.dot(k_old[i,:],j_new[i,:])

                data_rot_w[i,:,blk] = u[i,:,blk]*N.dot(i_old[i,:],k_new[i,:])+\
                             v[i,:,blk]*N.dot(j_old[i,:],k_new[i,:])+\
                             w[i,:,blk]*N.dot(k_old[i,:],k_new[i,:])
        else:
            for i in range(0,sensors):
                data_rot_u[i,:,blk] = u[i,:,blk]*N.dot(i_old[common,:],i_new[common,:])+\
                             v[i,:,blk]*N.dot(j_old[common,:],i_new[common,:])+\
                             w[i,:,blk]*N.dot(k_old[common,:],i_new[common,:])

                data_rot_v[i,:,blk] = u[i,:,blk]*N.dot(i_old[common,:],j_new[common,:])+\
                             v[i,:,blk]*N.dot(j_old[common,:],j_new[common,:])+\
                             w[i,:,blk]*N.dot(k_old[common,:],j_new[common,:])

                data_rot_w[i,:,blk] = u[i,:,blk]*N.dot(i_old[common,:],k_new[common,:])+\
                             v[i,:,blk]*N.dot(j_old[common,:],k_new[common,:])+\
                             w[i,:,blk]*N.dot(k_old[common,:],k_new[common,:])

    if return_basis:
        return data_rot_u,data_rot_v,data_rot_w,(i_new,j_new,k_new)
    else:
        return data_rot_u,data_rot_v,data_rot_w
