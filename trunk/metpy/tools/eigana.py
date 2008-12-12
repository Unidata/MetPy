#!/usr/bin/python

__all__ = ['orthomax','pca','procrustes','promax']

import numpy as np
import scipy.signal as SS

def pca(data,**args):
    """
    compute loadings and scores from data matrix 'data'

    return loadings, scores,

    if return_eig=True, return loadings,pcscore,(e_val,e_vec)

    required args:

        data : array
               data array

    optional args:

        return_eig : bool
                     True/False - return tuple of eigenvalues,eigenvectors
                    (default: False)

        rotate : string
                 Rotate loadings using specified rotation
                 valid: 'promax','orthomax'

        rotation_type: string
                       specify type of rotation
                       valid : 'orthogonal','oblique'
                       default : 'orthogonal'
                       works for rotate='promax'

        use: when set to 'corr', use correlation matrix
             when set to 'cov', use covariance matrix
             (default 'corr')

        use_loadings: when unset, use all loadings to compute new scores
                      when set to interger, use that many loadings to compute scores
                      (defaul: 'all')
    """
    keys=args.keys()

    if 'use' in keys:
        use = args['use']
    else:
        use = 'corr'

    if 'use_loadings' in keys:
        use_loadings = args['use_loadings']
    else:
        use_loadings = 'all'

    if 'return_eig' in keys:
        return_eig = args['return_eig']
    else:
        return_eig = False

    if 'rotate' in keys:
        rotate = args['rotate']
    else:
        rotate = None

    if 'rotation_type' in keys:
        rotation_type = args['rotation_type']
    else:
        rotation_type = 'orthogonal'

    z=np.ones((data.shape))

    for i in range(0,len(data)):
        z[i,:]=(data[i,:]-data[i,:].mean())/data[i,:].std()

    z=z.transpose()

    if use == 'corr':
        mat=np.corrcoef(data)
    elif use == 'cov':
        mat=np.cov(data)

    e_val,e_vec=np.linalg.eig(np.matrix(mat))

    #loadings <- eig$vectors %*% sqrt(diag(eig$values))
    loadings = e_vec*np.sqrt(np.diag(e_val))

    if use_loadings != 'all':
        loadings=loadings[:,0:use_loadings]

    if rotate is 'promax':
        loadings,tmat = promax(loadings,rotation_type)
    elif rotate is 'orthomax':
        loadings,tmat = orothomax(loadings)

    #pcscore<-z%*%rloadPromax$rmat%*%solve(t(rloadPromax$rmat)%*%rloadPromax$rmat)
    pcscore = np.matrix(z)*np.matrix(loadings)*np.linalg.inv(np.matrix(loadings.transpose())*np.matrix(loadings))

    if return_eig:
        return loadings,pcscore,(e_val,e_vec)
    else:
        return loadings,pcscore

def reconstruct(loadings,pcscore):
    '''
    bring eigenvariables back to physical space (z_tilda)
    '''
    return pcscore*loadings.transpose()


'''

Rotate Factors

'''
#
# Procrustes function
#
def procrustes(A, target, rot_type):
    #PROCRUSTES Procrustes rotation of FA or PCA loadings.
    d,m=A.shape

    if rot_type == 'orthogonal':
        L,D,M = np.linalg.svd(np.dot(target.transpose(),A))
        T = np.dot(M.transpose(),L.transpose())

    elif rot_type == 'oblique':
        if A.shape[0] == A.shape[1]:
            T = np.linalg.solve(A,target)
        else:
            T = np.linalg.lstsq(A,target)[0]

        tmp1 = np.dot(T.transpose(),T)
        tmp2 = np.linalg.solve(tmp1,np.eye(m)) 
        T=np.dot(T,np.diag(np.sqrt(np.diag(tmp2))))
    
    B = np.dot(A,T)
    return B,T
#
# orthomax function
#
def orthomax(A, gamma=1, reltol=1.4901e-07, maxit=256):
    #ORTHOMAX Orthogonal rotation of FA or PCA loadings.
    d,m=A.shape
    B = np.copy(A)
    T = np.eye(m)

    converged = False
    if (0 <= gamma) & (gamma <= 1):
        while converged == False:
#           Use Lawley and Maxwell's fast version
            D = 0
            for k in range(1,maxit+1):
                Dold = D
                tmp11 = np.sum(np.power(B,2),axis=0)
                tmp1 = np.matrix(np.diag(np.array(tmp11).flatten()))
                tmp2 = gamma*B
                tmp3 = d*np.power(B,3)
                L,D,M=np.linalg.svd(np.dot(A.transpose(),tmp3-np.dot(tmp2,tmp1)))
                T = np.dot(L,M)
                D = np.sum(np.diag(D))
                B = np.dot(A,T)
                if (np.abs(D - Dold)/D < reltol):
                    converged = True
                    break
    else:
#       Use a sequence of bivariate rotations
        for iter in range(1,maxit+1):
            maxTheta = 0
            for i in range(0,m-1):
                for j in range(i,m):
                    Bi=B[:,i]
                    Bj=B[:,j]
                    u = np.multiply(Bi,Bi)-np.multiply(Bj,Bj)
                    v = 2*np.multiply(Bi,Bj)
                    usum=u.sum()
                    vsum=v.sum()
                    numer = 2*np.dot(u.transpose(),v)-2*gamma*usum*vsum/d
                    denom=np.dot(u.transpose(),u) - np.dot(v.transpose(),v) - gamma*(usum**2 - vsum**2)/d
                    theta = np.arctan2(numer,denom)/4
                    maxTheta=max(maxTheta,np.abs(theta))
                    Tij = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
                    B[:,[i,j]] = np.dot(B[:,[i,j]],Tij)
                    T[:,[i,j]] = np.dot(T[:,[i,j]],Tij)
            if (maxTheta < reltol):
                converged = True
                break
    return B,T
#
# Promax function
#
def promax(A, rot_type='orthogonal',power=4, gamma=1, normalize='off', reltol=1.4901e-07, maxit=256):
    #PROMAX Promax oblique rotation of FA or PCA loadings.
    d,m=A.shape
    if power < 0:
        power = 4

    # Create target matrix from orthomax (defaults to varimax) solution
    B0,T0 = orthomax(A, gamma, reltol, maxit)
    target = np.multiply(np.sign(B0),np.power(np.abs(B0),power))
    # Oblique rotation to target
    B,T = procrustes(A, target, rot_type)

    return B,T
