import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt

def stiffness_matrix(D,points,coords):
    assert (len(points) == 3)
    x0, y0, z0 = coords[points[0]]
    x1, y1, z1 = coords[points[1]]
    x2, y2, z2 = coords[points[2]]
    
    Ae = 0.5*abs((x0 - x1)*(y2 - y1) - (y0 - y1)*(x2 - x1))
    Be = np.array([[y1 - y2,     0.0, y2 - y0,     0.0, y0 - y1,     0.0],
               [    0.0, x2 - x1,     0.0, x0 - x2,     0.0, x1 - x0],
               [x2 - x1, y1 - y2, x0 - x2, y2 - y0, x1 - x0, y0 - y1]])/(2*Ae)
    K = Ae*np.matmul(Be.transpose(), np.matmul(D, Be))
    return K

def assemble_matrix(mesh,D):
    coords,points = mesh
    num_points = len(coords)*2
    K_global = scipy.sparse.csr_matrix((num_points,num_points),dtype=np.float32)
    
    for tri in points:
        K = stiffness_matrix(D,tri,coords)
        ent = np.empty(6,dtype='int')
        ent[0::2] = tri*2
        ent[1::2] = tri*2 + 1
        for i, ind in enumerate(ent):
            for j, jnd in enumerate(ent):
                K_global[ind,jnd] += K[i,j]
    return K_global

def set_bc(K, F, dof, val, dir_):
    if dir_ == 'y':
        dof = 2*dof + 1
    elif dir_ == 'x':
        dof = 2*dof
    K[dof] = 0.0
    K[dof, dof] = 1.0
    F[dof] = val
    
def contact_kernel(K,Kn,pairs,nx,ny):
    dof_slave = pairs[:,0]
    dof_master = pairs[:,1]
    
    C = np.array([nx,ny,-nx,-ny])
    kernel = Kn*np.outer(C, C.transpose())
    
    for j in range(2):
        K[2*dof_slave,2*dof_slave+j] += kernel[0,0+j]
        K[2*dof_slave,2*dof_master+j] += kernel[0,2+j]
        K[2*dof_slave+1,2*dof_slave+j] += kernel[1,0+j]
        K[2*dof_slave+1,2*dof_master+j] += kernel[1,2+j]
        K[2*dof_master,2*dof_slave+j] += kernel[2,0+j]
        K[2*dof_master,2*dof_master+j] += kernel[2,2+j]
        K[2*dof_master+1,2*dof_slave+j] += kernel[3,0+j]
        K[2*dof_master+1,2*dof_master+j] += kernel[3,2+j]
        
def plot(mesh,data=None):
    coord,node = mesh
    x,y,z = coord[:,0],coord[:,1],coord[:,2]
    
    fig = plt.figure(dpi=300)
    ax = plt.subplot(111)
    ax.set_aspect('equal')
    ax.set_xlim([x.min()-1,x.max()+1])
    ax.set_ylim([y.min()-1,y.max()+1])
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    
    if data is not None:
        if len(data) == len(coord):
            mag = data
        else:
            ux = data[0::2]
            uy = data[1::2]
            mag = np.sqrt(ux**2+uy**2)
        tric = ax.tricontourf(x,y,node,mag,256,cmap='coolwarm')
        cbar = plt.colorbar(tric)
    ax.triplot(x,y,node,color='k',linewidth=0.5)
    plt.show()
