import gmsh
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import matplotlib.pyplot as plt
import Assembly as FEM

def generate_mesh(name=None,width=None,height=None,ms=None,mf=None):
    gmsh.initialize()
    gmsh.model.add(name)

    # boundary nodes
    ll = gmsh.model.occ.addPoint(0,0,0,ms,1)
    lr = gmsh.model.occ.addPoint(width,0,0,ms,2)
    ur = gmsh.model.occ.addPoint(width,height,0,ms,3)
    ul = gmsh.model.occ.addPoint(0,height,0,ms,4)

    # boundary lines
    bt = gmsh.model.occ.addLine(ll,lr,1)
    rt = gmsh.model.occ.addLine(lr,ur,2)
    tp = gmsh.model.occ.addLine(ur,ul,3)
    lt = gmsh.model.occ.addLine(ul,ll,4)

    # boundary curve loop
    bd = gmsh.model.occ.addCurveLoop([bt,rt,tp,lt],1)

    # boundary surface
    s = gmsh.model.occ.addPlaneSurface([bd])

    # fracture geometry
    fl = gmsh.model.occ.addPoint(width*0.25,height*0.25,0,mf,5) # 0.35
    fr = gmsh.model.occ.addPoint(width*0.75,height*0.75,0,mf,6) # 0.65

    fe = gmsh.model.occ.addLine(fl,fr)

    gmsh.model.occ.synchronize()

    gmsh.model.mesh.embed(1,[fe],2,s)
    gmsh.model.mesh.embed(0,[fl,fr],2,s)

    gmsh.model.mesh.generate(2)
    gdim = 2
    gmsh.model.addPhysicalGroup(gdim-1, [bt],tag=1)
    gmsh.model.setPhysicalName(gdim-1, 1, 'Base')
    gmsh.model.addPhysicalGroup(gdim-1, [rt],tag=2)
    gmsh.model.setPhysicalName(gdim-1, 2, 'Right')
    gmsh.model.addPhysicalGroup(gdim-1, [tp],tag=3)
    gmsh.model.setPhysicalName(gdim-1, 3, 'Top')
    gmsh.model.addPhysicalGroup(gdim-1, [lt],tag=4)
    gmsh.model.setPhysicalName(gdim-1, 4, 'Left')
    gmsh.model.addPhysicalGroup(gdim-1, [fe],tag=5)
    gmsh.model.addPhysicalGroup(gdim, [s],tag=10)
    gmsh.model.setPhysicalName(gdim, 10, 'Omega')

    gmsh.model.addPhysicalGroup(gdim-2, [ur],tag=20)
    gmsh.model.setPhysicalName(gdim-2, 20, 'Point')

    gmsh.plugin.setNumber('Crack','Dimension',1)
    gmsh.plugin.setNumber('Crack','PhysicalGroup',5)
    gmsh.plugin.setNumber('Crack','NewPhysicalGroup',6)
    gmsh.plugin.run("Crack")
    gmsh.model.setPhysicalName(gdim-1, 5, 'Fracture_minus')
    gmsh.model.setPhysicalName(gdim-1, 6, 'Fracture_plus')

    gmsh.model.mesh.renumberNodes()
    gmsh.model.occ.synchronize()
    
    cell_types, cell_tags, cell_node_tags = gmsh.model.mesh.getElements(dim=2)
    cell_tags = cell_tags[0]
    cell_node_tags = cell_node_tags[0].reshape((len(cell_tags),3))
    cell_node_tags -= 1
    
    node_tags, coord, param_coords = gmsh.model.mesh.getNodes()
    coord = coord.reshape((int(len(coord)/3),3))
    
    node_slave,coord_slave = gmsh.model.mesh.getNodesForPhysicalGroup(1,5)
    node_master,coord_master = gmsh.model.mesh.getNodesForPhysicalGroup(1,6)
    
    coord_slave = coord_slave.reshape((int(len(coord_slave)/3),3))
    coord_master = coord_master.reshape((int(len(coord_master)/3),3))

    gmsh.finalize()
    
    k = 0
    h = np.zeros(len(cell_node_tags))
    for tri in cell_node_tags:
        x0, y0, z0 = coord[tri[0]]
        x1, y1, z1 = coord[tri[1]]
        x2, y2, z2 = coord[tri[2]]
        a = np.sqrt((x0-x1)**2+(y0-y1)**2+(z0-z1)**2)
        b = np.sqrt((x2-x1)**2+(y2-y1)**2+(z2-z1)**2)
        c = np.sqrt((x0-x2)**2+(y0-y2)**2+(z0-z2)**2)
        s = 0.5*(a+b+c)
        h[k] = 2*a*b*c/4/np.sqrt(s*(s-a)*(s-b)*(s-c))
        k += 1

    mesh = (coord,cell_node_tags)
    contact_node,contact_coord = (node_slave,node_master), (coord_slave,coord_master)
    pairs = np.zeros([len(contact_node[0]),2],dtype=int)

    for i in range(len(contact_node[0])):
        for j in range(len(contact_node[1])):
            if np.all(np.isclose(coord_slave[i,:],coord_master[j,:])):
                pairs[i,0] = contact_node[0][i] - 1
                pairs[i,1] = contact_node[1][j] - 1

    idx = np.where(pairs[:,0] == pairs[:,1])[0]
    pairs = np.delete(pairs,idx,axis=0)
    
    return mesh, pairs, h
        
mesh,pairs,cell_diameter = generate_mesh(name="Fracture",width=10,height=10,ms=1.0,mf=0.25)

dof_top = np.where(np.isclose(mesh[0][:,1],10))[0]
dof_base = np.where(np.isclose(mesh[0][:,1],0))[0]
dof_left = np.where(np.isclose(mesh[0][:,0],0))[0]
dof_right = np.where(np.isclose(mesh[0][:,0],10))[0]

unique, counts = np.unique(mesh[0],return_counts=True)
num_top_nodes = counts[np.where(unique == 10)[0]]
dof_top_bd = np.where((mesh[0][dof_top,0] == 0) | (mesh[0][dof_top,0] == 10))[0]

E,nu = 5e9,0.20
G = E/2/(1+nu)
D = E/((1+nu)*(1-2*nu))*np.array([[1-nu,nu,0],
                                  [nu,1-nu,0],
                                  [0,0,(1-2*nu)/2]])

Kn = E/min(cell_diameter)
K_global = FEM.assemble_matrix(mesh,D)
F_global = scipy.sparse.csr_matrix((len(mesh[0])*2,1),dtype=np.float32)

FEM.set_bc(K_global,F_global,dof_base,0.0,'y')
FEM.set_bc(K_global,F_global,dof_left,0.0,'x')
FEM.set_bc(K_global,F_global,dof_right,0.0,'x')

F_global[2*dof_top+1] = -10e6/(num_top_nodes-1) #Number of cells on top!!
F_global[2*dof_top[dof_top_bd]+1] *= 0.5
    
nx,ny = np.cos(np.radians(45)), -np.sin(np.radians(45))
FEM.contact_kernel(K_global,Kn,pairs,nx,ny)

u = scipy.sparse.linalg.spsolve(K_global,F_global,permc_spec='COLAMD',use_umfpack=True)

ux = u[0::2]
uy = u[1::2]
FEM.plot(mesh,uy)


def residual(K,F,u):
    return K.dot(u) - F.toarray().reshape(len(u))
def L2_norm(K,F,u):
    return np.linalg.norm(residual(K,F,u))
def gap(u,pairs,nx,ny):
    return (u[0::2][pairs[:,0]] - u[0::2][pairs[:,1]])*nx + (u[1::2][pairs[:,0]] - u[1::2][pairs[:,1]])*ny

gap = gap(u,pairs,nx,ny)
res = L2_norm(K_global,F_global,u)


fig = plt.figure(dpi=300)
plt.plot(mesh[0][pairs[:,0],0],gap,'o')
plt.xlabel('X-Coordiante [m]')
plt.ylabel('Normal gap [m]')
plt.show()
