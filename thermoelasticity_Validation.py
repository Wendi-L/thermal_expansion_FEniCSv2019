__author__ = "Wendi Liu <wendi.liu@stfc.ac.uk>;             \
            Alex Skillen <alex.skillen@stfc.ac.uk>;            \
            Malgorzata Zimon <malgorzata.zimon@uk.ibm.com>;    \
            Robert Sawko <RSawko@uk.ibm.com>;                \
            Charles Moulinec <charles.moulinec@stfc.ac.uk>;    \
            Chris Thompson <christhompson@uk.ibm.com>"
__date__ = "20-09-2018"
__copyright__ = "Copyright (C) 2018 SCD of STFC"
__license__ = "GNU GPL version 3 or any later version"

#_________________________________________________________________________________________
#
#%% Benchmark solution on ring plate proposed by Zander et al. (2012) and Li et al. (2018)
#
#%% Last changed: 21-09-2018 15:03
#
#%% Update on 3rd Feb 2024 to add MUI coupling function
#
#_________________________________________________________________________________________

#_________________________________________________________________________________________
#
#%% Import packages
#_________________________________________________________________________________________

from dolfin import *
from mshr import *
import numpy as np
import math
import sys
from time import sleep
from mpi4py import MPI

#_________________________________________________________________________________________
#
#%% Mode option
#_________________________________________________________________________________________

iInputTemperature = 1         #    0-The temperature distribution will be calculated; 1-The temperature distribution will be received by MUI.
rMUISearch = 0.5            #   MUI sampler search radius


#_________________________________________________________________________________________
#
#%% Set form compiler options
#_________________________________________________________________________________________

#Allow extrapolation
parameters["allow_extrapolation"] = True
# Turn on optimization
parameters["form_compiler"]["optimize"]     = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
# Use UFLACS to speed-up assembly and limit quadrature degree
parameters['form_compiler']['quadrature_degree'] = 4

#_________________________________________________________________________________________
#
#%% Initialize parallelized computation and set target folder
#_________________________________________________________________________________________


if iInputTemperature == 0:
    # MPI claims 
    LOCAL_COMM_WORLD = MPI.COMM_WORLD
elif iInputTemperature == 1:
    import mui4py
    import petsc4py

    # MUI parameters
    dimensionMUI = 3
    data_types = {"dispX": mui4py.FLOAT64,
                  "dispY": mui4py.FLOAT64,
                  "dispZ": mui4py.FLOAT64,
                  "temperature": mui4py.FLOAT64}
    # MUI interface creation
    domain = "structureDomain"
    config3d = mui4py.Config(dimensionMUI, mui4py.FLOAT64)

    # App common world claims
    LOCAL_COMM_WORLD = mui4py.mpi_split_by_app()
    
    iface = ["threeDInterface0"]
    ifaces3d = mui4py.create_unifaces(domain, iface, config3d)
    ifaces3d["threeDInterface0"].set_data_types(data_types)
    
    # Necessary to avoid hangs at PETSc vector communication
    petsc4py.init(comm=LOCAL_COMM_WORLD)

processID = LOCAL_COMM_WORLD.Get_rank()
# Folder directory 
outputFolderName = "./Benchmark_results/"
inputFolderName = "./Data_input/"

if processID == 0: print("********** THERMO-ELASTICITY SIMULATION BEGIN **********")

#_________________________________________________________________________________________
#
#%% Solid mechanical and thermal parameters input
#_________________________________________________________________________________________

E_s = 1.0        # Young's Modulus [Pa]
rho_s = 1000.0    # Density of solid [kg/m^3]
nu_s = 0.0        # Poisson ratio [-]

T_ref = 0.0        # Reference temperature [K]
T_inner = 3.0    # Temperature at inner circle boundary [K]
T_outer = 1.0    # Temperature at outer circle boundary [K]

alpha = 1.0        # Coefficient of thermal expansion [1/K]
c_m = 910e-6    # Specific heat capacity [m^2/(K*s^2)]
k = 1.0            # Thermal conductivity [kg*m/(K*s^3)]

#_________________________________________________________________________________________
#
#%% Calculate Lame and thermal parameters
#_________________________________________________________________________________________

mu_s = Constant(E_s/(2.0*(1.0 + nu_s))) 
lamda_s  = Constant(E_s*nu_s/((1+nu_s)*(1-2*nu_s)))
kappa  = Constant(alpha*(2*mu_s + 3*lamda_s))

if processID == 0: print("E: ", E_s, "[Pa] ", "rho: ", rho_s, "[kg/m^3] ", "nu: ", nu_s, "[-] ")
if processID == 0: print("Reference temperature: ", T_ref, "[K] ", "Temperature at inner circle boundary: ", T_inner, "[K] ", "Temperature at outer circle boundary: ", T_outer, "[K] ")
if processID == 0: print("Coefficient of thermal expansion: ", alpha, "[1/K] ", "Specific heat capacity: ", c_m, "[m^2/(K*s^2)] ", "Thermal conductivity: ", k, "[kg*m/(K*s^3)] ")

#_________________________________________________________________________________________
#
#%% Time marching parameter input
#_________________________________________________________________________________________

Time = float(10)# Total time [s]
dt = float(1.0)    # Time step [s]
t = 0            # Start time [s]

if processID == 0: print("Total time: ", Time, "Time step size: ", dt)

#_________________________________________________________________________________________
#
#%% Solid Model dimension input
#_________________________________________________________________________________________

x0_inner = 0.0         # Center point (x-axis) coordinate of the inner circle [m]
y0_inner = 0.0         # Center point (y-axis) coordinate of the inner circle [m]
x0_outer = 0.0         # Center point (x-axis) coordinate of the outer circle [m]
y0_outer = 0.0         # Center point (y-axis) coordinate of the outer circle [m]
R_inner = 0.25        #Radius of the inner circle [m]
R_outer = 1.0        #Radius of the outer circle [m]
H_cylinder = 0.05    #Height of the cylinder (under 3D condition only) [m]

#_________________________________________________________________________________________
#
#%% Solid calculation selection
#_________________________________________________________________________________________

iMeshMethos = 1                #    0-Generate mesh; 1-Load mesh from file.
iLoadXML = 0                #    0-Load mesh from HDF5 file; 1-Load mesh from XML file.
iInteractiveMeshShow = 0     #    0-Do not show the generated mesh; 1-Show the generated mesh interactively.
i3DMeshGeneration = 1        #    0-Generate 2D mesh; 1-Generate 3D mesh.
iHDF5FileExport = 0         #    0-The HDF5 File Export function closed; 1-The HDF5 File Export function opened.
iHDF5MeshExport = 0         #    0-The HDF5 Mesh Export function closed; 1-The HDF5 Mesh Export function opened. 
iHDF5SubdomainsExport = 1     #    0-The HDF5 Subdomains Export function closed; 1-The HDF5 Subdomains Export function opened.
iHDF5BoundariesExport = 1     #    0-The HDF5 Boundaries Export function closed; 1-The HDF5 Boundaries Export function opened.
iHDF5SubdomainsImport = 1     #    0-The HDF5 Subdomains Import function closed; 1-The HDF5 Subdomains Import function opened.
iHDF5BoundariesImport = 1     #    0-The HDF5 Boundaries Import function closed; 1-The HDF5 Boundaries Import function opened.
iDebug = 0                     #    0-Switch off the debug level codes (if any); 1-Switch on the debug level codes (if any).

#_________________________________________________________________________________________
#
#%% Solid Mesh numbers input
#_________________________________________________________________________________________

if iMeshMethos == 0:
    Ndomain = 100     # number of fragments
    Nmesh = 100     # mesh density

#_________________________________________________________________________________________
#
#%% Solid Mesh generation/input
#_________________________________________________________________________________________

if iMeshMethos == 0:
    # Generate mesh
    domain2D = Circle(Point(x0_outer, y0_outer), R_outer, Ndomain) - Circle(Point(x0_inner, y0_inner), R_inner, Ndomain)
    if i3DMeshGeneration == 0:
        if processID == 0: print("Generating 2D mesh...")
        mesh = generate_mesh(domain2D, Nmesh)
    elif i3DMeshGeneration == 1:
        if processID == 0: print("Generating 3D mesh...")
        # Extruded 2D geometry to 3D
        domain3D = Extrude2D(domain2D, H_cylinder) # The z "thickness"
        mesh = generate_mesh(domain3D, Nmesh)
    else:
        sys.exit("ERROR, please select the correct value for i3DMeshGeneration")
        
elif iMeshMethos == 1:
    # Load mesh from file
    if iLoadXML == 0:
        if processID == 0: print("Loading HDF5 mesh...",  flush=True)
        mesh = BoxMesh(LOCAL_COMM_WORLD, Point(0, 0, 0), Point(1, 1, 1), 5, 5, 5)
        hdf_in = HDF5File(LOCAL_COMM_WORLD, inputFolderName + "mesh_boundary_and_values.h5", "r")
        hdf_in.read(mesh, "/mesh", False)
        print("Done",  flush=True)
    elif iLoadXML == 1:
        if processID == 0: print("Loading XML mesh...")
        mesh = Mesh(inputFolderName + "cylinder.xml")
    else:
        sys.exit("ERROR, please select the correct value for iLoadXML")

else:
    sys.exit("ERROR, please select the correct mesh generation method")

if iHDF5FileExport == 1:
    if processID == 0: print("Exporting HDF5 mesh...",  flush=True)
    hdf_out = HDF5File(LOCAL_COMM_WORLD, outputFolderName + "mesh_boundary_and_values.h5", "w")
    if iHDF5MeshExport == 1: hdf_out.write(mesh, "/mesh")

if iInteractiveMeshShow == 1:
    if processID == 0: print("Interactive Mesh Show ...")
    plt.figure()
    p = plot(mesh, title = "Mesh plot")        
    plt.show()
    
#_________________________________________________________________________________________
#
#%% Define coefficients
#_________________________________________________________________________________________

deg = 2                         # Interpolation degree
gdim = mesh.geometry().dim()     # Geometry dimensions
I = Identity(gdim)                 # Identity matrix
times = []                         # Time list

#_________________________________________________________________________________________
#
#%% Define function spaces
#_________________________________________________________________________________________

ele_vec = VectorElement('CG', mesh.ufl_cell(), deg)         # Displacement Vector element
ele_fin = FiniteElement('CG', mesh.ufl_cell(), deg)         # Temperature Finite element
v_scalar = FunctionSpace(mesh, "P", deg)                    # Stress scalar function space
# V = FunctionSpace(mesh, MixedElement([ele_vec, ele_fin]))    # Mixed (displacement & Temperature) function space

if iInputTemperature == 0:                                                            
    V = FunctionSpace(mesh, MixedElement([ele_vec, ele_fin]))
elif iInputTemperature == 1:
    V = FunctionSpace(mesh, ele_vec)                        # displacement function space

#_________________________________________________________________________________________
#
#%% Define functions, test functions and trail functions
#_________________________________________________________________________________________

if iInputTemperature == 0:
    U = Function(V)
    Uold = Function(V)
    U_ = TestFunction(V)
    dU = TrialFunction(V)

    (u, T) = split(U)
    (uold, Told) = split(Uold)
    (u_, T_) = split(U_)
    (du, dT) = split(dU)

elif iInputTemperature == 1:
    u = Function(V)
    uold = Function(V)
    u_ = TestFunction(V)
    du = TrialFunction(V)
    dT = Function(v_scalar)

#_________________________________________________________________________________________
#
#%% Define SubDomains
#_________________________________________________________________________________________

if (iMeshMethos == 1) and (iHDF5SubdomainsImport == 1):
    if iLoadXML == 0:
        if processID == 0: print("Loading HDF5 subdomains ...",  flush=True)
        subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
        hdf_in.read(subdomains, "/subdomains")
    elif iLoadXML == 1:
        if processID == 0: print("Loading XML subdomains ...",  flush=True)
        subdomains = MeshFunction("size_t", mesh, inputFolderName + "cylinder_physical_region.xml")
    else:
        sys.exit("ERROR, please select the correct value for iLoadXML")
else:
    if processID == 0: print("Creating subdomains ...",  flush=True)
    class inner_boundary( SubDomain ):
        def inside (self , x, on_boundary ):
            tol = 1e-3
            return near(x[0]**2 + x[1]**2, R_inner**2, tol)

    class outerer_boundary( SubDomain ):
        def inside (self , x, on_boundary ):
            tol = 1e-3
            return near(x[0]**2 + x[1]**2, R_outer**2, tol)

if (iHDF5FileExport == 1) and (iHDF5SubdomainsExport == 1):
    if processID == 0: print("Exporting HDF5 subdomains...",  flush=True)
    subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    hdf_out.write(subdomains, "/subdomains")

#_________________________________________________________________________________________
#
#%% Define and mark mesh boundaries
#_________________________________________________________________________________________

if (iMeshMethos == 1) and (iHDF5BoundariesImport == 1):
    if iLoadXML == 0:
        if processID == 0: print("Loading HDF5 boundaries ...",  flush=True)
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
        hdf_in.read(boundaries, "/boundaries")
    elif iLoadXML == 1:
        if processID == 0: print("Loading XML boundaries ...",  flush=True)
        boundaries = MeshFunction("size_t", mesh, inputFolderName + "cylinder_facet_region.xml")        
    else:
        sys.exit("ERROR, please select the correct value for iLoadXML")
else:
    if processID == 0: print("Creating boundaries ...",  flush=True)
    boundaries = MeshFunction("size_t", mesh, gdim-1)

    boundaries.set_all(0)
    outerer_boundary().mark (boundaries,1)
    inner_boundary().mark (boundaries,2)

if (iHDF5FileExport == 1) and (iHDF5BoundariesExport == 1): 
    if processID == 0: print("Exporting HDF5 boundaries...",  flush=True)
    hdf_out.write(boundaries, "/boundaries")

if (iMeshMethos == 1) and (iLoadXML == 0): hdf_in.close()
if iHDF5FileExport == 1: hdf_out.close()

ds = Measure("ds", domain=mesh, subdomain_data=boundaries, metadata={'quadrature_degree': deg})

if processID == 0: print("Dofs: ",V.dim(), "Cells:", mesh.num_cells(),  flush=True)
if processID == 0: print("geometry dimension: ",gdim,  flush=True)

#_________________________________________________________________________________________
#
#%% Define boundary conditions
#_________________________________________________________________________________________

if iInputTemperature == 0:
    bc1 = DirichletBC(V.sub(0).sub(0), Expression(("0.25*cos(atan2(x[1],x[0]))"),degree=deg), boundaries, 2)
    bc2 = DirichletBC(V.sub(0).sub(1), Expression(("0.25*sin(atan2(x[1],x[0]))"),degree=deg), boundaries, 2)
    
    bc3 = DirichletBC(V.sub(0).sub(0), Constant(0.), boundaries, 1)    
    bc4 = DirichletBC(V.sub(0).sub(1), Constant(0.), boundaries, 1)
    
    bc5 = DirichletBC(V.sub(1), T_inner, boundaries, 2)
    bc6 = DirichletBC(V.sub(1), T_outer, boundaries, 1)
    
    if gdim == 2:
        if processID == 0: print("Creating 2D boundary conditions ...",  flush=True)
        bcs = [bc1, bc2, bc3, bc4, bc5, bc6]
    
    elif gdim == 3:
        if processID == 0: print("Creating 3D boundary conditions ...",  flush=True)
        bc7 = DirichletBC(V.sub(0).sub(2), Constant(0.), boundaries, 1)
        bc8 = DirichletBC(V.sub(0).sub(2), Constant(0.), boundaries, 2)
        
        bcs = [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8]
    
    else:
        sys.exit("ERROR, gdim is a wrong value")

elif iInputTemperature == 1:
        if processID == 0: print ("Creating 3D boundary conditions ...   ", end="", flush=True)
        print ("{FENICS} Here 01 ",  flush=True)
        # bc1 = DirichletBC(V.sub(0), Expression(("0.25*cos(atan2(x[1],x[0]))"),degree=deg), boundaries, 2)
        # bc2 = DirichletBC(V.sub(1), Expression(("0.25*sin(atan2(x[1],x[0]))"),degree=deg), boundaries, 2)
        print ("{FENICS} Here 02 ",  flush=True)
        bc3 = DirichletBC(V.sub(0), Constant(0.), boundaries, 1)    
        bc4 = DirichletBC(V.sub(1), Constant(0.), boundaries, 1)
        print ("{FENICS} Here 03 ",  flush=True)
        if gdim == 2:
            if processID == 0: print ("Creating 2D boundary conditions ...   ", end="", flush=True)
            bcs = [bc1, bc2, bc3, bc4]

        elif gdim == 3:
            print ("{FENICS} Here 04 ",  flush=True)
            if processID == 0: print ("Creating 3D boundary conditions ...   ", end="", flush=True)
            bc5 = DirichletBC(V.sub(2), Constant(0.), boundaries, 1)
            bc6 = DirichletBC(V.sub(2), Constant(0.), boundaries, 2)
        
            # bcs = [bc1, bc2, bc3, bc4, bc5, bc6]
            bcs = [bc3, bc4, bc5, bc6]

#_________________________________________________________________________________________
#
#%% Define Stress functions
#_________________________________________________________________________________________

# Define the linear Lagrangian Green strain tensor
def epsilon(v): 
    return (0.5 * (grad(v) + (grad(v).T)))

# Define the Second Piola-Kirchhoff stress tensor
def sigma(v, dT):
    return (lamda_s * tr(epsilon(v)) * I - kappa * dT * I + 2.0 * mu_s * epsilon(v))

# Define the von Mises stress tensor
def von_Mises (v, dT, dimension):

    vM_term_1_1 = (sigma(v, dT)[0,0] - sigma(v, dT)[1,1])**2
    vM_term_2_1 = sigma(v, dT)[0,1]**2
    
    if dimension == 3:
        vM_term_1_2 = (sigma(v, dT)[1,1] - sigma(v, dT)[2,2])**2
        vM_term_1_3 = (sigma(v, dT)[2,2] - sigma(v, dT)[0,0])**2
        vM_term_2_2 = sigma(v, dT)[1,2]**2
        vM_term_2_3 = sigma(v, dT)[2,0]**2
    elif dimension == 2:
        vM_term_1_2 = (sigma(v, dT)[1,1])**2
        vM_term_1_3 = (-sigma(v, dT)[0,0])**2
        vM_term_2_2 = 0
        vM_term_2_3 = 0
    else:
        sys.exit("ERROR, wrong dimension passed to von_Mises")

    return sqrt( 0.5*(vM_term_1_1 + vM_term_1_2 + vM_term_1_3) + 3.0*(vM_term_2_1 + vM_term_2_2 + vM_term_2_3) )     

# Define the function to project the von Mises stress tensor to a scalar function space
def von_Mises_projected (v, dT, dimension, scalar_function_space):
    return project(von_Mises(v, dT, dimension), scalar_function_space)

#_________________________________________________________________________________________
#
#%% Prepare post-process files
#_________________________________________________________________________________________

print("prepare post-process files",  flush=True)
# dis_file = File(outputFolderName + "displacement.pvd")
# str_file = File(outputFolderName + "stress.pvd")
# tmp_file = File(outputFolderName + "temperature.pvd")
print ("{FENICS} Here 05 ",  flush=True)
#_________________________________________________________________________________________
#
#%% Define the variational FORM functions of solid thermo-elasticities
#_________________________________________________________________________________________

# Define the mechanical weak variational form
Form_TE_mechanical = inner(sigma(du, dT), epsilon(u_))*dx

if iInputTemperature == 0:
    # Define the thermal weak variational form
    Form_TE_thermal = ((rho_s*c_m*(dT-Told)/dt)*T_ + (kappa*T_ref*tr(epsilon(du-uold))/dt)*T_)*dx
    Form_TE_thermal += (dot(k*grad(dT), grad(T_)))*dx
    
    Form_TE = Form_TE_mechanical + Form_TE_thermal
    
elif iInputTemperature == 1:
    Form_TE = Form_TE_mechanical

else:
    sys.exit("ERROR, please select the correct value for iInputTemperature")
print ("{FENICS} Here 06 ",  flush=True)
#_________________________________________________________________________________________
#
#%% Initialize solver
#_________________________________________________________________________________________

if iInputTemperature == 0:
    problem = LinearVariationalProblem(lhs(Form_TE), rhs(Form_TE), U, bcs=bcs, \
        form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": deg})
elif iInputTemperature == 1:
    problem = LinearVariationalProblem(lhs(Form_TE), rhs(Form_TE), u, bcs=bcs, \
        form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": deg})
else:
    sys.exit("ERROR, please select the correct value for iInputTemperature")

solver = LinearVariationalSolver(problem)

solver.parameters['linear_solver'] = "mumps"
print ("{FENICS} Here 07 ",  flush=True)
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#%% Define MUI samplers and commit ZERO step
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if iInputTemperature == 1:

    # send_min_X = sys.float_info.max
    # send_min_Y = sys.float_info.max
    # send_min_Z = sys.float_info.max
    #
    # send_max_X = -sys.float_info.max
    # send_max_Y = -sys.float_info.max
    # send_max_Z = -sys.float_info.max
    # print ("{FENICS} Here 10 ",  flush=True)
    # dT_temp0 = Function(v_scalar)
    # dT_vec0 = dT_temp0.vector().get_local()
    # print ("{FENICS} Here 11 ",  flush=True)
    # for i in range(len(dT_vec0)):
    #     if (v_scalar.tabulate_dof_coordinates()[i][0] < send_min_X):
    #         send_min_X = v_scalar.tabulate_dof_coordinates()[i][0]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][1] < send_min_Y):
    #         send_min_Y = v_scalar.tabulate_dof_coordinates()[i][1]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][2] < send_min_Z):
    #         send_min_Z = v_scalar.tabulate_dof_coordinates()[i][2]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][0] > send_max_X):
    #         send_max_X = v_scalar.tabulate_dof_coordinates()[i][0]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][1] > send_max_Y):
    #         send_max_Y = v_scalar.tabulate_dof_coordinates()[i][1]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][2] > send_max_Z):
    #         send_max_Z = v_scalar.tabulate_dof_coordinates()[i][2]
    # print ("{FENICS} Here 08 ",  flush=True)
    # # Set up sending span
    # span_push = mui4py.geometry.Box([send_min_X, send_min_Y, send_min_Z],
    #                                 [send_max_X, send_max_Y, send_max_Z])
    # synchronised=False
    # # Announce the MUI send span
    # ifaces3d["threeDInterface0"].announce_send_span(0, (Time/dt), span_push, synchronised)
    #
    # recv_min_X = sys.float_info.max
    # recv_min_Y = sys.float_info.max
    # recv_min_Z = sys.float_info.max
    #
    # recv_max_X = -sys.float_info.max
    # recv_max_Y = -sys.float_info.max
    # recv_max_Z = -sys.float_info.max
    #
    # for i in range(len(dT_vec0)):
    #     if (v_scalar.tabulate_dof_coordinates()[i][0] < recv_min_X):
    #         recv_min_X = v_scalar.tabulate_dof_coordinates()[i][0]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][1] < recv_min_Y):
    #         recv_min_Y = v_scalar.tabulate_dof_coordinates()[i][1]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][2] < recv_min_Z):
    #         recv_min_Z = v_scalar.tabulate_dof_coordinates()[i][2]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][0] > recv_max_X):
    #         recv_max_X = v_scalar.tabulate_dof_coordinates()[i][0]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][1] > recv_max_Y):
    #         recv_max_Y = v_scalar.tabulate_dof_coordinates()[i][1]
    #
    #     if (v_scalar.tabulate_dof_coordinates()[i][2] > recv_max_Z):
    #         recv_max_Z = v_scalar.tabulate_dof_coordinates()[i][2]
    # print ("{FENICS} Here 09 ",  flush=True)
    # # Set up receiving span
    # span_fetch = mui4py.geometry.Box([recv_min_X, recv_min_Y, recv_min_Z],
    #                                  [recv_max_X, recv_max_Y, recv_max_Z])
    #
    # # Announce the MUI receive span
    # ifaces3d["threeDInterface0"].announce_recv_span(0, (Time/dt), span_fetch, synchronised)

    if processID == 0: print ("{FENICS} Defining MUI samplers ...   ", end="", flush=True)
    s_sampler = mui4py.SamplerPseudoNearestNeighbor(rMUISearch)
    t_sampler = mui4py.ChronoSamplerExact()

    a = ifaces3d["threeDInterface0"].commit(0)
print ("{FENICS} Here 12 ",  flush=True)
#_________________________________________________________________________________________
#
#%% Define time loops
#_________________________________________________________________________________________

while t <= Time:
    if processID == 0: print("Time: ", t,  flush=True)
    
    # Update real time    
    times.append(t)

    # MUI Fetch
    if iInputTemperature == 1:
        if processID == 0: print("Obtaining Temperature Space at Present Time Step ...   ", end="", flush=True)
        dT_temp = Function(v_scalar)
        dT_vec = dT_temp.vector().get_local()
        for i in range(len(dT_vec)):
            # dist = math.sqrt(pow(v_scalar.tabulate_dof_coordinates()[i][0],2) + pow(v_scalar.tabulate_dof_coordinates()[i][1],2))
            # dT_vec[i] = 9.3453*pow(dist,6) - 40.27*pow(dist,5) + 72.808*pow(dist,4) - 72.138*pow(dist,3) + 43.406*pow(dist,2) - 17.756*dist + 5.6052
            dT_vec[i] = ifaces3d["threeDInterface0"].fetch("temperature", v_scalar.tabulate_dof_coordinates()[i], len(times), s_sampler, t_sampler)
        dT_temp.vector().set_local(dT_vec)
        dT_temp.vector().apply("insert")
        dT.assign(dT_temp)

    if processID == 0: print("Solving ...   ", end="", flush=True)

    # Solving the thermo-elasticity functions inside the time loop
    solver.solve()
    
    # MUI Push
    if iInputTemperature == 1:
        if processID == 0: print("Sending Displacement at Present Time Step ...   ", end="", flush=True)
        d_vec_x = u.vector().get_local()[0::3]
        d_vec_y = u.vector().get_local()[1::3]
        d_vec_z = u.vector().get_local()[2::3]
        
        for i in range(len(d_vec_x)):
            ifaces3d["threeDInterface0"].push("dispX", V.tabulate_dof_coordinates()[i], (d_vec_x[i]))
        
        for i in range(len(d_vec_y)):
            ifaces3d["threeDInterface0"].push("dispY", V.tabulate_dof_coordinates()[i], (d_vec_y[i]))
        
        for i in range(len(d_vec_z)):
            ifaces3d["threeDInterface0"].push("dispZ", V.tabulate_dof_coordinates()[i], (d_vec_z[i]))
        
        a = ifaces3d["threeDInterface0"].commit(len(dT_vec))

    # Calculate Von. Mises Stress
    if iInputTemperature == 0:
        vM_Stress = von_Mises_projected(u, T, gdim, v_scalar)
    elif iInputTemperature == 1:
        vM_Stress = von_Mises_projected(u, dT, gdim, v_scalar)
    else:
        sys.exit("ERROR, please select the correct value for iInputTemperature")

    # Move to next interval
    t += dt
    
    if iInputTemperature == 0:
        dispi,tempi = U.split(True)
        Uold.assign(U)

        # Rename parameters
        dispi.rename('Displacement', 'dispi')
        tempi.rename('Temperature', 'tempi')
        vM_Stress.rename('von_Mises stress', 'sigma')
    
        # Write to file on displacement, temperature and Von. Mises Stress    
        # dis_file << dispi
        # tmp_file << tempi
        # str_file << vM_Stress

    elif iInputTemperature == 1:
        uold.assign(u)

        u.rename('Displacement', 'dispi')
        dT.rename('Temperature', 'tempi')
        vM_Stress.rename('von_Mises stress', 'sigma')

        # dis_file << u
        # tmp_file << dT
        # str_file << vM_Stress

    else:
        sys.exit("ERROR, please select the correct value for iInputTemperature")

#_________________________________________________________________________________________
#
#%% Finishing Log
#_________________________________________________________________________________________

if processID == 0: 
    print("********** THERMO-ELASTICITY SIMULATION COMPLETED **********")