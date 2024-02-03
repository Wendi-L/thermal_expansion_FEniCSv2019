__author__ = "Wendi Liu <wendi.liu@stfc.ac.uk>; 			\
			Alex Skillen <alex.skillen@stfc.ac.uk>;			\
			Malgorzata Zimon <malgorzata.zimon@uk.ibm.com>;	\
			Robert Sawko <RSawko@uk.ibm.com>;				\
			Charles Moulinec <charles.moulinec@stfc.ac.uk>;	\
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
from fenics import *
from mshr import *
import numpy as np
import math
import sys
from time import sleep
import matplotlib.pyplot as plt

#_________________________________________________________________________________________
#
#%% Adjust log level
#_________________________________________________________________________________________

set_log_active(True)
set_log_level(PROGRESS) # PROGRESS DEBUG LEVEL False

# Starts the wall clock
tic()

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

# MPI claims 
MPI_Claims = mpi_comm_world()
processID = MPI.rank(mpi_comm_world())
# Folder directory 
outputFolderName = "./Benchmark_results/"
inputFolderName = "./Data_input/"

if processID == 0: print("********** THERMO-ELASTICITY SIMULATION BEGIN **********")

#_________________________________________________________________________________________
#
#%% Solid mechanical and thermal parameters input
#_________________________________________________________________________________________

E_s = 1.0		# Young's Modulus [Pa]
rho_s = 1000.0	# Density of solid [kg/m^3]
nu_s = 0.0		# Poisson ratio [-]

T_ref = 0.0		# Reference temperature [K]
T_inner = 3.0	# Temperature at inner circle boundary [K]
T_outer = 1.0	# Temperature at outer circle boundary [K]

alpha = 1.0		# Coefficient of thermal expansion [1/K]
c_m = 910e-6	# Specific heat capacity [m^2/(K*s^2)]
k = 1.0			# Thermal conductivity [kg*m/(K*s^3)]

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
dt = float(1.0)	# Time step [s]
t = 0			# Start time [s]

if processID == 0: print("Total time: ", Time, "Time step size: ", dt)

#_________________________________________________________________________________________
#
#%% Solid Model dimension input
#_________________________________________________________________________________________

x0_inner = 0.0 		# Center point (x-axis) coordinate of the inner circle [m]
y0_inner = 0.0 		# Center point (y-axis) coordinate of the inner circle [m]
x0_outer = 0.0 		# Center point (x-axis) coordinate of the outer circle [m]
y0_outer = 0.0 		# Center point (y-axis) coordinate of the outer circle [m]
R_inner = 0.25		#Radius of the inner circle [m]
R_outer = 1.0		#Radius of the outer circle [m]
H_cylinder = 0.05	#Height of the cylinder (under 3D condition only) [m]

#_________________________________________________________________________________________
#
#%% Solid calculation selection
#_________________________________________________________________________________________

iMeshMethos = 1				#	0-Generate mesh; 1-Load mesh from file.
iLoadXML = 0				#	0-Load mesh from HDF5 file; 1-Load mesh from XML file.
iInteractiveMeshShow = 0 	#	0-Do not show the generated mesh; 1-Show the generated mesh interactively.
i3DMeshGeneration = 1		#	0-Generate 2D mesh; 1-Generate 3D mesh.
iHDF5FileExport = 1 		#	0-The HDF5 File Export function closed; 1-The HDF5 File Export function opened.
iHDF5MeshExport = 1 		#	0-The HDF5 Mesh Export function closed; 1-The HDF5 Mesh Export function opened. 
iHDF5SubdomainsExport = 1 	#	0-The HDF5 Subdomains Export function closed; 1-The HDF5 Subdomains Export function opened.
iHDF5BoundariesExport = 1 	#	0-The HDF5 Boundaries Export function closed; 1-The HDF5 Boundaries Export function opened.
iHDF5SubdomainsImport = 1 	#	0-The HDF5 Subdomains Import function closed; 1-The HDF5 Subdomains Import function opened.
iHDF5BoundariesImport = 1 	#	0-The HDF5 Boundaries Import function closed; 1-The HDF5 Boundaries Import function opened.
iDebug = 0 					#	0-Switch off the debug level codes (if any); 1-Switch on the debug level codes (if any).

#_________________________________________________________________________________________
#
#%% Solid Mesh numbers input
#_________________________________________________________________________________________

if iMeshMethos == 0:
	Ndomain = 100 	# number of fragments
	Nmesh = 100 	# mesh density

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
		if processID == 0: print("Loading HDF5 mesh...")
		mesh = Mesh()
		hdf_in = HDF5File(MPI_Claims, inputFolderName + "mesh_boundary_and_values.h5", "r")
		hdf_in.read(mesh, "/mesh", False)
	elif iLoadXML == 1:
		if processID == 0: print("Loading XML mesh...")
		mesh = Mesh(inputFolderName + "cylinder.xml")
	else:
		sys.exit("ERROR, please select the correct value for iLoadXML")

else:
	sys.exit("ERROR, please select the correct mesh generation method")

if iHDF5FileExport == 1:
	if processID == 0: print("Exporting HDF5 mesh...")
	hdf_out = HDF5File(MPI_Claims, outputFolderName + "mesh_boundary_and_values.h5", "w")
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

deg = 2 						# Interpolation degree
gdim = mesh.geometry().dim() 	# Geometry dimensions
I = Identity(gdim) 				# Identity matrix
times = [] 						# Time list

#_________________________________________________________________________________________
#
#%% Define function spaces
#_________________________________________________________________________________________

ele_vec = VectorElement('CG', mesh.ufl_cell(), deg) 		# Displacement Vector element
ele_fin = FiniteElement('CG', mesh.ufl_cell(), deg) 		# Temperature Finite element
v_scalar = FunctionSpace(mesh, "P", deg)					# Stress scalar function space
V = FunctionSpace(mesh, MixedElement([ele_vec, ele_fin]))	# Mixed (displacement & Temperature) function space

#_________________________________________________________________________________________
#
#%% Define functions, test functions and trail functions
#_________________________________________________________________________________________

U = Function(V)
Uold = Function(V)
U_ = TestFunction(V)
dU = TrialFunction(V)

(u, T) = split(U)
(uold, Told) = split(Uold)
(u_, T_) = split(U_)
(du, dT) = split(dU)

#_________________________________________________________________________________________
#
#%% Define SubDomains
#_________________________________________________________________________________________

if (iMeshMethos == 1) and (iHDF5SubdomainsImport == 1):
	if iLoadXML == 0:
		if processID == 0: print("Loading HDF5 subdomains ...")
		subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
		hdf_in.read(subdomains, "/subdomains")
	elif iLoadXML == 1:
		if processID == 0: print("Loading XML subdomains ...")
		subdomains = MeshFunction("size_t", mesh, inputFolderName + "cylinder_physical_region.xml")
	else:
		sys.exit("ERROR, please select the correct value for iLoadXML")
else:
	if processID == 0: print("Creating subdomains ...")
	class inner_boundary( SubDomain ):
		def inside (self , x, on_boundary ):
			tol = 1e-3
			return near(x[0]**2 + x[1]**2, R_inner**2, tol)

	class outerer_boundary( SubDomain ):
		def inside (self , x, on_boundary ):
			tol = 1e-3
			return near(x[0]**2 + x[1]**2, R_outer**2, tol)

if (iHDF5FileExport == 1) and (iHDF5SubdomainsExport == 1):
	if processID == 0: print("Exporting HDF5 subdomains...")
	subdomains = MeshFunction("size_t", mesh, mesh.topology().dim())
	hdf_out.write(subdomains, "/subdomains")

#_________________________________________________________________________________________
#
#%% Define and mark mesh boundaries
#_________________________________________________________________________________________

if (iMeshMethos == 1) and (iHDF5BoundariesImport == 1):
	if iLoadXML == 0:
		if processID == 0: print("Loading HDF5 boundaries ...")
		boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
		hdf_in.read(boundaries, "/boundaries")
	elif iLoadXML == 1:
		if processID == 0: print("Loading XML boundaries ...")
		boundaries = MeshFunction("size_t", mesh, inputFolderName + "cylinder_facet_region.xml")		
	else:
		sys.exit("ERROR, please select the correct value for iLoadXML")
else:
	if processID == 0: print("Creating boundaries ...")
	boundaries = MeshFunction("size_t", mesh, gdim-1)

	boundaries.set_all(0)
	outerer_boundary().mark (boundaries,1)
	inner_boundary().mark (boundaries,2)

if (iHDF5FileExport == 1) and (iHDF5BoundariesExport == 1): 
	if processID == 0: print("Exporting HDF5 boundaries...")
	hdf_out.write(boundaries, "/boundaries")

if (iMeshMethos == 1) and (iLoadXML == 0): hdf_in.close()
if iHDF5FileExport == 1: hdf_out.close()

ds = Measure("ds", domain=mesh, subdomain_data=boundaries, metadata={'quadrature_degree': deg})

if processID == 0: print("Dofs: ",V.dim(), "Cells:", mesh.num_cells())
if processID == 0: print("geometry dimension: ",gdim)

#_________________________________________________________________________________________
#
#%% Define boundary conditions
#_________________________________________________________________________________________

bc1 = DirichletBC(V.sub(0).sub(0), Expression(("0.25*cos(atan2(x[1],x[0]))"),degree=deg), boundaries, 2)
bc2 = DirichletBC(V.sub(0).sub(1), Expression(("0.25*sin(atan2(x[1],x[0]))"),degree=deg), boundaries, 2)

bc3 = DirichletBC(V.sub(0).sub(0), Constant(0.), boundaries, 1)	
bc4 = DirichletBC(V.sub(0).sub(1), Constant(0.), boundaries, 1)

bc5 = DirichletBC(V.sub(1), T_inner, boundaries, 2)
bc6 = DirichletBC(V.sub(1), T_outer, boundaries, 1)

if gdim == 2:
	if processID == 0: print("Creating 2D boundary conditions ...")
	bcs = [bc1, bc2, bc3, bc4, bc5, bc6]

elif gdim == 3:
	if processID == 0: print("Creating 3D boundary conditions ...")
	bc7 = DirichletBC(V.sub(0).sub(2), Constant(0.), boundaries, 1)
	bc8 = DirichletBC(V.sub(0).sub(2), Constant(0.), boundaries, 2)
	
	bcs = [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8]

else:
	sys.exit("ERROR, gdim is a wrong value")

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

print("prepare post-process files")
dis_file = File(outputFolderName + "displacement.pvd")
str_file = File(outputFolderName + "stress.pvd")
tmp_file = File(outputFolderName + "temperature.pvd")

#_________________________________________________________________________________________
#
#%% Define the variational FORM functions of solid thermo-elasticities
#_________________________________________________________________________________________

# Define the mechanical weak variational form
Form_TE_mechanical = inner(sigma(du, dT), epsilon(u_))*dx

# Define the thermal weak variational form
Form_TE_thermal = ((rho_s*c_m*(dT-Told)/dt)*T_ + (kappa*T_ref*tr(epsilon(du-uold))/dt)*T_)*dx
Form_TE_thermal += (dot(k*grad(dT), grad(T_)))*dx

Form_TE = Form_TE_mechanical + Form_TE_thermal

#_________________________________________________________________________________________
#
#%% Initialize solver
#_________________________________________________________________________________________

problem = LinearVariationalProblem(lhs(Form_TE), rhs(Form_TE), U, bcs=bcs, \
	form_compiler_parameters={"cpp_optimize": True, "representation": "uflacs", "quadrature_degree": deg})

solver = LinearVariationalSolver(problem)

solver.parameters['linear_solver'] = "mumps"

#_________________________________________________________________________________________
#
#%% Define time loops
#_________________________________________________________________________________________

while t <= Time:
	if processID == 0: print("Time: ", t)
	
	# Update real time	
	times.append(t)

	# Solving the thermo-elasticity functions inside the time loop
	solver.solve()
	
	# Calculate Von. Mises Stress
	vM_Stress = von_Mises_projected(u, T, gdim, v_scalar)
	
	# Move to next interval
	t += dt
	dispi,tempi = U.split(True)
	Uold.assign(U)
	
	# Rename parameters
	dispi.rename('Displacement', 'dispi')
	tempi.rename('Temperature', 'tempi')
	vM_Stress.rename('von_Mises stress', 'sigma')

	# Write to file on displacement, temperature and Von. Mises Stress	
	dis_file << dispi
	tmp_file << tempi
	str_file << vM_Stress

#_________________________________________________________________________________________
#
#%% Plot figures at final time step
#_________________________________________________________________________________________

plt.figure()
p = plot(von_Mises (u, T, gdim), title="von Mises stress")
plt.xlim((-4*R_inner, 4*R_inner))
plt.ylim((-4*R_inner, 4*R_inner))
plt.colorbar(p)
plt.savefig(outputFolderName + 'von_Mises_stress.png')

plt.figure()
p = plot(T, title="Temperature variation")
plt.xlim((-4*R_inner, 4*R_inner))
plt.ylim((-4*R_inner, 4*R_inner))
plt.colorbar(p)
plt.savefig(outputFolderName + 'Temperature.png')

if gdim == 2:
	plt.figure()
	p = plot(u, title="Displacement variation")
	plt.xlim((-4*R_inner, 4*R_inner))
	plt.ylim((-4*R_inner, 4*R_inner))
	plt.colorbar(p)
	plt.savefig(outputFolderName + 'Displacement.png')

#_________________________________________________________________________________________
#
#%% Calculate wall time
#_________________________________________________________________________________________

# Finish the wall clock
simtime = toc()

if processID == 0: 
	print("Total Simulation time: %g [s]" % simtime)
	print("\n")
	print("********** THERMO-ELASTICITY SIMULATION COMPLETED **********")