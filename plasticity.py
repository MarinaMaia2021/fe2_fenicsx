import numpy as np
import matplotlib.pyplot as plt

import gmsh
from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc
from petsc4py import PETSc

hsize = 0.2

Re = 1.3
Ri = 1.0

# Initialize geometry
gmsh.initialize()
gdim = 2
model_rank = 0
if MPI.COMM_WORLD.rank == 0:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Model")

    geom = gmsh.model.geo
    center = geom.add_point(0, 0, 0)
    p1 = geom.add_point(Ri, 0, 0)
    p2 = geom.add_point(Re, 0, 0)
    p3 = geom.add_point(0, Re, 0)
    p4 = geom.add_point(0, Ri, 0)

    x_radius = geom.add_line(p1, p2)
    outer_circ = geom.add_circle_arc(p2, center, p3)
    y_radius = geom.add_line(p3, p4)
    inner_circ = geom.add_circle_arc(p4, center, p1)

    boundary = geom.add_curve_loop([x_radius, outer_circ, y_radius, inner_circ])
    surf = geom.add_plane_surface([boundary])

    geom.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

    gmsh.model.addPhysicalGroup(gdim, [surf], 1)
    gmsh.model.addPhysicalGroup(gdim - 1, [x_radius], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [y_radius], 2, name="left")
    gmsh.model.addPhysicalGroup(gdim - 1, [inner_circ], 3, name="inner")

    gmsh.model.mesh.generate(gdim)

mesh_data = io.gmsh.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim
)

domain = mesh_data.mesh
facets = mesh_data.facet_tags

gmsh.finalize()

# Material parameters
E = fem.Constant(domain, 70e3)  # in MPa
nu = fem.Constant(domain, 0.3)
lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2.0 / (1 + nu)
sig0 = fem.Constant(domain, 250.0)
Et = E / 100.0
H = E * Et / (E - Et)

deg_u = 2
shape = (gdim,)
V = fem.functionspace(domain, ("P", deg_u, shape))

Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()
bottom_dofsy = fem.locate_dofs_topological((V.sub(1), Vy), gdim - 1, facets.find(1))
top_dofsx = fem.locate_dofs_topological((V.sub(0), Vx), gdim - 1, facets.find(2))

def bottom_inside(x):
    return np.logical_and(np.isclose(x[0], Ri), np.isclose(x[1], 0))

bottom_inside_dof = fem.locate_dofs_geometrical((V.sub(0), Vx), bottom_inside)[0]

u0x = fem.Function(Vx)
u0y = fem.Function(Vy)
bcs = [
    fem.dirichletbc(u0x, top_dofsx, V.sub(0)),
    fem.dirichletbc(u0y, bottom_dofsy, V.sub(1)),
]

n = ufl.FacetNormal(domain)
q_lim = float(2 / np.sqrt(3) * np.log(Re / Ri) * sig0)
loading = fem.Constant(domain, 0.0)

# Quadrature Element Setup
deg_quad = 2
W0e = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(), scheme="default", degree=deg_quad
)
We = basix.ufl.quadrature_element(
    domain.basix_cell(), value_shape=(4,), scheme="default", degree=deg_quad
)
W = fem.functionspace(domain, We)
W0 = fem.functionspace(domain, W0e)

sig = fem.Function(W)
sig_old = fem.Function(W)
n_elas = fem.Function(W)
beta = fem.Function(W0)
p = fem.Function(W0, name="Cumulative_plastic_strain")
dp = fem.Function(W0)
u = fem.Function(V, name="Total_displacement")
du = fem.Function(V, name="Iteration_correction")
Du = fem.Function(V, name="Current_increment")
v = ufl.TrialFunction(V)
u_ = ufl.TestFunction(V)

P0 = fem.functionspace(domain, ("DG", 0))
p_avg = fem.Function(P0, name="Plastic_strain")

def eps(v):
    e = ufl.sym(ufl.grad(v))
    return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]])

def elastic_behavior(eps_el):
    return lmbda * ufl.tr(eps_el) * ufl.Identity(3) + 2 * mu * eps_el

def as_3D_tensor(X):
    return ufl.as_tensor([[X[0], X[3], 0], [X[3], X[1], 0], [0, 0, X[2]]])

def to_vect(X):
    return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1]])

ppos = lambda x: ufl.max_value(x, 0)

def constitutive_update(Δε, old_sig, old_p):
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + elastic_behavior(Δε)
    s = ufl.dev(sig_elas)
    
    # Regularize to prevent 0 / 0 during initial pure elastic states
    sig_eq = ufl.sqrt(3 / 2.0 * ufl.inner(s, s))
    sig_eq_safe = ufl.max_value(sig_eq, 1e-10)
    
    f_elas = sig_eq - sig0 - H * old_p
    dp = ppos(f_elas) / (3 * mu + H)
    
    f_elas_safe = ufl.conditional(ufl.eq(f_elas, 0.0), 1e-10, f_elas)
    n_elas = s / sig_eq_safe * ppos(f_elas) / f_elas_safe
    
    beta = 3 * mu * dp / sig_eq_safe
    new_sig = sig_elas - beta * s
    return to_vect(new_sig), to_vect(n_elas), beta, dp

def sigma_tang(eps_tensor):
    N_elas = as_3D_tensor(n_elas)
    return (
        elastic_behavior(eps_tensor)
        - 3 * mu * (3 * mu / (3 * mu + H) - beta) * ufl.inner(N_elas, eps_tensor) * N_elas
        - 2 * mu * beta * ufl.dev(eps_tensor)
    )

ds = ufl.Measure("ds", domain=domain, subdomain_data=facets)
dx = ufl.Measure(
    "dx",
    domain=domain,
    metadata={"quadrature_degree": deg_quad, "quadand_scheme": "default"},
)

Residual = ufl.inner(eps(u_), as_3D_tensor(sig)) * dx - ufl.inner(-loading * n, u_) * ds(3)
tangent_form = ufl.inner(eps(v), sigma_tang(eps(u_))) * dx

basix_celltype = getattr(basix.CellType, domain.topology.cell_type.name)
quadrature_points, weights = basix.make_quadrature(basix_celltype, deg_quad)

map_c = domain.topology.index_map(domain.topology.dim)
num_cells = map_c.size_local + map_c.num_ghosts
cells = np.arange(0, num_cells, dtype=np.int32)

def interpolate_quadrature(ufl_expr, function):
    expr_expr = fem.Expression(ufl_expr, quadrature_points)
    expr_eval = expr_expr.eval(domain, cells)
    function.x.array[:] = expr_eval.flatten()[:]

# Compiled forms for optimized execution
residual_form_compiled = fem.form(-Residual)
tangent_form_compiled = fem.form(tangent_form)

# Allocate explicit PETSc structures
A_tang = fem.petsc.create_matrix(tangent_form_compiled)
b_res = fem.petsc.create_vector(V)  # Passes the FunctionSpace directly

# Configure direct linear solver
solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")
solver.setOperators(A_tang)

# Iteration parameters
Nitermax, tol = 200, 1e-6
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr + 1)[1:] ** 0.5
results = np.zeros((Nincr + 1, 3))

# Zero out initial conditions
functions_to_zero = [sig, sig_old, p, u, n_elas, beta]
for f in functions_to_zero:
    f.x.petsc_vec.set(0.0)
    f.x.scatter_forward()

# Execution Step Loop
for i, t in enumerate(load_steps):
    loading.value = t * q_lim
    
    # Calculate initial residual for step
    with b_res.localForm() as b_loc:
        b_loc.set(0.0)
    fem.petsc.assemble_vector(b_res, residual_form_compiled)
    
    du.x.array[:] = 0.0
    fem.petsc.apply_lifting(b_res, [tangent_form_compiled], bcs=[bcs], x0=[du.x.petsc_vec])
    b_res.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b_res, bcs, du.x.petsc_vec)
    
    nRes0 = b_res.norm()
    nRes = nRes0
    Du.x.array[:] = 0.0
    Du.x.scatter_forward()
    
    niter = 0
    while nRes / nRes0 > tol and niter < Nitermax:
        # Assemble matrix
        A_tang.zeroEntries()
        fem.petsc.assemble_matrix(A_tang, tangent_form_compiled, bcs=bcs)
        A_tang.assemble()
        
        # System solve
        solver.solve(b_res, du.x.petsc_vec)
        du.x.scatter_forward()
        
        # Update increment
        Du.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
        Du.x.scatter_forward()
        
        # Rebuild and interpolate graphs inside loop
        Δε = eps(Du)
        sig_, n_elas_, beta_, dp_ = constitutive_update(Δε, sig_old, p)
        
        interpolate_quadrature(sig_, sig)
        interpolate_quadrature(n_elas_, n_elas)
        interpolate_quadrature(beta_, beta)
        
        # Assemble refreshed residual vector
        with b_res.localForm() as b_loc:
            b_loc.set(0.0)
        fem.petsc.assemble_vector(b_res, residual_form_compiled)
        
        fem.petsc.apply_lifting(b_res, [tangent_form_compiled], bcs=[bcs], x0=[du.x.petsc_vec])
        b_res.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_res, bcs, du.x.petsc_vec)
        
        nRes = b_res.norm()
        niter += 1

    # Apply converged increments to state
    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()
    
    interpolate_quadrature(dp_, dp)
    p.x.petsc_vec.axpy(1.0, dp.x.petsc_vec)
    p.x.scatter_forward()
    
    sig_old.x.array[:] = sig.x.array[:]
    sig_old.x.scatter_forward()
    
    if len(bottom_inside_dof) > 0:
        results[i + 1, :] = (u.x.array[bottom_inside_dof[0]], t, niter)

# Generate visualizations
if len(bottom_inside_dof) > 0:
    plt.figure()
    plt.plot(results[:, 0], results[:, 1], "-oC3")
    plt.xlabel("Displacement of inner boundary")
    plt.ylabel(r"Applied pressure $q/q_{lim}$")
    plt.grid(True)
    plt.show()

if len(bottom_inside_dof) > 0:
    plt.figure()
    plt.bar(np.arange(Nincr + 1), results[:, 2], color="C2")
    plt.xlabel("Loading step")
    plt.ylabel("Number of iterations")
    plt.xlim(0)
    plt.grid(True)
    plt.show()