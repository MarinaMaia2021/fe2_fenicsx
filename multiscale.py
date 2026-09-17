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
Rm = 1.15  # Intermediate radius dividing plastic and elastic zones

# Initialize geometry
gmsh.initialize()
gdim = 2
model_rank = 0
if MPI.COMM_WORLD.rank == 0:
    gmsh.option.setNumber("General.Terminal", 0)
    gmsh.model.add("Model")

    geom = gmsh.model.geo
    center = geom.add_point(0, 0, 0)
    
    # Define points along axes
    p_i_x = geom.add_point(Ri, 0, 0)
    p_m_x = geom.add_point(Rm, 0, 0)
    p_e_x = geom.add_point(Re, 0, 0)
    
    p_e_y = geom.add_point(0, Re, 0)
    p_m_y = geom.add_point(0, Rm, 0)
    p_i_y = geom.add_point(0, Ri, 0)

    # Lines and Arcs
    x_radius_inner = geom.add_line(p_i_x, p_m_x)
    x_radius_outer = geom.add_line(p_m_x, p_e_x)
    
    outer_circ = geom.add_circle_arc(p_e_x, center, p_e_y)
    middle_circ = geom.add_circle_arc(p_m_x, center, p_m_y)
    inner_circ = geom.add_circle_arc(p_i_y, center, p_i_x)
    
    y_radius_outer = geom.add_line(p_e_y, p_m_y)
    y_radius_inner = geom.add_line(p_m_y, p_i_y)

    # Surfaces
    loop_plastic = geom.add_curve_loop([x_radius_inner, middle_circ, y_radius_inner, inner_circ])
    surf_plastic = geom.add_plane_surface([loop_plastic])
    
    loop_elastic = geom.add_curve_loop([x_radius_outer, outer_circ, y_radius_outer, -middle_circ])
    surf_elastic = geom.add_plane_surface([loop_elastic])

    geom.synchronize()

    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", hsize)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", hsize)

    # Physical Groups
    gmsh.model.addPhysicalGroup(gdim, [surf_plastic], 1, name="Plastic_Zone")
    gmsh.model.addPhysicalGroup(gdim, [surf_elastic], 2, name="Elastic_Zone")
    
    gmsh.model.addPhysicalGroup(gdim - 1, [x_radius_inner, x_radius_outer], 1, name="bottom")
    gmsh.model.addPhysicalGroup(gdim - 1, [y_radius_outer, y_radius_inner], 2, name="left")
    gmsh.model.addPhysicalGroup(gdim - 1, [inner_circ], 3, name="inner")

    gmsh.model.mesh.generate(gdim)

mesh_data = io.gmsh.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, model_rank, gdim=gdim
)

domain = mesh_data.mesh
facets = mesh_data.facet_tags
cell_tags = mesh_data.cell_tags

gmsh.finalize()

# ==========================================
# Material Parameters & Indicators
# ==========================================
# Create a cell indicator tracking material distributions
DG0 = fem.functionspace(domain, ("DG", 0))
material_indicator = fem.Function(DG0)
material_indicator.x.array[cell_tags.find(1)] = 1.0  # 1.0 -> Plastic Zone
material_indicator.x.array[cell_tags.find(2)] = 2.0  # 2.0 -> Elastic Zone
material_indicator.x.scatter_forward()

nu = fem.Constant(domain, 0.3)

# Plastic Zone Properties (Original File Values)
E_plas = fem.Constant(domain, 70e3)
lmbda_plas = E_plas * nu / (1 + nu) / (1 - 2 * nu)
mu_plas = E_plas / 2.0 / (1 + nu)
sig0 = fem.Constant(domain, 250.0)
Et = E_plas / 100.0
H = E_plas * Et / (E_plas - Et)

# Elastic Zone Properties (Adjust as needed, set to 70e3 here)
E_elas = fem.Constant(domain, 70e3) 
lmbda_elas = E_elas * nu / (1 + nu) / (1 - 2 * nu)
mu_elas = E_elas / 2.0 / (1 + nu)

# ==========================================
# Function Spaces and Setup
# ==========================================
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

# Helper Tensors
def eps(v):
    e = ufl.sym(ufl.grad(v))
    return ufl.as_tensor([[e[0, 0], e[0, 1], 0], [e[0, 1], e[1, 1], 0], [0, 0, 0]])

def elastic_behavior(eps_el, lmbda_param, mu_param):
    return lmbda_param * ufl.tr(eps_el) * ufl.Identity(3) + 2 * mu_param * eps_el

def as_3D_tensor(X):
    return ufl.as_tensor([[X[0], X[3], 0], [X[3], X[1], 0], [0, 0, X[2]]])

def to_vect(X):
    return ufl.as_vector([X[0, 0], X[1, 1], X[2, 2], X[0, 1]])

ppos = lambda x: ufl.max_value(x, 0)

# ==========================================
# Modular Constitutive Models
# ==========================================
def update_pure_elastic_model(Δε, old_sig):
    sig_n = as_3D_tensor(old_sig)
    new_sig = sig_n + elastic_behavior(Δε, lmbda_elas, mu_elas)
    zero_tensor = ufl.as_tensor(np.zeros((3, 3)))
    return to_vect(new_sig), to_vect(zero_tensor), 0.0, 0.0

def update_plastic_model(Δε, old_sig, old_p):
    sig_n = as_3D_tensor(old_sig)
    sig_elas = sig_n + elastic_behavior(Δε, lmbda_plas, mu_plas)
    s = ufl.dev(sig_elas)
    
    sig_eq = ufl.sqrt(3 / 2.0 * ufl.inner(s, s))
    sig_eq_safe = ufl.max_value(sig_eq, 1e-10)
    
    f_elas = sig_eq - sig0 - H * old_p
    dp = ppos(f_elas) / (3 * mu_plas + H)
    
    f_elas_safe = ufl.conditional(ufl.eq(f_elas, 0.0), 1e-10, f_elas)
    n_elas_tensor = s / sig_eq_safe * ppos(f_elas) / f_elas_safe
    
    beta_val = 3 * mu_plas * dp / sig_eq_safe
    new_sig = sig_elas - beta_val * s
    return to_vect(new_sig), to_vect(n_elas_tensor), beta_val, dp

def constitutive_update(Δε, old_sig, old_p):
    sig_p, n_p, beta_p, dp_p = update_plastic_model(Δε, old_sig, old_p)
    sig_e, n_e, beta_e, dp_e = update_pure_elastic_model(Δε, old_sig)
    
    is_plastic = ufl.eq(material_indicator, 1.0)
    
    new_sig = ufl.conditional(is_plastic, sig_p, sig_e)
    n_elas_val = ufl.conditional(is_plastic, n_p, n_e)
    beta_val = ufl.conditional(is_plastic, beta_p, beta_e)
    dp_val = ufl.conditional(is_plastic, dp_p, dp_e)
    
    return new_sig, n_elas_val, beta_val, dp_val

def sigma_tang(eps_tensor):
    N_elas = as_3D_tensor(n_elas)
    plastic_tang = (
        elastic_behavior(eps_tensor, lmbda_plas, mu_plas)
        - 3 * mu_plas * (3 * mu_plas / (3 * mu_plas + H) - beta) * ufl.inner(N_elas, eps_tensor) * N_elas
        - 2 * mu_plas * beta * ufl.dev(eps_tensor)
    )
    elastic_tang = elastic_behavior(eps_tensor, lmbda_elas, mu_elas)
    
    is_plastic = ufl.eq(material_indicator, 1.0)
    return ufl.conditional(is_plastic, plastic_tang, elastic_tang)

# ==========================================
# Variational Form & Assembly Setup
# ==========================================
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

residual_form_compiled = fem.form(-Residual)
tangent_form_compiled = fem.form(tangent_form)

A_tang = fem.petsc.create_matrix(tangent_form_compiled)
b_res = fem.petsc.create_vector(V)

solver = PETSc.KSP().create(domain.comm)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
solver.getPC().setFactorSolverType("mumps")
solver.setOperators(A_tang)

Nitermax, tol = 200, 1e-6
Nincr = 20
load_steps = np.linspace(0, 1.1, Nincr + 1)[1:] ** 0.5
results = np.zeros((Nincr + 1, 3))

functions_to_zero = [sig, sig_old, p, u, n_elas, beta]
for f in functions_to_zero:
    f.x.petsc_vec.set(0.0)
    f.x.scatter_forward()

# ==========================================
# Solution Loop
# ==========================================
for i, t in enumerate(load_steps):
    loading.value = t * q_lim
    
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
        A_tang.zeroEntries()
        fem.petsc.assemble_matrix(A_tang, tangent_form_compiled, bcs=bcs)
        A_tang.assemble()
        
        solver.solve(b_res, du.x.petsc_vec)
        du.x.scatter_forward()
        
        Du.x.petsc_vec.axpy(1.0, du.x.petsc_vec)
        Du.x.scatter_forward()
        
        Δε = eps(Du)
        sig_, n_elas_, beta_, dp_ = constitutive_update(Δε, sig_old, p)
        
        interpolate_quadrature(sig_, sig)
        interpolate_quadrature(n_elas_, n_elas)
        interpolate_quadrature(beta_, beta)
        
        with b_res.localForm() as b_loc:
            b_loc.set(0.0)
        fem.petsc.assemble_vector(b_res, residual_form_compiled)
        
        fem.petsc.apply_lifting(b_res, [tangent_form_compiled], bcs=[bcs], x0=[du.x.petsc_vec])
        b_res.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        fem.petsc.set_bc(b_res, bcs, du.x.petsc_vec)
        
        nRes = b_res.norm()
        niter += 1

    u.x.petsc_vec.axpy(1.0, Du.x.petsc_vec)
    u.x.scatter_forward()
    
    interpolate_quadrature(dp_, dp)
    p.x.petsc_vec.axpy(1.0, dp.x.petsc_vec)
    p.x.scatter_forward()
    
    sig_old.x.array[:] = sig.x.array[:]
    sig_old.x.scatter_forward()
    
    if len(bottom_inside_dof) > 0:
        results[i + 1, :] = (u.x.array[bottom_inside_dof[0]], t, niter)

# ==========================================
# Post-Processing Plots
# ==========================================
if len(bottom_inside_dof) > 0:
    plt.figure()
    plt.plot(results[:, 0], results[:, 1], "-oC3")
    plt.xlabel("Displacement of inner boundary")
    plt.ylabel(r"Applied pressure $q/q_{lim}$")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.bar(np.arange(Nincr + 1), results[:, 2], color="C2")
    plt.xlabel("Loading step")
    plt.ylabel("Number of iterations")
    plt.xlim(0)
    plt.grid(True)
    plt.show()