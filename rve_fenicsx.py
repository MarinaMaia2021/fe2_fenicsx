import numpy as np
import matplotlib.pyplot as plt

import gmsh
from mpi4py import MPI
import ufl
import basix
from dolfinx import mesh, fem, io
import dolfinx.fem.petsc
import dolfinx_mpc.utils
from dolfinx_mpc import LinearProblem
from petsc4py import PETSc

Lx = 1.0
Ly = 1 #np.sqrt(3) / 2.0 * Lx
c = 0.5 * Lx
R = 0.15 * Lx
h = 0.08 * Lx

corners = np.array([[0.0, 0.0], [Lx, 0.0], [Lx + c, Ly], [c, Ly]])
corners = np.array([[0.0, 0.0], [Lx, 0.0], [Lx, Ly], [0, Ly]])
a1 = corners[1, :] - corners[0, :]  # first vector generating periodicity
a2 = corners[3, :] - corners[0, :]  # second vector generating periodicity
fibers_center = np.vstack([corners, np.array([0.5, 0.5])])
                                             #[1, .5],
                                          #   [0, .5]])])
# -

# The geometry is then generated using `gmsh` Python API and the Open Cascade kernel. We tag the matrix with tag `1` and the inclusions with tag `2`. The bottom, right, top and left boundaries are respectively tagged `1, 2, 3, 4`.

# + tags=["hide-input", "hide-output"]
gdim = 2  # domain geometry dimension
fdim = 1  # facets dimension
gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if model_rank == 0:
    points = [occ.add_point(*corner, 0) for corner in fibers_center]
    lines = [occ.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    loop = occ.add_curve_loop(lines)
    unit_cell = occ.add_plane_surface([loop])
    inclusions = [occ.add_disk(*corner, 0, R, R) for corner in corners]
    vol_dimTag = (gdim, unit_cell)
    out = occ.intersect(
        [vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False
    )
    incl_dimTags = out[0]
    occ.synchronize()
    occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
    occ.synchronize()

    # tag physical domains and facets
    gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
    gmsh.model.addPhysicalGroup(
        gdim,
        [tag for _, tag in incl_dimTags],
        2,
        name="Inclusions",
    )
    gmsh.model.addPhysicalGroup(fdim, [7, 20, 10], 1, name="bottom")
    gmsh.model.addPhysicalGroup(fdim, [9, 19, 16], 2, name="right")
    gmsh.model.addPhysicalGroup(fdim, [15, 18, 12], 3, name="top")
    gmsh.model.addPhysicalGroup(fdim, [11, 17, 5], 4, name="left")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)

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
Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
y = ufl.SpatialCoordinate(domain)

deg_u = 2
shape = (gdim,)
V = fem.functionspace(domain, ("P", deg_u, shape))

Vx, _ = V.sub(0).collapse()
Vy, _ = V.sub(1).collapse()

# ### Periodic boundary conditions enforcement using `dolfinx_mpc`
#
# We must now define the periodic boundary conditions for the fluctuation field. For that, we make use of `dolfinx_mpc` providing a `MultiPointConstraint` object which can account for periodicity conditions. Note that periodic conditions do not fix rigid body translations. To remove them we choose here, for simplicity, to fix the displacement of the single point of coordinate `(0, 0)`. An alternative can consist of introducing constant Lagrange multipliers as discussed in the legacy demo. This solution requires however to use `Real` elements which require special care and are currently available in the `scifem` package https://github.com/scientificcomputing/scifem.

point_dof = fem.locate_dofs_geometrical(
    V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0)
)
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, V)]

n = ufl.FacetNormal(domain)
q_lim = float(2 / np.sqrt(3) * sig0)

loading = fem.Constant(domain, 0.0)

# We first instantiate the `MultiPointConstraint` object `mpc` defined with respect to our function space `V`. The function `create_periodic_constraint_topological` enables to link dofs on corresponding surfaces. We must first define a mapping between points on corresponding surfaces. For instance, we apply the first condition to the right surface, tagged `2`. The `periodic_relation_left_right` transforms input coordinates on the right surface to points on the left surface as follows: $(x, y) \mapsto (x-L_x, y)$. More generally, the mapping is $(x, y) \mapsto (x-a_{1x}, y-a_{1y})$ for the first periodicity-generating base vector $\ba_1$. We do the same for the top surface, tagged `3`, which is mapped to th
# e bottom one using the second periodicity-generating base vector $\ba_2.$

# +
def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a1[0]
    out_x[1] = x[1] - a1[1]
    out_x[2] = x[2]
    return out_x


def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a2[0]
    out_x[1] = x[1] - a2[1]
    out_x[2] = x[2]
    return out_x


mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(
    V, facets, 2, periodic_relation_left_right, bcs
)
mpc.create_periodic_constraint_topological(
    V, facets, 3, periodic_relation_bottom_top, bcs
)
mpc.finalize()
# -

u = fem.Function(mpc.function_space, name="Displacement")
v = fem.Function(mpc.function_space, name="Periodic_fluctuation")

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

print(loading, n)
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

Nitermax, tol = 10, 1e-6
Nincr = 10
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
    print(loading.value)
    
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
        print(nRes, nRes0, niter)
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
    
#     results[i + 1, :] = (u.x.array[bottom_inside_dof[0]], t, niter)

