# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, io, geometry
import dolfinx.fem.petsc
from dolfinx.io.gmsh import model_to_mesh
import dolfinx_mpc.utils
from dolfinx_mpc import NonlinearProblem
import matplotlib.pyplot as plt

Lx = 1.0
Ly = 1.0 
c = 0.5 * Lx
R = 0.2 * Lx
h = 0.1 * Lx

corners = np.array([[0.0, 0.0], [Lx, 0.0], [Lx, Ly], [0, Ly]])
a1 = corners[1, :] - corners[0, :]  
a2 = corners[3, :] - corners[0, :]  
fibers_center = np.vstack([corners, np.array([0.4, 0.25])])

# --- Mesh Generation ---
gdim = 2  
fdim = 1  
gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if model_rank == 0:
    points = [occ.add_point(*corner, 0) for corner in corners]
    lines = [occ.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    loop = occ.add_curve_loop(lines)
    unit_cell = occ.add_plane_surface([loop])
    inclusions = [occ.add_disk(*corner, 0, R, R) for corner in fibers_center]
    vol_dimTag = (gdim, unit_cell)
    
    out = occ.intersect(
        [vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False
    )
    incl_dimTags = out[0]
    occ.synchronize()
    occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
    occ.synchronize()

    # Dynamic Geometric Boundary Identification
    # Pull all boundary fragments based on their exact spatial locations
    bottom_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, Lx + 0.01, 0.01, 0.01, fdim)
    right_edges  = gmsh.model.getEntitiesInBoundingBox(Lx - 0.01, -0.01, -0.01, Lx + 0.01, Ly + 0.01, 0.01, fdim)
    top_edges    = gmsh.model.getEntitiesInBoundingBox(-0.01, Ly - 0.01, -0.01, Lx + 0.01, Ly + 0.01, 0.01, fdim)
    left_edges   = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, 0.01, Ly + 0.01, 0.01, fdim)

    bottom_tags = [tag for _, tag in bottom_edges]
    right_tags  = [tag for _, tag in right_edges]
    top_tags    = [tag for _, tag in top_edges]
    left_tags   = [tag for _, tag in left_edges]

    # --- FORCES TRUE MESH PERIODICITY PATTERNS IN GMSH ---
    # Translate left mesh nodes to the right side
    translation_right = [1, 0, 0, Lx,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    for l_tag, r_tag in zip(left_tags, right_tags):
        gmsh.model.mesh.setPeriodic(fdim, [r_tag], [l_tag], translation_right)

    # Translate bottom mesh nodes to the top side
    translation_top = [1, 0, 0, 0,  0, 1, 0, Ly,  0, 0, 1, 0,  0, 0, 0, 1]
    for b_tag, t_tag in zip(bottom_tags, top_tags):
        gmsh.model.mesh.setPeriodic(fdim, [t_tag], [b_tag], translation_top)

    # Assign Physical Groups safely using the extracted tags
    gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
    gmsh.model.addPhysicalGroup(gdim, [tag for _, tag in incl_dimTags], 2, name="Inclusions")
    gmsh.model.addPhysicalGroup(fdim, bottom_tags, 1, name="bottom")
    gmsh.model.addPhysicalGroup(fdim, right_tags, 2, name="right")
    gmsh.model.addPhysicalGroup(fdim, top_tags, 3, name="top")
    gmsh.model.addPhysicalGroup(fdim, left_tags, 4, name="left")
    
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(gdim)

mesh_data = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
domain = mesh_data.mesh
cells = mesh_data.cell_tags
facets = mesh_data.facet_tags
gmsh.finalize()

vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))
print("Volume:", vol)

# --- Material Fields & Constants ---
def create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k

E_field = create_piecewise_constant_field(domain, cells, {1: 8000.0, 2: 20000.0}, name="YoungModulus")
nu_field = create_piecewise_constant_field(domain, cells, {1: 0.3, 2: 0.25}, name="PoissonRatio")

# 2. Calibrated Elasto-Plastic Matrix Parameters
sig0 = fem.Constant(domain, 80.0)         # Matrix yields precisely at 80 MPa
H_mod = fem.Constant(domain, 1200.)    # Linear plastic hardening slope
eta = fem.Constant(domain, 10.)           # Dropped significantly to eliminate rate-dependent overstress
n_val = fem.Constant(domain, 1.0)          # Keep linear rate scaling
dt = fem.Constant(domain, 1.0)             # Time step length

# Create a phase indicator/mask field (1.0 = Matrix, 0.0 = Fibers)
phase_mask = create_piecewise_constant_field(domain, cells, {1: 1.0, 2: 0.0}, name="PhaseIndicator")

# Plane Stress Elastic Stiffness Components
C11 = E_field / (1.0 - nu_field**2)
C12 = nu_field * E_field / (1.0 - nu_field**2)
C33 = E_field / (2.0 * (1.0 + nu_field))

# --- Internal State Variables (DP0/Discontinuous Lagrange Space) ---
QuadratureSpace_Scalar = fem.functionspace(domain, ("DP", 0))
QuadratureSpace_Tensor = fem.functionspace(domain, ("DP", 0, (3,)))

alpha_old = fem.Function(QuadratureSpace_Scalar, name="AccumulatedPlasticStrain_Old")
ep_old = fem.Function(QuadratureSpace_Tensor, name="PlasticStrainTensor_Old")

# --- Variational & Constitutive Mechanics ---
Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
y = ufl.SpatialCoordinate(domain)

def epsilon(v):
    return ufl.sym(ufl.grad(v))

def voigt_strain(v):
    """Returns local total strain vector [e_xx, e_yy, 2*e_xy] including macro strain."""
    total_eps = Eps + epsilon(v)
    return ufl.as_vector([total_eps[0, 0], total_eps[1, 1], 2.0 * total_eps[0, 1]])

def plane_stress_constitutive_update(v, ep_old_vec, alpha_old_val):
    """
    Performs a radial-return projection for rate-independent classical plasticity 
    exclusively on the matrix, preserving purely elastic behavior in the fibers.
    """
    eps_total = voigt_strain(v)
    
    # Elastic trial stress prediction
    eps_elastic_trial = eps_total - ep_old_vec
    s_trial_xx = C11 * eps_elastic_trial[0] + C12 * eps_elastic_trial[1]
    s_trial_yy = C12 * eps_elastic_trial[0] + C11 * eps_elastic_trial[1]
    s_trial_xy = C33 * eps_elastic_trial[2]
    
    # Deviatoric stress projector for Plane Stress
    mean_trial = (s_trial_xx + s_trial_yy) / 3.0
    dev_xx = s_trial_xx - mean_trial
    dev_yy = s_trial_yy - mean_trial
    dev_zz = -mean_trial
    
    # Von Mises equivalent trial stress
    sigma_eq_trial = ufl.sqrt(1.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2 + 2.0 * s_trial_xy**2))
    
    # Current flow limit definition with isotropic linear hardening
    current_yield_limit = sig0 + H_mod * alpha_old_val
    yield_function = sigma_eq_trial - current_yield_limit
    
    # --- Rate-Independent Closed-Form Return Mapping ---
    # Shear modulus (needed for plastic projection stiffness denominator)
    mu = C33 
    
    # Exact plastic multiplier required to pull stress back onto the yield surface
    # Mask out plastic flow increment inside fibers (where phase_mask == 0) via multiplication
    delta_gamma_elasto_plastic = phase_mask * (yield_function / (3.0 * mu + H_mod))
    
    # Apply conditional: Only activate plastic increment if the yield surface is breached
    delta_gamma = ufl.conditional(ufl.gt(yield_function, 0.0), delta_gamma_elasto_plastic, 0.0)
    
    # Plastic flow normals
    normal_xx = 1.5 * dev_xx / (sigma_eq_trial + 1e-10)
    normal_yy = 1.5 * dev_yy / (sigma_eq_trial + 1e-10)
    normal_xy = 3.0 * s_trial_xy / (sigma_eq_trial + 1e-10)
    
    # State update applications (Corrected plastic relaxation terms)
    sigma_xx = s_trial_xx - (C11 * normal_xx + C12 * normal_yy) * delta_gamma
    sigma_yy = s_trial_yy - (C12 * normal_xx + C11 * normal_yy) * delta_gamma
    sigma_xy = s_trial_xy - C33 * normal_xy * delta_gamma
    
    alpha_new = alpha_old_val + delta_gamma
    ep_new_xx = ep_old_vec[0] + normal_xx * delta_gamma
    ep_new_yy = ep_old_vec[1] + normal_yy * delta_gamma
    ep_new_xy = ep_old_vec[2] + normal_xy * delta_gamma
    
    stress_vector = ufl.as_vector([sigma_xx, sigma_yy, sigma_xy])
    ep_vector_new = ufl.as_vector([ep_new_xx, ep_new_yy, ep_new_xy])
        
    return stress_vector, ep_vector_new, alpha_new

# Define Unknowns
V = fem.functionspace(domain, ("P", 2, (gdim,)))
u_ = ufl.TestFunction(V)
dv = ufl.TrialFunction(V) 

# Map stress vector back to standard symmetric 2D UFL Tensor format
def stress_vector_to_tensor(s_vec):
    return ufl.as_tensor([[s_vec[0], s_vec[2]], [s_vec[2], s_vec[1]]])

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

# --- Boundary Conditions & MPC ---
point_dof = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0))
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, V)]

mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(V, facets, 2, periodic_relation_left_right, bcs)
mpc.create_periodic_constraint_topological(V, facets, 3, periodic_relation_bottom_top, bcs)
mpc.finalize()

# Wrap unknowns into MPC-reduced solution space
v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
u = fem.Function(mpc.function_space, name="Displacement")

# Evaluate Stress and History Variables
stress_vec, ep_vec_next, alpha_next = plane_stress_constitutive_update(v, ep_old, alpha_old)
sigma_tensor = stress_vector_to_tensor(stress_vec)

# Residual Form
F_form = ufl.inner(sigma_tensor, epsilon(u_)) * ufl.dx

# Jacobian Form (Algorithmic tangent automated via UFL auto-differentiation)
dv = ufl.TrialFunction(V)
J_form = ufl.derivative(F_form, v, dv)

# Solver Setup 
problem = NonlinearProblem(F_form, v, mpc, bcs=bcs, J=J_form, petsc_options={
    "snes_type": "newtonls",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu"
})

# --- Projection expressions using local space elements explicitly ---
alpha_expr = fem.Expression(alpha_next, QuadratureSpace_Scalar.element.interpolation_points)
ep_expr = fem.Expression(ep_vec_next, QuadratureSpace_Tensor.element.interpolation_points)

def update_history_variables():
    """Safely projects updated history states from current converged step down to memory tracks."""
    alpha_old.interpolate(alpha_expr)
    ep_old.interpolate(ep_expr)
    
# --- FIXED: Coordinate Cell Evaluation Point Setup ---
target_point = np.array([0.5, 0.05, 0.0])  # Shifted to safely target an element inside the matrix phase
bb_tree = geometry.bb_tree(domain, domain.topology.dim)  # Fixed syntax error
cell_candidates = geometry.compute_collisions_points(bb_tree, target_point)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, target_point)

# Retrieve local target cell index (using first match found on this processor)
if len(colliding_cells.links(0)) > 0:
    target_cell_idx = colliding_cells.links(0)[0]
    print(f"Tracking closest element index: {target_cell_idx}")
else:
    raise RuntimeError("Target point is outside the generated domain mesh topology bounds.")

# Setup explicit state tracking arrays
history_local_strain_xx = []
history_local_stress_xx = []

# Projector structures to extract local Voigt tensor evaluations to plot arrays
W_DG0 = fem.functionspace(domain, ("DG", 0))
local_strain_field = fem.Function(W_DG0)
local_stress_field = fem.Function(W_DG0)

strain_xx_expr = fem.Expression(voigt_strain(v)[0], W_DG0.element.interpolation_points)
stress_xx_expr = fem.Expression(sigma_tensor[0, 0], W_DG0.element.interpolation_points)    

# --- Visualization Setup ---
import pyvista
from dolfinx import plot
pyvista.set_jupyter_backend("static")

W0 = fem.functionspace(domain, ("DG", 0))

def build_deformation_meshes(u, v, scale=1.0):
    topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
    undeformed = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    undeformed.cell_data["material"] = cells.values

    disp = np.zeros((geometry.shape[0], 3))
    disp[:, :2] = u.x.array.reshape(-1, 2)

    # Re-evaluate current active stress tensor field for visualization
    stress_expr = fem.Expression(sigma_tensor[0, 0], W0.element.interpolation_points)
    sigma_xx = fem.Function(W0)
    sigma_xx.interpolate(stress_expr)
    
    deformed = pyvista.UnstructuredGrid(topology, cell_types, geometry.copy())
    deformed.point_data["u"] = disp
    deformed.set_active_vectors("u")
    deformed.cell_data["stress"] = sigma_xx.x.array
    warped = deformed.warp_by_vector("u", factor=scale)
    return undeformed, warped

# Monotonic macroscopic Loading profile
macro_strains = [
    np.array([[0.000, 0.00], [0.00, 0.00]]),
    np.array([[0.002, 0.00], [0.00, -0.0006]]),
    np.array([[0.004, 0.00], [0.00, -0.0012]]),
    np.array([[0.006, 0.00], [0.00, -0.0018]]),
    np.array([[0.008, 0.00], [0.00, -0.0024]]),
    np.array([[0.010, 0.00], [0.00, -0.0030]]),  # Around here it hits 80 MPa and yields
    np.array([[0.012, 0.00], [0.00, -0.0036]]),
    np.array([[0.014, 0.00], [0.00, -0.0042]]),
    np.array([[0.016, 0.00], [0.00, -0.0048]]),
    np.array([[0.018, 0.00], [0.00, -0.0054]]),
    np.array([[0.020, 0.00], [0.00, -0.0060]])   
]

plotter = pyvista.Plotter(off_screen=True)
plotter.open_gif("deformation_viscoplastic.gif")

xmin, ymin = 0.0, 0.0
xmax, ymax = np.max(corners[:, 0]), np.max(corners[:, 1])
margin = 0.15 * max(xmax - xmin, ymax - ymin)
fixed_bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin, -1, 1)

# --- Incremental Time-Loading Loop ---
for iStep, strain in enumerate(macro_strains):
    print(f"\n--- Time Step {iStep} ---")
    Eps.value = strain

    # Execute return-mapping optimization inside global constraint framework
    problem.solve()
    
    v_array = v.x.array
    print(f"Fluctuation Field v -> Max Magnitude: {np.max(np.abs(v_array)):.6e}, Mean: {np.mean(v_array):.6e}")
    
    # Store converged integration point internal states for the next time-step
    update_history_variables()
    
    # Build full displacement field: u = E.y + v
    u.interpolate(
        fem.Expression(
            ufl.dot(Eps, y),
            mpc.function_space.element.interpolation_points
        )
    )
    u.x.array[:] += v.x.array[:]
    
    # --- LIVE DEBUG PRINT STATEMENTS ---
    # We query the numerical values from our tracking fields here
    avg_alpha = np.mean(alpha_old.x.array)
    max_alpha = np.max(alpha_old.x.array)
    
    # The plastic strain tensor has 3 components per cell [ep_xx, ep_yy, ep_xy]
    ep_array_reshaped = ep_old.x.array.reshape(-1, 3)
    max_ep_xx = np.max(np.abs(ep_array_reshaped[:, 0]))
    
    print(f"DEBUG [Global]: Mean Accum. Plastic Strain = {avg_alpha:.6e}")
    print(f"DEBUG [Global]: Max Accum. Plastic Strain  = {max_alpha:.6e}")
  #  print(f"DEBUG [Global]: Max Local Plastic Strain xx = {max_ep_xx:.6e}")
    
    # Extract and record point history data values 
    local_strain_field.interpolate(strain_xx_expr)
    local_stress_field.interpolate(stress_xx_expr)
    
    tracked_strain_val = local_strain_field.x.array[target_cell_idx]
    tracked_stress_val = local_stress_field.x.array[target_cell_idx]
    tracked_alpha_val = alpha_old.x.array[target_cell_idx]
    
    print(f"DEBUG [Tracked Element {target_cell_idx}]: Strain_xx = {tracked_strain_val:.5f}, Stress_xx = {tracked_stress_val:.2f} MPa, Alpha = {tracked_alpha_val:.6e}")
    
    history_local_strain_xx.append(tracked_strain_val)
    history_local_stress_xx.append(tracked_stress_val)
    
    # Process effective homogenized stresses \Sigma = <\sigma>
    Sigma = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            test_strain = np.zeros((2, 2))
            test_strain[i, j] = 1.0
            Eps_.value = test_strain
            Sigma[i, j] = fem.assemble_scalar(
                fem.form(ufl.inner(sigma_tensor, Eps_) * ufl.dx)
            ) / vol
            
    print("Macroscopic Stress Tensor \Sigma:")
    print(Sigma)
       
    # Total displacement projection
    u.interpolate(
        fem.Expression(
            ufl.dot(Eps, y),
            mpc.function_space.element.interpolation_points
        )
    )
    u.x.array[:] += v.x.array[:]
    
    undeformed, warped = build_deformation_meshes(u, v, scale=5.0)
    
    plotter.clear()
    plotter.add_mesh(
        warped, scalars="stress", cmap="coolwarm",
        show_edges=False, line_width=0.25, # clim=[0, 350],
    )
    
    plotter.add_mesh(
        undeformed, scalars="material", categories=True,
        cmap=["lightgray", "gray"], style="wireframe", line_width=.5,
        edge_opacity=0.5, show_edges=True
    )
    plotter.view_xy()
    plotter.camera.SetParallelProjection(True)
    plotter.reset_camera(bounds=fixed_bounds)
    plotter.add_text(f"Step {iStep}", font_size=12)
    plotter.show()
    plotter.write_frame()

plotter.close()

# --- Generate and Render Stress-Strain Material Curve ---
plt.figure(figsize=(7, 5))
plt.plot(history_local_strain_xx, history_local_stress_xx, 'o-', color='crimson', linewidth=2, label="Element Local Response")
plt.xlabel(r"Local Strain $\varepsilon_{xx}$", fontsize=11)
plt.ylabel(r"Local Stress $\sigma_{xx}$ (MPa)", fontsize=11)
plt.title(f"Selected material point loading path", fontsize=12, fontweight='bold')
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.tight_layout()
plt.savefig("local_point_stress_strain.png", dpi=300)
plt.show()

# # ==============================================================================
# # RIGOROUS HISTORICAL DIAGNOSTICS FOR THE FIBER PHASE (TAG 2)
# # ==============================================================================
# # Locate all global indices belonging to inclusion cells (fibers)
# fiber_cell_indices = cells.find(2)

# # Extract history arrays belonging exclusively to the fibers
# fiber_alphas = alpha_old.x.array[fiber_cell_indices]
# # Plastic strain tensor has 3 values per cell, reshape to access [ep_xx, ep_yy, ep_xy]
# fiber_eps_tensor = ep_old.x.array.reshape(-1, 3)[fiber_cell_indices]

# max_fiber_alpha = np.max(fiber_alphas)
# mean_fiber_alpha = np.mean(fiber_alphas)
# max_fiber_ep_xx = np.max(np.abs(fiber_eps_tensor[:, 0]))

# print("\n=== VERIFICATION: INCLUSIONS / FIBERS PHASE (TAG 2) ===")
# print(f"Number of fiber cells evaluated: {len(fiber_cell_indices)}")
# print(f"Max alpha in fibers:    {max_fiber_alpha:.6e}")
# print(f"Mean alpha in fibers:   {mean_fiber_alpha:.6e}")
# print(f"Max |ep_xx| in fibers:  {max_fiber_ep_xx:.6e}")

# # Assert verification check
# if np.isclose(max_fiber_alpha, 0.0, atol=1e-12):
#     print("SUCCESS: Fibers are functioning as a purely elastic material!")
# else:
#     print("WARNING: Plastic strain leakage detected inside the fibers.")