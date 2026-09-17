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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  4 11:20:54 2026

@author: malvesmaia
"""

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
fibers_center = np.vstack([corners, 
                           np.array([[0.4, 0.25],
                                     [0.7, 0.6]])])

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

    bottom_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, Lx + 0.01, 0.01, 0.01, fdim)
    right_edges  = gmsh.model.getEntitiesInBoundingBox(Lx - 0.01, -0.01, -0.01, Lx + 0.01, Ly + 0.01, 0.01, fdim)
    top_edges    = gmsh.model.getEntitiesInBoundingBox(-0.01, Ly - 0.01, -0.01, Lx + 0.01, Ly + 0.01, 0.01, fdim)
    left_edges   = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, 0.01, Ly + 0.01, 0.01, fdim)

    bottom_tags = [tag for _, tag in bottom_edges]
    right_tags  = [tag for _, tag in right_edges]
    top_tags    = [tag for _, tag in top_edges]
    left_tags   = [tag for _, tag in left_edges]

    translation_right = [1, 0, 0, Lx,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
    for l_tag, r_tag in zip(left_tags, right_tags):
        gmsh.model.mesh.setPeriodic(fdim, [r_tag], [l_tag], translation_right)

    translation_top = [1, 0, 0, 0,  0, 1, 0, Ly,  0, 0, 1, 0,  0, 0, 0, 1]
    for b_tag, t_tag in zip(bottom_tags, top_tags):
        gmsh.model.mesh.setPeriodic(fdim, [t_tag], [b_tag], translation_top)

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
        cs = cell_markers.find(tag)
        k.x.array[cs] = np.full_like(cs, value, dtype=np.float64)
    return k

E_field = create_piecewise_constant_field(domain, cells, {1: 8000.0, 2: 20000.0}, name="YoungModulus")
nu_field = create_piecewise_constant_field(domain, cells, {1: 0.3, 2: 0.25}, name="PoissonRatio")

sig0 = fem.Constant(domain, 80.0)         
H_mod = fem.Constant(domain, 1200.)    
phase_mask = create_piecewise_constant_field(domain, cells, {1: 1.0, 2: 0.0}, name="PhaseIndicator")

C11 = E_field / (1.0 - nu_field**2)
C12 = nu_field * E_field / (1.0 - nu_field**2)
C33 = E_field / (2.0 * (1.0 + nu_field))

QuadratureSpace_Scalar = fem.functionspace(domain, ("DP", 0))
QuadratureSpace_Tensor = fem.functionspace(domain, ("DP", 0, (3,)))

alpha_old = fem.Function(QuadratureSpace_Scalar, name="AccumulatedPlasticStrain_Old")
ep_old = fem.Function(QuadratureSpace_Tensor, name="PlasticStrainTensor_Old")

# --- MACROSCOPIC PARAMETERS (Standard Constants, no basix bugs!) ---
E_xx_macro = fem.Constant(domain, 0.0)
E_yy_macro = fem.Constant(domain, 0.0)
E_xy_macro = fem.Constant(domain, 0.0)

Eps_ = fem.Constant(domain, np.zeros((2, 2)))
y = ufl.SpatialCoordinate(domain)

def epsilon(v_vec):
    return ufl.sym(ufl.grad(v_vec))

def voigt_strain(v_vec):
    """Combines background macro constants with local micro fluctuations."""
    Eps_macro = ufl.as_tensor([[E_xx_macro, E_xy_macro], 
                               [E_xy_macro, E_yy_macro]])
    total_eps = Eps_macro + epsilon(v_vec)
    return ufl.as_vector([total_eps[0, 0], total_eps[1, 1], 2.0 * total_eps[0, 1]])

def plane_stress_constitutive_update(v_vec, ep_old_vec, alpha_old_val):
    eps_total = voigt_strain(v_vec)
    
    eps_elastic_trial = eps_total - ep_old_vec
    s_trial_xx = C11 * eps_elastic_trial[0] + C12 * eps_elastic_trial[1]
    s_trial_yy = C12 * eps_elastic_trial[0] + C11 * eps_elastic_trial[1]
    s_trial_xy = C33 * eps_elastic_trial[2]
    
    mean_trial = (s_trial_xx + s_trial_yy) / 3.0
    dev_xx = s_trial_xx - mean_trial
    dev_yy = s_trial_yy - mean_trial
    dev_zz = -mean_trial
    
    sigma_eq_trial = ufl.sqrt(1.5 * (dev_xx**2 + dev_yy**2 + dev_zz**2 + 2.0 * s_trial_xy**2))
    current_yield_limit = sig0 + H_mod * alpha_old_val
    yield_function = sigma_eq_trial - current_yield_limit
    
    mu = C33 
    delta_gamma_elasto_plastic = phase_mask * (yield_function / (3.0 * mu + H_mod))
    delta_gamma = ufl.conditional(ufl.gt(yield_function, 0.0), delta_gamma_elasto_plastic, 0.0)
    
    normal_xx = 1.5 * dev_xx / (sigma_eq_trial + 1e-10)
    normal_yy = 1.5 * dev_yy / (sigma_eq_trial + 1e-10)
    normal_xy = 3.0 * s_trial_xy / (sigma_eq_trial + 1e-10)
    
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

# Standard vector space for periodic fluctuations v
V = fem.functionspace(domain, ("P", 2, (gdim,)))
u_ = ufl.TestFunction(V)

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

# --- Boundary Conditions assigned to Subspace index 0 (v component) ---
def corner_00(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0)

def corner_01(x):
    return np.isclose(x[0], 0.0) & np.isclose(x[1], Ly)

dof_00 = fem.locate_dofs_geometrical(V, corner_00)
bcs = [fem.dirichletbc(np.array([0.0, 0.0], dtype=np.float64), dof_00, V)]

# Eliminate rigid body rotation on corner 01
V_x, _ = V.sub(0).collapse()
dof_01_x, _ = fem.locate_dofs_geometrical((V.sub(0), V_x), corner_01)
bcs.append(fem.dirichletbc(fem.Constant(domain, 0.0), dof_01_x, V_x))

# Generate standard topological Periodic Constraints
mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(V, facets, 2, periodic_relation_left_right, bcs)
mpc.create_periodic_constraint_topological(V, facets, 3, periodic_relation_bottom_top, bcs)
mpc.finalize()

v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
u = fem.Function(mpc.function_space, name="Displacement")

# Evaluate Stress State Expressions
stress_vec, ep_vec_next, alpha_next = plane_stress_constitutive_update(v, ep_old, alpha_old)
def stress_vector_to_tensor(s_vec):
    return ufl.as_tensor([[s_vec[0], s_vec[2]], [s_vec[2], s_vec[1]]])
sigma_tensor = stress_vector_to_tensor(stress_vec)

F_form = ufl.inner(sigma_tensor, epsilon(u_)) * ufl.dx

dv = ufl.TrialFunction(V)
J_form = ufl.derivative(F_form, v, dv)

# Solver Layout
problem = NonlinearProblem(F_form, v, mpc, bcs=bcs, J=J_form, petsc_options={
    "snes_type": "newtonls",
    "snes_atol": 1e-6,
    "snes_rtol": 1e-6,
    "snes_max_it": 50,
    "ksp_type": "preonly",
    "pc_type": "lu"
})

alpha_expr = fem.Expression(alpha_next, QuadratureSpace_Scalar.element.interpolation_points)
ep_expr = fem.Expression(ep_vec_next, QuadratureSpace_Tensor.element.interpolation_points)

def update_history_variables():
    alpha_old.interpolate(alpha_expr)
    ep_old.interpolate(ep_expr)
    
target_point = np.array([0.5, 0.5, 0.0])
bb_tree = geometry.bb_tree(domain, domain.topology.dim)
cell_candidates = geometry.compute_collisions_points(bb_tree, target_point)
colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, target_point)

if len(colliding_cells.links(0)) > 0:
    target_cell_idx = colliding_cells.links(0)[0]
else:
    raise RuntimeError("Target point outside bounds.")

history_local_strain_xx = []
history_local_stress_xx = []

W_DG0 = fem.functionspace(domain, ("DG", 0))
local_strain_field = fem.Function(W_DG0)
local_stress_field = fem.Function(W_DG0)

strain_xx_expr = fem.Expression(voigt_strain(v)[0], W_DG0.element.interpolation_points)
stress_xx_expr = fem.Expression(sigma_tensor[0, 0], W_DG0.element.interpolation_points)    

import pyvista
from dolfinx import plot
pyvista.set_jupyter_backend("static")

W0 = fem.functionspace(domain, ("DG", 0))

def build_deformation_meshes(u_field, scale=1.0):
    topology, cell_types, geom = plot.vtk_mesh(u_field.function_space)
    undeformed = pyvista.UnstructuredGrid(topology, cell_types, geom)
    undeformed.cell_data["material"] = cells.values

    disp = np.zeros((geom.shape[0], 3))
    disp[:, :2] = u_field.x.array.reshape(-1, 2)

    stress_expr = fem.Expression(sigma_tensor[0, 0], W0.element.interpolation_points)
    sigma_xx = fem.Function(W0)
    sigma_xx.interpolate(stress_expr)
    
    deformed = pyvista.UnstructuredGrid(topology, cell_types, geom.copy())
    deformed.point_data["u"] = disp
    deformed.set_active_vectors("u")
    deformed.cell_data["stress"] = sigma_xx.x.array
    warped = deformed.warp_by_vector("u", factor=scale)
    return undeformed, warped

plotter = pyvista.Plotter(off_screen=True)
plotter.open_gif("deformation_viscoplastic.gif")

xmin, ymin = 0.0, 0.0
xmax, ymax = np.max(corners[:, 0]), np.max(corners[:, 1])
margin = 0.15 * max(xmax - xmin, ymax - ymin)
fixed_bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin, -1, 1)

# --- HELPER FUNCTIONS FOR GLOBAL UNCOUPLED MACRO RELAXATION ---
def get_macro_stresses():
    """Integrates the current local stress tensor to output the 2x2 homogenized macro stress matrix."""
    Sigma_out = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            test_strain = np.zeros((2, 2))
            test_strain[i, j] = 1.0
            Eps_.value = test_strain
            Sigma_out[i, j] = fem.assemble_scalar(fem.form(ufl.inner(sigma_tensor, Eps_) * ufl.dx)) / vol
    return Sigma_out

def generate_cyclic_loading_profile(steps_per_phase=10, hold_steps=5):
    """
    Generates a piecewise linear macro-displacement/strain profile:
    1. Load from 0.00 to 0.02
    2. Unload from 0.02 back to 0.01
    3. Reload from 0.01 up to 0.03
    4. Hold constant at 0.30 for a specified number of steps
    """
    # Phase 1: Load to 0.02
    phase1 = np.linspace(0.00, 0.02, steps_per_phase + 1)
    
    # Phase 2: Unload to 0.00 (omit first element to prevent duplicates)
    phase2 = np.linspace(0.02, 0.01, steps_per_phase + 1)[1:]
    
    # Phase 3: Reload to 0.40 (omit first element to prevent duplicates)
    phase3 = np.linspace(0.01, 0.03, steps_per_phase + 1)[1:]
    
    # Phase 4: Hold constant at 0.30
    phase4 = np.full(hold_steps, 0.03)
    
    # Concatenate all phases into a single seamless loading history
    full_profile = np.concatenate([phase1, phase2, phase3, phase4])
    
    return full_profile

# --- Example Usage ---
# Adjust the steps to make your simulation finer or coarser
#prescribed_displacements = generate_cyclic_loading_profile(steps_per_phase=15, hold_steps=8)
prescribed_displacements = np.linspace(0.00, 0.02, 10)

# Active tracking variables for Newton optimization of E_yy and E_xy
guess_E_yy = 0.0
guess_E_xy = 0.0

for iStep, current_strain in enumerate(prescribed_displacements):
    print(f"\n==================================================")
    print(f"--- Time Step {iStep} (Prescribed Macro E_xx = {current_strain:.4f}) ---")
    print(f"==================================================")
    
    E_xx_macro.value = current_strain
    
    # --- Native Secant/Newton Relaxation Loop for Macro Stresses ---
    # We iteratively adjust guess_E_yy and guess_E_xy until Sigma_yy and Sigma_xy go to 0.0
    relax_converged = False
    for iRelax in range(3):
        E_yy_macro.value = guess_E_yy
        E_xy_macro.value = guess_E_xy
        
        problem.solve()
        Sigma = get_macro_stresses()
        
        err_yy = Sigma[1, 1]
        err_xy = Sigma[0, 1]
        
        if max(abs(err_yy), abs(err_xy)) < 1e-4:
            relax_converged = True
            break
            
        # Numerical Tangent Estimation to compute the macroscopic numerical compliance matrix
        perturb = 1e-5
        
        E_yy_macro.value = guess_E_yy + perturb
        problem.solve()
        Sigma_p_yy = get_macro_stresses()
        
        E_yy_macro.value = guess_E_yy
        E_xy_macro.value = guess_E_xy + perturb
        problem.solve()
        Sigma_p_xy = get_macro_stresses()
        
        # Build 2x2 macro tangent matrix dSigma / dE_macro
        dSig_dEyy = (Sigma_p_yy - Sigma) / perturb
        dSig_dExy = (Sigma_p_xy - Sigma) / perturb
        
        J_macro = np.array([
            [dSig_dEyy[1, 1], dSig_dExy[1, 1]],
            [dSig_dEyy[0, 1], dSig_dExy[0, 1]]
        ])
        
        # Newton correction step
        residuals = np.array([err_yy, err_xy])
        delta_E = np.linalg.solve(J_macro, -residuals)
        
        guess_E_yy += delta_E[0]
        guess_E_xy += delta_E[1]
        
    # Re-apply converged values and execute final lock step
    E_yy_macro.value = guess_E_yy
    E_xy_macro.value = guess_E_xy
    problem.solve()
    update_history_variables()
    
    print(f"-> Converged Floating Parameters: E_yy = {guess_E_yy:.6e}, E_xy = {guess_E_xy:.6e}")
    print("Final Macroscopic Stress Tensor \Sigma:")
    print(Sigma)
    
    # 3. Rebuild full total displacement field mapping for PyVista visualization
    Eps_macro_numeric = ufl.as_tensor([[E_xx_macro, E_xy_macro], [E_xy_macro, E_yy_macro]])
    u.interpolate(
        fem.Expression(
            ufl.dot(Eps_macro_numeric, y) + v,
            u.function_space.element.interpolation_points
        )
    )
    
    # --- Live Debug Computations ---
    avg_alpha = np.mean(alpha_old.x.array)
    print(f"DEBUG [Global]: Mean Accum. Plastic Strain = {avg_alpha:.6e}")
    
    local_strain_field.interpolate(strain_xx_expr)
    local_stress_field.interpolate(stress_xx_expr)
    
    tracked_strain_val = local_strain_field.x.array[target_cell_idx]
    tracked_stress_val = local_stress_field.x.array[target_cell_idx]
    
    print(f"DEBUG [Tracked Element {target_cell_idx}]: Strain_xx = {tracked_strain_val:.5f}, Stress_xx = {tracked_stress_val:.2f} MPa")
    
    history_local_strain_xx.append(tracked_strain_val)
    history_local_stress_xx.append(tracked_stress_val)
    
    undeformed, warped = build_deformation_meshes(u, scale=5.0)
    
    plotter.clear()
    plotter.add_mesh(warped, scalars="stress", cmap="coolwarm", 
                    # clim = [-140, 240],
                     show_edges=False, line_width=0.25)
    plotter.add_mesh(undeformed, scalars="material", categories=True, cmap=["lightgray", "gray"], style="wireframe", line_width=.5, edge_opacity=0.5, show_edges=True)
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

# ==============================================================================
#                 EXTENDED SENSITIVITY ANALYSIS SECTION (ADD TO END)
# ==============================================================================

def run_homogenized_simulation(E_matrix, E_inclusion, sig0_matrix):
    """
    Resets the internal state variables and runs the cyclic loading history
    with a specific set of material parameters, returning the final macro stress.
    """
    # 1. Reset history tracking variables to zero
    alpha_old.x.array[:] = 0.0
    ep_old.x.array[:] = 0.0
    v.x.array[:] = 0.0
    u.x.array[:] = 0.0
    
    # 2. Re-assign the targeted material fields dynamically
    # Update Young's Modulus mapping
    matrix_cells = cells.find(1)
    inclusion_cells = cells.find(2)
    E_field.x.array[matrix_cells] = np.full_like(matrix_cells, E_matrix, dtype=np.float64)
    E_field.x.array[inclusion_cells] = np.full_like(inclusion_cells, E_inclusion, dtype=np.float64)
    
    # Update recalculation terms dependent on E (Stiffness component arrays)
    # Note: Modern UFL forms naturally pull updated values from functions automatically
    
    # Update the Yield stress Constant value
    sig0.value = sig0_matrix
    
    # 3. Execute the loading loop sequence
    # (We can use a slightly coarser profile here to save computation time)
    evaluation_profile = generate_cyclic_loading_profile(steps_per_phase=5, hold_steps=2)
    
    # Active tracking variables for Newton optimization of E_yy and E_xy
    guess_E_yy = 0.0
    guess_E_xy = 0.0
    
    macro_strain_history = []
    macro_stress_history = []
    
    for iStep, current_strain in enumerate(prescribed_displacements):
        print(f"\n==================================================")
        print(f"--- Time Step {iStep} (Prescribed Macro E_xx = {current_strain:.4f}) ---")
        print(f"==================================================")
        
        E_xx_macro.value = current_strain
        
        # --- Native Secant/Newton Relaxation Loop for Macro Stresses ---
        # We iteratively adjust guess_E_yy and guess_E_xy until Sigma_yy and Sigma_xy go to 0.0
        relax_converged = False
        for iRelax in range(3):
            E_yy_macro.value = guess_E_yy
            E_xy_macro.value = guess_E_xy
            
            problem.solve()
            Sigma = get_macro_stresses()
            
            err_yy = Sigma[1, 1]
            err_xy = Sigma[0, 1]
            
            if max(abs(err_yy), abs(err_xy)) < 1e-4:
                relax_converged = True
                break
                
            # Numerical Tangent Estimation to compute the macroscopic numerical compliance matrix
            perturb = 1e-5
            
            E_yy_macro.value = guess_E_yy + perturb
            problem.solve()
            Sigma_p_yy = get_macro_stresses()
            
            E_yy_macro.value = guess_E_yy
            E_xy_macro.value = guess_E_xy + perturb
            problem.solve()
            Sigma_p_xy = get_macro_stresses()
            
            # Build 2x2 macro tangent matrix dSigma / dE_macro
            dSig_dEyy = (Sigma_p_yy - Sigma) / perturb
            dSig_dExy = (Sigma_p_xy - Sigma) / perturb
            
            J_macro = np.array([
                [dSig_dEyy[1, 1], dSig_dExy[1, 1]],
                [dSig_dEyy[0, 1], dSig_dExy[0, 1]]
            ])
            
            # Newton correction step
            residuals = np.array([err_yy, err_xy])
            delta_E = np.linalg.solve(J_macro, -residuals)
            
            guess_E_yy += delta_E[0]
            guess_E_xy += delta_E[1]
            
        # Re-apply converged values and execute final lock step
        E_yy_macro.value = guess_E_yy
        E_xy_macro.value = guess_E_xy
        problem.solve()
        update_history_variables()
        
        print(f"-> Converged Floating Parameters: E_yy = {guess_E_yy:.6e}, E_xy = {guess_E_xy:.6e}")
        print("Final Macroscopic Stress Tensor \Sigma:")
        print(Sigma)
        
        macro_strain_history.append(current_strain)
        macro_stress_history.append(Sigma[0, 0])
    
    return np.array(macro_strain_history), np.array(macro_stress_history)

# --- Compute Baseline Reference Stress ---
print("\n--- Running Baseline Sensitivity Reference Simulation ---")
E_m_base = 8000.0
E_i_base = 20000.0
sig0_base = 80.0

# --- Define Perturbation Fraction (e.g., 1%) ---
delta_fraction = 0.01

print("Running Baseline Reference...")
strains_base, stresses_base = run_homogenized_simulation(E_m_base, E_i_base, sig0_base)

# 2. Evaluate Matrix Stiffness Variation (+15%)
print("Running Matrix Stiffness Variation...")
E_m_perturbed = E_m_base * (1.0 + delta_fraction)
_, stresses_Em_high = run_homogenized_simulation(E_m_perturbed, E_i_base, sig0_base)
delta_Em = E_m_perturbed - E_m_base
dSxx_dEm = (stresses_Em_high[-1] - stresses_base[-1]) / delta_Em

# 3. Evaluate Inclusion Stiffness Variation (+15%)
print("Running Inclusion Stiffness Variation...")
E_i_perturbed = E_i_base * (1.0 + delta_fraction)
_, stresses_Ei_high = run_homogenized_simulation(E_m_base, E_i_perturbed, sig0_base)
delta_Ei = E_i_perturbed - E_i_base
dSxx_dEi = (stresses_Ei_high[-1] - stresses_base[-1]) / delta_Ei

# 4. Evaluate Matrix Yield Limit Variation (+15%)
print("Running Matrix Yield Strength Variation...")
sig0_perturbed = sig0_base * (1.0 + delta_fraction)
_, stresses_sig0_high = run_homogenized_simulation(E_m_base, E_i_base, sig0_perturbed)
delta_sig0 = sig0_perturbed - sig0_base
dSxx_dsig0 = (stresses_sig0_high[-1] - stresses_base[-1]) / delta_sig0

# ==============================================================================
#                RENDER OVERLAYED HOMOGENIZED MATERIAL CURVES
# ==============================================================================
plt.figure(figsize=(9, 6))

# Plot the multi-phase paths
plt.plot(strains_base, stresses_base, color='black', linewidth=1.0, label="Baseline Reference Case")
plt.plot(strains_base, stresses_Em_high, '-', color='royalblue', linewidth=1.0, label=f"Higher Matrix Stiffness (+15% $E_m$)")
plt.plot(strains_base, stresses_Ei_high, '-', color='seagreen', linewidth=1.0, label=f"Higher Inclusion Stiffness (+15% $E_i$)")
plt.plot(strains_base, stresses_sig0_high, '-', color='darkorange', linewidth=1.0, label=f"Higher Matrix Yield Strength (+15% $\sigma_0$)")
plt.legend()

# ==============================================================================
#                 POST-PROCESS AND VISUALIZE SENSITIVITIES
# ==============================================================================
parameters = [r'$E_{matrix}$', r'$E_{inclusion}$', r'$\sigma_{0, matrix}$']
sensitivities = [dSxx_dEm, dSxx_dEi, dSxx_dsig0]

# Compute Normalized Sensitivities (Elasticity index) to see relative importance percentage
# (dSigma / dParam) * (Param / Sigma)
normalized_sensitivities = [
    dSxx_dEm * (E_m_base / stresses_base[-1]),
    dSxx_dEi * (E_i_base / stresses_base[-1]),
    dSxx_dsig0 * (sig0_base / stresses_base[-1])
]

print("\n==================================================")
print("          SENSITIVITY ANALYSIS RESULTS            ")
print("==================================================")
for param, raw, norm in zip(parameters, sensitivities, normalized_sensitivities):
    print(f"Parameter: {param:15s} | Raw Gradient: {raw:12.4f} | Normalized Sensitivity: {norm:12.4f}")

# Render a bar chart comparing the parameter impacts
plt.figure(figsize=(8, 4))
colors = ['royalblue', 'seagreen', 'darkorange']
plt.bar(parameters, normalized_sensitivities, color=colors, edgecolor='black', alpha=0.8, width=0.5)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.ylabel("Normalized Sensitivity Index\n" r"$(\partial \Sigma_{xx} / \partial \theta) \cdot (\theta / \Sigma_{xx})$", fontsize=11)
plt.title("Sensitivity of Final Macro Stress $\Sigma_{xx}$ to Material Properties", fontsize=12, fontweight='bold')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("macroscopic_sensitivity_analysis.png", dpi=300)
plt.show()