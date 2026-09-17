import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
import matplotlib.pyplot as plt
from macromodel import Macromodel 
from rve_j2_linear_clean import Micromodel     
from customnewtonproblem import CustomNewtonProblem
import sys
import pyvista as pv
import os
from quadrature import macroscaleQuadratureMap, plottingDomain
from auxFunctions import _setup_solver, _create_loading_function, print_cell_coordinates

# Avoids paraview from opening
if not os.getenv("DISPLAY"):
    pv.OFF_SCREEN = True
    
def strain_vec(u):
    epsilon = ufl.sym(ufl.grad(u))
    factor = 2.0
    return ufl.as_vector([epsilon[0,0], epsilon[1,1], factor * epsilon[0,1]])

# User-defined choices for solver and debugging
directSolverMacro = True
directSolverMicro = False
verboseSolver = True
verboseQuad = False
verboseMacro = False
verboseMicro = False
maxMacroSubsteps = 2      

# ==============================================================================
# Macroscopic domain
# ==============================================================================

L, H = 2., 1.
nx, ny = 6, 3

# Create macroscopic mesh
macromodel = Macromodel(L, H, nx, ny)
#macromodel._create_mesh_with_gmsh()
macromodel._create_mesh()
fdim = macromodel.fdim
gdim = macromodel.gdim

# Create macroscopic fields and variables
macromodel._setup_functions( )

# Get number of elements in the macroscopic domain
num_cells = macromodel.domain.topology.index_map(gdim).size_local

# Setup boundary conditions on macroscale
macromodel._setup_bc()

if macromodel.domain.comm.Get_rank() == 0:
    print(f"Allocating master RVE template and initializing solver.")
    
# TODO: create function for setting material properties of micromodel phases
# TODO: add validation curves from jive
# TODO: plot macro and micromodels

# Initializing master RVE  
master_rve = Micromodel(directSolver = directSolverMicro,
                        verbose = verboseMicro) 

# Instantiate the quadrature map with the new historical functions
macro_qmap = macroscaleQuadratureMap(
    macro_domain=macromodel.domain,       
    master_rve=master_rve,
    u_macro=macromodel.u,
    W_macro_tensor=macromodel.W_tensor,
    macro_stress_field=macromodel.stress_field,
    W_macro_tangent=macromodel.W_tangent,
    macro_tangent_field=macromodel.tangent_field,
    verbose = verboseQuad
)

# Define weak forms
dx_m = ufl.Measure("dx", domain=macromodel.domain, metadata={"quadrature_degree": 1})
sig_macro_tensor = ufl.as_tensor([[macromodel.stress_field[0], macromodel.stress_field[2]], 
                              [macromodel.stress_field[2], macromodel.stress_field[1]]])
F_macro_form = ufl.inner(sig_macro_tensor, ufl.sym(ufl.grad(macromodel.u_test))) * dx_m
C_tangent = ufl.as_matrix([
    [macromodel.tangent_field[0], macromodel.tangent_field[1], macromodel.tangent_field[2]],
    [macromodel.tangent_field[3], macromodel.tangent_field[4], macromodel.tangent_field[5]],
    [macromodel.tangent_field[6], macromodel.tangent_field[7], macromodel.tangent_field[8]]
])
J_macro_form = ufl.inner(C_tangent * strain_vec(macromodel.u_trial), strain_vec(macromodel.u_test)) * dx_m
        
# Define solver type
ksp_solver = _setup_solver(macromodel.domain, directSolverMacro, 
                           tag = 'macro_', 
                           verbose = verboseSolver)

# ==============================================================================
# Main 
# ==============================================================================

# Track strain-stress of element track_cell_id for visual inspection
track_cell_id = 0  
plot_tracked_cell = True
target_cell_history = {"strain_xx": [], "stress_xx": []}

# To collect average displacement and load at the right edge of the macro domain
displacement_history = []
load_history = []

# Create load function to prescribe displacement at the right edge 
step_size = 0.001
n_steps = 60
macro_loading = _create_loading_function(n_steps=n_steps, start = 0, end = n_steps*step_size)
#macro_loading = _create_loading_function(load_type = 'cyclic', n_steps = 15, unl_norm=0.03, rel_norm=0.05)
#macro_loading = _create_loading_function(load_type = 'gp', n_steps=50, seed = 1)

if macromodel.domain.comm.Get_rank() == 0:
    print("\n--- Starting FE2 simulation ---")

# Define macroscopic problem
macro_problem = CustomNewtonProblem(
    quadrature_map=macro_qmap,
    directSolver = directSolverMacro,
    verbose = verboseMacro,
    F=F_macro_form,
    J=J_macro_form,
    u=macromodel.u,
    bcs=macromodel.bcs,
    max_it=20,
    atol=1e-4,
    rtol=1e-3
)

# For plotting only
xmin, ymin = 0.0, 0.0
xmax, ymax = L + 0.2, H + 0.05
margin = 0.05 * max(xmax - xmin, ymax - ymin)
fixed_bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin, -1, 1)
plot = plottingDomain()

# Loading loop
prev_disp = 0.0
for step_index, target_disp in enumerate(macro_loading):
    delta_disp = target_disp - prev_disp
    substeps = 1
    step_converged = False
        
    if macromodel.domain.comm.Get_rank() == 0:
        print(f"\n--- STEP {step_index} ---"); sys.stdout.flush()
        print(f"\n    Macro time step {step_index:02d} | Prescribed displacement: {target_disp:.4f} mm")    
        
    # Save state at the beginning of the stpe for backup
    u_step_start = macromodel.u.x.array.copy()
        
    while substeps <= maxMacroSubsteps and not step_converged:
        sub_delta = delta_disp / substeps
        
        if macromodel.domain.comm.Get_rank() == 0 and substeps > 1:
            print(f"  [SUB-STEPPING] Attempting with {substeps} sub-steps (sub-increment: {sub_delta:.6f} mm)")
            
        # Reset displacement & state to beginning of step
        macromodel.u.x.array[:] = u_step_start
        macromodel.u.x.scatter_forward()
            
        substep_failed = False
        for substep_idx in range(1, substeps + 1):
            sub_target = prev_disp + substep_idx * sub_delta 
            print(f"    Prescribed displacement: {sub_target:.4f} mm")    

            # Apply displacement
            macromodel.applied_pull.value = sub_target
           
            # Solve macroscopic problem 
            sub_converged, total_newton_its = macro_problem.solve()
                    
            # Finish all cpus before heading to the next step
            macromodel.domain.comm.Barrier()
            
            if not sub_converged:
                if macromodel.domain.comm.Get_rank() == 0:
                    print(f"    Sub-step {substep_idx}/{substeps} failed to converge.", flush=True)
                substep_failed = True
                break
            else:
                # Advance material internal variables for this sub-step
                macro_qmap.advance(macromodel.u)

        if not substep_failed:
            step_converged = True
        else:
            # Sub-step failed: rollback material state and increase substeps
            macro_qmap.rollback()
            substeps += 1
        
        if step_converged:
            prev_disp = target_disp
            if macromodel.domain.comm.Get_rank() == 0 and substeps > 1:
                print(f"  [CONVERGED] Macro step {step_index} successfully converged using {substeps} sub-steps!")
                print("  [DEBUG] Solved displacement.", flush = True)
        else:
            print("  [DEBUG] Reverting to old macroscopic state.", flush = True)            
            macro_qmap.rollback()
            substeps += 1
        
        # Signal if convergence was reached or not at the macroscale
        if macromodel.domain.comm.Get_rank() == 0:
            if not step_converged:            
                print(f"  [CRITICAL] Macro time step {step_index:02d} failed to converged in {total_newton_its}. Aborting simulation loop.")          
            else:
                print(f"  [CONVERGED] Macro time step {step_index:02d} converged in {total_newton_its} iterations.")    
                #print(macromodel.u.x.array)
                
                # Collecting strain-stress quantities from tracked element
                if track_cell_id < macro_qmap.local_cells_count:
                    stress_start = track_cell_id * macro_qmap.stress_bs
                    evaluated_stress = macromodel.stress_field.x.array[stress_start:stress_start + macro_qmap.stress_bs]
                    
                    # Pull latest local strain tracking
                    macro_qmap.local_strain_field.interpolate(macro_qmap.strain_expr)
                    evaluated_strain = macro_qmap.local_strain_field.x.array[stress_start:stress_start + macro_qmap.stress_bs]
                    
                    target_cell_history["strain_xx"].append(evaluated_strain[0])
                    target_cell_history["stress_xx"].append(evaluated_stress[0])
            
                # Boundaries reaction force reduction calculation at the right edge
                # get nodes at x = L
                facets_f = mesh.locate_entities(macromodel.domain, fdim, lambda x: np.isclose(x[0], L))
                sorted_f = np.argsort(facets_f)
                # Assign specific tag to this set (= 2 in this case)
                f_tag = mesh.meshtags(macromodel.domain, fdim, facets_f[sorted_f], np.full_like(facets_f, 2, dtype=np.int32)[sorted_f])
                # Create boundary integration over subdomain specified
                ds_m = ufl.Measure("ds", domain=macromodel.domain, subdomain_data=f_tag)
                # Defines outward normal vectors along entire macroscopic domain
                n_m = ufl.FacetNormal(macromodel.domain)
                
                # Compute force only over the subdomain (only boundaries marked with the tag = 2)
                total_force = fem.assemble_scalar(fem.form(ufl.dot(sig_macro_tensor, n_m)[0] * ds_m(2)))
                total_force = macromodel.domain.comm.allreduce(total_force, op=MPI.SUM)
                    
                # Store load-displacement data
                if macromodel.domain.comm.Get_rank() == 0:
                    displacement_history.append(target_disp)
                    load_history.append(total_force)
                    print(f"  Integrated macro load force: {total_force:10.3f} N", flush = True)
            
                # Check what each rank sees
                num_local_cells = macromodel.domain.topology.index_map(macromodel.domain.topology.dim).size_local
                print(f"Rank {macromodel.domain.comm.Get_rank()} has {num_local_cells} elements.", flush=True)
                
                # Force sync
                macromodel.domain.comm.Barrier()
                
                # Call plotting function
                plot.update(macromodel.domain, gdim, macromodel.u, macro_qmap, macromodel.stress_field, 
                            step_index, fixed_bounds)
                macromodel.domain.comm.Barrier()
            
    if not step_converged:   
        if macromodel.domain.comm.Get_rank() == 0:
            print(f"\n[FATAL] Macro step {step_index} failed with {maxMacroSubsteps} sub-steps. Stopping simulation.")
            break

# Plot final load-displacement curve at the right edge of the macroscopic domain
if macromodel.domain.comm.Get_rank() == 0:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(displacement_history, load_history, 'b-o', markersize=4, linewidth=1.5)
    ax.set_xlabel("Average displacement at x = L [mm]")
    ax.set_ylabel("Total reaction force at x = L [N]")
    ax.grid(True)
    plt.tight_layout()
    plt.savefig("load_displacement_curve.png", dpi=150)
    plt.close()
    print("Load-displacement curve saved.")
    
    if plot_tracked_cell:
        plt.figure(figsize=(5.5, 3.8))
        plt.plot(target_cell_history["strain_xx"], target_cell_history["stress_xx"], 'o--', color='darkblue', linewidth=1.8, label=f"Cell id {track_cell_id}")
        plt.xlabel(r"Macroscopic strain $\varepsilon_{xx}$ [-]", fontsize=10)
        plt.ylabel(r"Macroscopic stress $\sigma_{xx}$ [MPa]", fontsize=10)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"fe2_local_response_cell_{track_cell_id}.png", dpi=300)
        plt.close()
        print("Strain-stress curve of element saved.")

#print_cell_coordinates(macro_domain, track_cell_id)        