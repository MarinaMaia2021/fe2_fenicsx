#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 14:02:19 2026

@author: malvesmaia
"""

import numpy as np
from mpi4py import MPI
import ufl
from dolfinx import fem, mesh
import matplotlib.pyplot as plt
from macromodel import Macromodel
from customnewtonproblem import CustomNewtonProblem
import sys
import pyvista as pv
import os
from quadrature import macroscaleQuadratureMap, plottingDomain
from auxFunctions import _create_loading_function, _print_cell_coordinates, _strain_vec, _compute_plotting_bounds, _read_external_file

# Avoids paraview from opening
if not os.getenv("DISPLAY"):
    pv.OFF_SCREEN = True

# User-defined choices for solver and debugging
key_micromodel = 'j2'  # 'j2' 'composite_coarse' 
direct_solver_macro = True
direct_solver_micro = True
verbose_quad = False
verbose_macro = False
verbose_micro = False
max_substeps_macro = 2
strain_factor = 2.0

def _load_micromodel_class(key_micromodel):
    """Imports and returns the Micromodel class matching `key_micromodel`."""
    key = key_micromodel.lower()
    if key == 'j2':
        from rve_j2_linear_clean import Micromodel
    elif key == 'composite_coarse':
        from rve_fiber_matrix import Micromodel
    elif key == 'composite_medium':
        from rve_ext_clean_refac import Micromodel
    else:
        raise ValueError(f"Unknown micromodel type: '{key_micromodel}'")
    return Micromodel


def _build_macro_weak_forms(macromodel, strain_factor):
    """Assembles the macroscopic residual (F) and tangent (J) weak forms."""
    dx_m = ufl.Measure("dx", domain=macromodel.domain, metadata={"quadrature_degree": 1})

    sig_macro_tensor = ufl.as_tensor([
        [macromodel.stress_field[0], macromodel.stress_field[2]],
        [macromodel.stress_field[2], macromodel.stress_field[1]],
    ])
    F_macro_form = ufl.inner(sig_macro_tensor, ufl.sym(ufl.grad(macromodel.u_test))) * dx_m

    C_tangent = ufl.as_matrix([
        [macromodel.tangent_field[0], macromodel.tangent_field[1], macromodel.tangent_field[2]],
        [macromodel.tangent_field[3], macromodel.tangent_field[4], macromodel.tangent_field[5]],
        [macromodel.tangent_field[6], macromodel.tangent_field[7], macromodel.tangent_field[8]],
    ])
    J_macro_form = ufl.inner(
        C_tangent * _strain_vec(macromodel.u_trial, strain_factor),
        _strain_vec(macromodel.u_test, strain_factor),
    ) * dx_m

    return sig_macro_tensor, F_macro_form, J_macro_form


def _setup_right_edge_measure(macromodel, L, fdim, tag=2):
    """
    Tags the facets at x = L once and returns the boundary measure/normal needed
    to integrate the reaction force there. The macro mesh doesn't change during
    the simulation (so this only needs to be built once).
    """
    facets = mesh.locate_entities(macromodel.domain, fdim, lambda x: np.isclose(x[0], L))
    order = np.argsort(facets)
    facet_tags = mesh.meshtags(macromodel.domain, fdim, facets[order], np.full_like(facets, tag, dtype=np.int32)[order])

    ds_right = ufl.Measure("ds", domain=macromodel.domain, subdomain_data=facet_tags)
    n_macro = ufl.FacetNormal(macromodel.domain)
    return ds_right(tag), n_macro


def _compute_reaction_force(macromodel, sig_macro_tensor, ds_right, n_macro):
    """Integrates the horizontal traction over the right edge and sums it across ranks."""
    local_force = fem.assemble_scalar(fem.form(ufl.dot(sig_macro_tensor, n_macro)[0] * ds_right))
    return macromodel.domain.comm.allreduce(local_force, op=MPI.SUM)


def _try_solve_macro_step(macro_problem, macro_qmap, macromodel, u_step_start,
                           prev_disp, target_disp, substeps, is_root):
    """
    Attempts to advance from `prev_disp` to `target_disp` using `substeps` equal
    sub-increments, restarting from `u_step_start` each time this is called.

    Returns (converged, total_newton_its). On failure, the micro/macro state is
    rolled back to the start of the step.
    """
    macromodel.u.x.array[:] = u_step_start
    macromodel.u.x.scatter_forward()

    delta_disp = target_disp - prev_disp
    sub_delta = delta_disp / substeps
    total_newton_its = 0

    for substep_idx in range(1, substeps + 1):
        sub_target = prev_disp + substep_idx * sub_delta
        if is_root:
            print(f"    Prescribed displacement: {sub_target:.4f} mm")

        macromodel.applied_pull.value = sub_target
        sub_converged, total_newton_its = macro_problem.solve()
        macromodel.domain.comm.Barrier()

        if not sub_converged:
            if is_root:
                print(f"    Sub-step {substep_idx}/{substeps} failed to converge.", flush=True)
            macro_qmap.rollback()
            return False, total_newton_its

        # Advance material internal variables for this sub-step
        macro_qmap.advance(macromodel.u)

    return True, total_newton_its


def _record_converged_step(step_index, target_disp, macromodel, macro_qmap, gdim, fdim, L,
                            sig_macro_tensor, ds_right, n_macro, track_cell_id,
                            target_cell_history, displacement_history, load_history,
                            plot, fixed_bounds, is_root):
    """Collects tracked-cell strain/stress, the reaction force, and updates the live plot."""
    if track_cell_id < macro_qmap.local_cells_count:
        stress_start = track_cell_id * macro_qmap.stress_bs
        stress_slice = slice(stress_start, stress_start + macro_qmap.stress_bs)
        evaluated_stress = macromodel.stress_field.x.array[stress_slice]

        macro_qmap.local_strain_field.interpolate(macro_qmap.strain_expr)
        evaluated_strain = macro_qmap.local_strain_field.x.array[stress_slice]

        target_cell_history["strain_xx"].append(evaluated_strain[0])
        target_cell_history["stress_xx"].append(evaluated_stress[0])

    total_force = _compute_reaction_force(macromodel, sig_macro_tensor, ds_right, n_macro)

    if is_root:
        displacement_history.append(target_disp)
        load_history.append(total_force)
        print(f"  Integrated macro load force: {total_force:10.3f} N", flush=True)

    num_local_cells = macromodel.domain.topology.index_map(macromodel.domain.topology.dim).size_local
    print(f"Rank {macromodel.domain.comm.Get_rank()} has {num_local_cells} elements.", flush=True)
    macromodel.domain.comm.Barrier()
    
    plot.update(macromodel.domain, gdim, macromodel.u, macro_qmap, macromodel.stress_field, step_index, fixed_bounds)
    macromodel.domain.comm.Barrier()


def _plot_load_displacement_curve(displacement_history, 
                                  load_history,
                                  displacement_reference = None,
                                  load_reference = None,
                                  path="load_displacement_curve.png"):
    # TODO: make it prettier
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(displacement_history, load_history, 'b-o', label = 'fenicsx dolfinx', markersize=4, linewidth=1.5)
    if displacement_reference is not None:
        ax.plot(displacement_reference, load_reference, 'x', color = 'red',label = 'jive', linestyle = 'dashed', markersize=4, linewidth=1.5)
    ax.set_xlabel("Average displacement at x = L [mm]")
    ax.set_ylabel("Total reaction force at x = L [N]")
    plt.legend(loc = 'upper left')
    ax.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print("Load-displacement curve saved.")


def _plot_tracked_cell_response(target_cell_history, track_cell_id):
    # TODO: make it prettier
    plt.figure(figsize=(5.5, 3.8))
    plt.plot(target_cell_history["strain_xx"], target_cell_history["stress_xx"],
              'o--', color='darkblue', linewidth=1.8, label=f"Cell id {track_cell_id}")
    plt.xlabel(r"Macroscopic strain $\varepsilon_{xx}$ [-]", fontsize=10)
    plt.ylabel(r"Macroscopic stress $\sigma_{xx}$ [MPa]", fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"fe2_local_response_cell_{track_cell_id}.png", dpi=300)
    plt.close()
    print("Strain-stress curve of element saved.")

def main():
    Micromodel = _load_micromodel_class(key_micromodel)

    # ==========================================================================
    # Macroscopic domain
    # ==========================================================================

    L, H = 1.0, 1.0
    nx, ny = 2, 2

    macromodel = Macromodel(L, H, nx, ny)
    macromodel._create_mesh()
    gdim, fdim = macromodel.gdim, macromodel.fdim

    macromodel._setup_functions()
    macromodel._setup_bc()

    is_root = macromodel.domain.comm.Get_rank() == 0
    if is_root:
        print("Allocating master RVE template and initializing solver.")

    # TODO: create function for setting material properties of micromodel phases
    # TODO: debug why simulation crashes with macroscopic mesh refinement
    # TODO: plot macro and micromodels

    master_rve = Micromodel(direct_solver=direct_solver_micro,
                             verbose=verbose_micro,
                             strain_factor=strain_factor)

    macro_qmap = macroscaleQuadratureMap(
        macro_domain=macromodel.domain,
        master_rve=master_rve,
        u_macro=macromodel.u,
        W_macro_tensor=macromodel.W_tensor,
        macro_stress_field=macromodel.stress_field,
        W_macro_tangent=macromodel.W_tangent,
        macro_tangent_field=macromodel.tangent_field,
        verbose=verbose_quad
    )

    sig_macro_tensor, F_macro_form, J_macro_form = _build_macro_weak_forms(macromodel, strain_factor)
    ds_right, n_macro = _setup_right_edge_measure(macromodel, L, fdim)

    # ==========================================================================
    # Main
    # ==========================================================================

    # Track strain-stress of element track_cell_id for visual inspection
    track_cell_id = 2
    plot_tracked_cell = True
    target_cell_history = {"strain_xx": [], "stress_xx": []}

    # Average displacement and load at the right edge of the macro domain
    displacement_history = []
    load_history = []

    # Create the load function prescribing displacement at the right edge        
    step_size = 0.001
    n_steps = 60        
    if key_micromodel == 'composite_medium':
        # TOD: debug why is there a different stiffness depending on step_size
        step_size = 0.00025
        n_steps = 240

    macro_loading = _create_loading_function(n_steps=n_steps, start=0, end=(n_steps - 1) * step_size)
   # macro_loading = _create_loading_function(load_type='cyclic', n_steps=15, unl_norm=0.03, rel_norm=0.05)
    # macro_loading = _create_loading_function(load_type='gp', n_steps=50, seed=1)


    if is_root:
        print("\n--- Starting FE2 simulation ---")

    macro_problem = CustomNewtonProblem(
        quadrature_map=macro_qmap,
        directSolver=direct_solver_macro,
        verbose=verbose_macro,
        F=F_macro_form,
        J=J_macro_form,
        u=macromodel.u,
        bcs=macromodel.bcs,
    )

    fixed_bounds = _compute_plotting_bounds(L, H)
    plot = plottingDomain()

    # Loading loop. For each target displacement, try to solve directly.
    # If it fails, roll back and retry with more (and smaller) substeps.
    prev_disp = 0.0
    for step_index, target_disp in enumerate(macro_loading):
        if is_root:
            print(f"\n--- STEP {step_index} ---")
            print(f"\n    Macro time step {step_index:02d} | Prescribed displacement: {target_disp:.4f} mm")
            sys.stdout.flush()

        u_step_start = macromodel.u.x.array.copy()
        substeps = 1
        step_converged = False
        total_newton_its = 0

        while substeps <= max_substeps_macro and not step_converged:
            if is_root and substeps > 1:
                print(f"  [SUB-STEPPING] Attempting with {substeps} sub-steps "
                      f"(sub-increment: {(target_disp - prev_disp) / substeps:.6f} mm)")

            step_converged, total_newton_its = _try_solve_macro_step(
                macro_problem, macro_qmap, macromodel, u_step_start,
                prev_disp, target_disp, substeps, is_root,
            )

            if not step_converged:
                substeps += 1

        if is_root:
            if step_converged:
                if substeps > 1:
                    print(f"  [CONVERGED] Macro step {step_index} successfully converged using {substeps} sub-steps!")
                print(f"  [CONVERGED] Macro time step {step_index:02d} converged in {total_newton_its} iterations.")
            else:
                print(f"  [CRITICAL] Macro time step {step_index:02d} failed to converge in {total_newton_its}. "
                      f"Aborting simulation loop.")

        if not step_converged:
            if is_root:
                print(f"\n[FATAL] Macro step {step_index} failed with {max_substeps_macro} sub-steps. Stopping simulation.")
            break

        prev_disp = target_disp
        _record_converged_step(
            step_index, target_disp, macromodel, macro_qmap, gdim, fdim, L,
            sig_macro_tensor, ds_right, n_macro, track_cell_id,
            target_cell_history, displacement_history, load_history,
            plot, fixed_bounds, is_root,
        )

    # Plot final load-displacement curve at the right edge of the macroscopic domain
    if is_root:
        disp_jive, load_jive = None, None
        load_disp_filename = 'load_displacement_curve_' + key_micromodel + '.png'
        
        load_disp_jive_file = 'macro_jive_' + key_micromodel + '.dat'
        try:
            disp_jive, load_jive = _read_external_file(load_disp_jive_file, columns = [1, 2])
        except (FileNotFoundError, RuntimeError, TypeError, NameError):
            print('No JIVE data for plotting/comparison.')
        
        _plot_load_displacement_curve(displacement_history, load_history, 
                                      disp_jive, load_jive,
                                      path = load_disp_filename)
        
        if plot_tracked_cell:
            _plot_tracked_cell_response(target_cell_history, track_cell_id)
            
        _print_cell_coordinates(macromodel.domain, track_cell_id )


if __name__ == "__main__":
    main()
