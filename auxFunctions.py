#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  9 14:44:20 2026

@author: malvesmaia
"""

def _create_loading_function(load_type='monotonic', n_steps=10, **kwargs):
    """
    Generate a loading profile of the requested type.

    Parameters
    ----------
    load_type : {'monotonic', 'cyclic', 'gp'}
        - 'monotonic': linear function from `start` to `end`.
        - 'cyclic':    load / unload / reload / hold profile.
        - 'gp':        smooth random (non-proportional and non-monotonic) profile.
    n_steps : int
        Number of steps. For 'cyclic', this is steps *per phase*.
    **kwargs
        Extra parameters specific to the chosen load_type
        (see _monotonic_loading, _cyclic_loading, _gp_loading).

    Returns
    -------
    np.ndarray
    """
    key_generators = {
        'monotonic': _monotonic_loading,
        'cyclic': _cyclic_loading,
        'gp': _gp_loading,
    }

    try:
        generator = key_generators[load_type.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown load_type '{load_type}'. Valid options: {list(key_generators)}"
        )

    return generator(n_steps, **kwargs)

def _monotonic_loading(n_steps=10, start=0.00, end=0.04):
    import numpy as np
    """Linear function from `start` to `end` over `n_steps`."""
    return np.linspace(start, end, n_steps)

def _cyclic_loading(steps_per_phase=10, unl_norm=0.02, end_unl_norm=0.01,
                     rel_norm=0.03, hold_steps=None):
    """
    Piecewise-linear load / unload / reload / hold profile:
        1. Load:    0.00        -> unl_norm      (steps_per_phase steps)
        2. Unload:  unl_norm    -> end_unl_norm  (steps_per_phase steps)
        3. Reload:  end_unl_norm -> rel_norm     (steps_per_phase steps)
        4. Hold:    constant at rel_norm         (hold_steps steps)

    If `hold_steps` is not given, it defaults to half of `steps_per_phase`.
    """
    import numpy as np
    if hold_steps is None:
        hold_steps = int (steps_per_phase/2)

    load_phase = np.linspace(0.00, unl_norm, steps_per_phase + 1)
    unload_phase = np.linspace(unl_norm, end_unl_norm, steps_per_phase + 1)[1:]
    reload_phase = np.linspace(end_unl_norm, rel_norm, steps_per_phase + 1)[1:]
    hold_phase = np.full(hold_steps, rel_norm)

    return np.concatenate([load_phase, unload_phase, reload_phase, hold_phase])

def _gp_loading(n_steps=10, lengthscale=20, variance=8e-4, seed=13):
    """
    Smooth, non-proportional and non-monotonic profile sampled from a 
    zero-mean Gaussian Process (RBF kernel), conditioned to start at (x=0, y=0).
    """
    
    import numpy as np
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    rng = np.random.default_rng(seed)

    x = np.linspace(0, n_steps - 1, n_steps).reshape(-1, 1)    # Time step index
    kernel = C(variance, "fixed") * RBF(lengthscale, "fixed")  # Setting hyperparameters of GP
    
    # Initialize the Gaussian Process
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, alpha=1e-14)

    # Condition the GP on the prior data point to always start from x=0, y=0
    x_train = np.array([[0.0]])
    y_train = np.array([0.0])
    gp.fit(x_train, y_train)
    
    seed = rng.integers(0, 10000)
    y = gp.sample_y(x, n_samples=1, random_state=seed)
    
    return y[:,0]

def _setup_solver(domain, directSolver = True, tag = 'macro_', verbose = False):
    """
    
    
    Parameters
    ----------
    domain : TYPE
        DESCRIPTION.
    directSolver : TYPE, optional
        DESCRIPTION. The default is True.
    tag : TYPE, optional
        DESCRIPTION. The default is 'macro_'.
    verbose : TYPE, optional
        DESCRIPTION. The default is False.

    Returns
    -------
    ksp_solver : TYPE
        DESCRIPTION.

    """
    from petsc4py import PETSc
    ksp_solver = PETSc.KSP().create(domain.comm)
    if directSolver == True:
        ksp_solver.setType(PETSc.KSP.Type.PREONLY)
        pc = ksp_solver.getPC()
        pc.setType(PETSc.PC.Type.LU)
        pc.setFactorSolverType("mumps")    
        ksp_solver.setOptionsPrefix(tag)
        ksp_solver.setTolerances(rtol=1e-5, atol=1e-6, max_it=20)
        ksp_solver.setFromOptions()
        ksp_solver.setErrorIfNotConverged(True)
        if verbose: ksp_solver.setMonitor(lambda ksp, it, rnorm: print(tag[:-1] + f" direct solver. Iteration {it}: residual norm {rnorm:.3e}"))
    else:
        ksp_solver.setType(PETSc.KSP.Type.GMRES)
        pc = ksp_solver.getPC()
        pc.setType(PETSc.PC.Type.GAMG)
        ksp_solver.setTolerances(rtol=1e-5, atol=1e-6, max_it=10)
        ksp_solver.setFromOptions()
        ksp_solver.setErrorIfNotConverged(True)
        if verbose: ksp_solver.setMonitor(lambda ksp, it, rnorm: print(tag[:-1] + f" iterative solver. Iteration {it}: residual norm {rnorm:.3e}"))
        ksp_solver.setOptionsPrefix(tag)
    return ksp_solver

def _strain_vec(u, strain_factor = 2.0):
    import ufl
    epsilon = ufl.sym(ufl.grad(u))
    return ufl.as_vector([epsilon[0,0], epsilon[1,1], strain_factor * epsilon[0,1]])

def _compute_plotting_bounds(L, H):
    xmin, ymin = 0.0, 0.0
    xmax, ymax = L + 0.2, H + 0.05
    margin = 0.05 * max(xmax - xmin, ymax - ymin)
    fixed_bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin, -1, 1)
    return fixed_bounds

def _print_cell_coordinates(domain, cell_id):
    """
    

    Parameters
    ----------
    domain : TYPE
        DESCRIPTION.
    cell_id : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    
    import numpy as np
    # Get the connectivity mapping cells to vertices
    gdim = domain.topology.dim
    conn = domain.topology.connectivity(gdim, 0)
    
    # Extract the vertex indices belonging to the target cell
    vertex_indices = conn.links(cell_id)
    
    # Fetch the coordinates of these vertices
    vertex_coords = domain.geometry.x[vertex_indices]
    
    # Compute the centroid 
    centroid_x = np.mean(vertex_coords[:, 0])
    centroid_y = np.mean(vertex_coords[:, 1])
    
    print("\n--- Tracked Macro Integration Point Info ---")
    print(f"Target Cell ID: {cell_id}")
    print(f"Vertex Coordinates:\n{vertex_coords[:, :2]}")
    print(f"Cell Centroid Location: X = {centroid_x:.3f} mm, Y = {centroid_y:.3f} mm")    