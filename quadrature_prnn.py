#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 19:57:02 2026

@author: malvesmaia
"""

import jax
import jax.numpy as jnp
import numpy as np
from petsc4py import PETSc

class macroscaleQuadratureMap:
    def __init__(self, macro_domain, u_macro, W_macro_tensor, 
                 macro_stress_field, W_macro_tangent, macro_tangent_field,
                 prnn_model, prnn_params, prnn_material):
        
        from dolfinx import fem
        import ufl
        
        self.macro_domain = macro_domain
        self.comm = macro_domain.comm
        self.u_macro = u_macro
        self.macro_stress_field = macro_stress_field
        self.macro_tangent_field = macro_tangent_field
        
        # JAX components
        self.model = prnn_model
        self.params = prnn_params
        self.material = prnn_material
        
        # Pre-compile JIT functions for speed
        self._stress_and_tangent_jit = jax.jit(self._compute_stress_and_tangent)

        self.gdim = self.macro_domain.topology.dim
        self.local_cells_count = self.macro_domain.topology.index_map(self.gdim).size_local

        # History initialization
        self.n_matpts = prnn_model.n_matpts  # Get directly from model
        
        # 1 material point has 6 plastic strain components and 1 scalar eq. strain
        self.ep_dim = self.n_matpts * 6 
        self.alpha_dim = self.n_matpts * 1  
        
        # Re-initialize spaces with the correct block sizes per cell
        history_space = fem.functionspace(self.macro_domain, ("DG", 0, (self.alpha_dim,)))
        ep_space      = fem.functionspace(self.macro_domain, ("DG", 0, (self.ep_dim,)))
        
        self.alpha_old_func = fem.Function(history_space)
        self.alpha_tmp_func = fem.Function(history_space)
        self.ep_old_func    = fem.Function(ep_space)
        self.ep_tmp_func    = fem.Function(ep_space)

        # Strain Projection
        eps = ufl.sym(ufl.grad(self.u_macro))
        strain_vec = ufl.as_vector([eps[0, 0], eps[1, 1], eps[0, 1]])
        self.strain_expr = fem.Expression(strain_vec, W_macro_tensor.element.interpolation_points)
        self.local_strain_field = fem.Function(W_macro_tensor)

        self.stress_bs = W_macro_tensor.dofmap.index_map_bs
        self.tangent_bs = W_macro_tangent.dofmap.index_map_bs

    def _compute_stress_and_tangent(self, params, eps_in, material, h_old):
        # Forward pass returning stress and updated history state dict/object
        stress, h_new = self.model.apply(params, eps_in, material, h_old)
        
        # Jacobian (Tangent operator) evaluated strictly with respect to the input macro strain
        tangent = jax.jacfwd(lambda e: self.model.apply(params, e, material, h_old)[0])(eps_in)
        return stress, tangent, h_new

    def update(self):
        self.local_strain_field.interpolate(self.strain_expr)
        self.local_strain_field.x.scatter_forward()
        from jax_j2 import HistState
        
        b_size = 1
        m_pts = self.model.n_matpts
        expected_batch = b_size * m_pts
        
        print('Neural Network Quadrature Update Intersect Started', flush=True)
        
        for local_idx in range(self.local_cells_count):
            s_start, t_start = local_idx * self.stress_bs, local_idx * self.tangent_bs
            h_start = local_idx * self.alpha_old_func.function_space.dofmap.index_map_bs
            ep_start = local_idx * self.ep_old_func.function_space.dofmap.index_map_bs
            
            # Input Strain Extraction
            E_macro = self.local_strain_field.x.array[s_start : s_start + self.stress_bs]
            eps_in = jnp.array(E_macro).reshape(b_size, 1, -1)
            
            # --- FIX: ALWAYS read from the converged historical state of the PREVIOUS time step ---
            raw_alpha = self.alpha_old_func.x.array[h_start : h_start + self.alpha_dim]
            raw_ep = self.ep_old_func.x.array[ep_start : ep_start + self.ep_dim]
            
            # Packing state container for JAX Model
            h_old = HistState(
                eps_plastic=jnp.array(raw_ep).reshape(expected_batch, 6),
                eps_p_eq=jnp.array(raw_alpha).reshape(expected_batch)
            )
            
            # Execute compiled JIT network evaluation
            stress, tangent, h_new = self._stress_and_tangent_jit(self.params, eps_in, self.material, h_old)
            
            # Write physical outputs directly into FEniCSx macroscopic fields
            self.macro_stress_field.x.array[s_start : s_start + self.stress_bs] = np.array(stress).flatten()
            self.macro_tangent_field.x.array[t_start : t_start + self.tangent_bs] = np.array(tangent).flatten()
            
            # Save the new updates to temporary variables (staged until advance() is called)
            self.alpha_tmp_func.x.array[h_start : h_start + self.alpha_dim] = np.array(h_new.eps_p_eq).flatten()
            self.ep_tmp_func.x.array[ep_start : ep_start + self.ep_dim] = np.array(h_new.eps_plastic).flatten()
    
        # Triggered communication for ghost cells
        self.alpha_tmp_func.x.scatter_forward()
        self.ep_tmp_func.x.scatter_forward()
        self.macro_stress_field.x.scatter_forward()
        self.macro_tangent_field.x.scatter_forward()
        
        # Ensure linear algebra layer sees the updated ghost values across partitions
        for f in [self.macro_stress_field, self.macro_tangent_field]:
            if hasattr(f.x, "petsc_vec"):
                f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

    def advance(self):
        # Commit staging buffers into converged historical state fields
        self.alpha_old_func.x.array[:] = self.alpha_tmp_func.x.array[:]
        self.ep_old_func.x.array[:]    = self.ep_tmp_func.x.array[:]

# class macroscaleQuadratureMap:
#     def __init__(self, macro_domain, u_macro, W_macro_tensor, 
#                  macro_stress_field, W_macro_tangent, macro_tangent_field,
#                  prnn_model, prnn_params, prnn_material):
        
#         from dolfinx import fem
#         import ufl
        
#         self.macro_domain = macro_domain
#         self.comm = macro_domain.comm
#         self.u_macro = u_macro
#         self.macro_stress_field = macro_stress_field
#         self.macro_tangent_field = macro_tangent_field
        
#         # JAX components
#         self.model = prnn_model
#         self.params = prnn_params
#         self.material = prnn_material
        
#         # Pre-compile JIT functions for speed
#         self._stress_and_tangent_jit = jax.jit(self._compute_stress_and_tangent)

#         self.gdim = self.macro_domain.topology.dim
#         self.local_cells_count = self.macro_domain.topology.index_map(self.gdim).size_local

#         # History initialization
#         self.n_matpts = prnn_model.n_matpts  # Get directly from model
        
#         # 1 material point has 6 plastic strain components and 1 scalar eq. strain
#         self.ep_dim = self.n_matpts * 6 
#         self.alpha_dim = self.n_matpts * 1  
        
#         # Re-initialize spaces with the correct block sizes per cell
#         history_space = fem.functionspace(self.macro_domain, ("DG", 0, (self.alpha_dim,)))
#         ep_space      = fem.functionspace(self.macro_domain, ("DG", 0, (self.ep_dim,)))
        
#         self.alpha_old_func = fem.Function(history_space)
#         self.alpha_tmp_func = fem.Function(history_space)
#         self.ep_old_func    = fem.Function(ep_space)
#         self.ep_tmp_func    = fem.Function(ep_space)

#         # Strain Projection
#         eps = ufl.sym(ufl.grad(self.u_macro))
#         strain_vec = ufl.as_vector([eps[0, 0], eps[1, 1], eps[0, 1]])
#         self.strain_expr = fem.Expression(strain_vec, W_macro_tensor.element.interpolation_points)
#         self.local_strain_field = fem.Function(W_macro_tensor)

#         self.stress_bs = W_macro_tensor.dofmap.index_map_bs
#         self.tangent_bs = W_macro_tangent.dofmap.index_map_bs

#     def _compute_stress_and_tangent(self, params, eps_in, material, h_old):
#         # Forward pass
#         stress, h_new = self.model.apply(params, eps_in, material, h_old)
#         # Jacobian (Tangent)
#         tangent = jax.jacfwd(lambda e: self.model.apply(params, e, material, h_old)[0])(eps_in)
#         return stress, tangent, h_new

#     def update(self):
#         self.local_strain_field.interpolate(self.strain_expr)
#         self.local_strain_field.x.scatter_forward()
#         from jax_j2 import HistState
        
#         b_size = 1
#         m_pts = self.model.n_matpts
#         expected_batch = b_size * m_pts
        
#         for local_idx in range(self.local_cells_count):
#             s_start, t_start = local_idx * self.stress_bs, local_idx * self.tangent_bs
#             h_start = local_idx * self.alpha_old_func.function_space.dofmap.index_map_bs
#             ep_start = local_idx * self.ep_old_func.function_space.dofmap.index_map_bs
            
#             # Input
#             E_macro = self.local_strain_field.x.array[s_start : s_start + self.stress_bs]
#             raw_alpha = self.alpha_old_func.x.array[h_start : h_start + self.alpha_dim]
#             raw_ep = self.ep_old_func.x.array[ep_start : ep_start + self.ep_dim]
            
#             h_old = HistState(
#                 eps_plastic=jnp.array(raw_ep).reshape(expected_batch, 6),
#                 eps_p_eq=jnp.array(raw_alpha).reshape(expected_batch)
#             )
            
#             eps_in = jnp.array(E_macro).reshape(b_size, 1, -1)
            
#             # Call JIT function
#             print('Element ', local_idx, flush = True)
#             stress, tangent, h_new = self._stress_and_tangent_jit(self.params, eps_in, self.material, h_old)
#             print('Strain ', eps_in)
#             print('Stress ', stress)
#             print('Tangent ', np.array(tangent).flatten())
            
#             # Output writing
#             self.macro_stress_field.x.array[s_start : s_start + self.stress_bs] = np.array(stress).flatten()
#             self.macro_tangent_field.x.array[t_start : t_start + self.tangent_bs] = np.array(tangent).flatten()
            
#             # Secure your FEniCSx function updates:
#             self.alpha_tmp_func.x.array[h_start : h_start + self.alpha_dim] = np.array(h_new.eps_p_eq).flatten()
#             self.ep_tmp_func.x.array[ep_start : ep_start + self.ep_dim] = np.array(h_new.eps_plastic).flatten()
    
#         # Triggered communication for ghosts
#         self.alpha_tmp_func.x.scatter_forward()
#         self.ep_tmp_func.x.scatter_forward()
#         self.macro_stress_field.x.scatter_forward()
#         self.macro_tangent_field.x.scatter_forward()
        
#         # Ensure linear algebra layer sees the updated ghost values
#         for f in [self.macro_stress_field, self.macro_tangent_field]:
#             if hasattr(f.x, "petsc_vec"):
#                 f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

#     def advance(self):
#         self.alpha_old_func.x.array[:] = self.alpha_tmp_func.x.array[:]
#         self.ep_old_func.x.array[:]    = self.ep_tmp_func.x.array[:]