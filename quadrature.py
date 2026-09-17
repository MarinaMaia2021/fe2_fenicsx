#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 12:59:17 2026

@author: malvesmaia
"""

class macroscaleQuadratureMap:
    def __init__(self, macro_domain, master_rve, u_macro, W_macro_tensor, 
                 macro_stress_field, W_macro_tangent, macro_tangent_field,
                 verbose = False):
        
        from dolfinx import fem
        import ufl

        self.verbose = verbose
        self.macro_domain = macro_domain  
        self.master_rve = master_rve  
        self.u_macro = u_macro
        self.u_macro_converged = self.u_macro.copy()
        self.macro_stress_field = macro_stress_field
        self.macro_tangent_field = macro_tangent_field
        
        self.factor = 2.0
        
        self.comm = self.macro_domain.comm
        self.gdim = self.macro_domain.topology.dim
        self.local_cells_count = self.macro_domain.topology.index_map(self.gdim).size_local

        # Initialize history
        ep_eq_dim = self.master_rve.ep_eq_old.x.array.shape[0]
        ep_dim    = self.master_rve.ep_old.x.array.shape[0]
        
        ep_eq_space = fem.functionspace(self.macro_domain, ("DG", 0, (ep_eq_dim,)))
        ep_space      = fem.functionspace(self.macro_domain, ("DG", 0, (ep_dim,)))
        
        self.ep_eq_old_func = fem.Function(ep_eq_space)
        self.ep_eq_tmp_func = fem.Function(ep_eq_space)
        self.ep_old_func    = fem.Function(ep_space)
        self.ep_tmp_func    = fem.Function(ep_space)
        
        # Checking dimension of history vectors
        if self.verbose:
            print(ep_dim, flush = True)
            print(ep_eq_dim, flush = True)
        
        # Fluctuating field of micromodel
        self.v_dim = self.master_rve.v.x.array.shape[0]
        self.v_history = [None] * self.local_cells_count
        self.v_staging = [None] * self.local_cells_count
    
        # Defining strain for micromodel 
        eps = ufl.sym(ufl.grad(self.u_macro))
        strain_vec = ufl.as_vector([eps[0, 0], eps[1, 1], self.factor * eps[0, 1]])
        self.strain_expr = fem.Expression(strain_vec, W_macro_tensor.element.interpolation_points)
        self.local_strain_field = fem.Function(W_macro_tensor)

        # Get dimension of stress and tangent
        self.stress_bs = W_macro_tensor.dofmap.index_map_bs
        self.tangent_bs = W_macro_tangent.dofmap.index_map_bs

    def update(self, u_macro_current):
        from petsc4py import PETSc

        if self.verbose: print('Quadrature update.', flush = True)
    
        if u_macro_current is not None:
            self.u_macro.x.array[:] = u_macro_current.x.array[:]    
            self.u_macro.x.scatter_forward()
                
        # Synchronize strain and propagate to ghosts
        self.local_strain_field.interpolate(self.strain_expr)
        self.local_strain_field.x.scatter_forward()
        
        microConvergence_all = []
        rve = self.master_rve
        
        for local_idx in range(self.local_cells_count):
            if self.verbose: print('Macroscopic element idx ', local_idx, flush = True)
            s_start = local_idx * self.stress_bs
            t_start = local_idx * self.tangent_bs
            h_start = local_idx * self.ep_eq_old_func.function_space.dofmap.index_map_bs
            ep_start = local_idx * self.ep_old_func.function_space.dofmap.index_map_bs
            
            # Retrieve initial guess for fluctuation field
            if self.v_staging[local_idx] is not None:
                v_init = self.v_history[local_idx]
            else:
                v_init = self.v_staging[local_idx]
                
            # Retrieve history                
            ep_old_slice    = self.ep_old_func.x.array[ep_start : ep_start + self.ep_old_func.function_space.dofmap.index_map_bs]
            ep_eq_old_slice = self.ep_eq_old_func.x.array[h_start : h_start + self.ep_eq_old_func.function_space.dofmap.index_map_bs]
            
            # Extract strain of integration point local_idx
            E_macro = self.local_strain_field.x.array[s_start : s_start + self.stress_bs]
            if self.verbose:
                print('Homogenized strain ', E_macro, flush = True)
            
            # Compute RVE response
            microConvergence, Sigma, C_tangent, v_conv, ep_curr_slice, ep_eq_curr_slice = rve.evaluate_homogenized_properties(E_macro, ep_old_slice, ep_eq_old_slice, v_init )
            microConvergence_all.append(microConvergence)
            
            if microConvergence:
                # Update macro fields
                self.macro_stress_field.x.array[s_start : s_start + self.stress_bs] = Sigma.flatten()
                self.macro_tangent_field.x.array[t_start : t_start + self.tangent_bs] = C_tangent.flatten()
                          
                # Store new history
                self.ep_tmp_func.x.array[ep_start : ep_start + self.ep_old_func.function_space.dofmap.index_map_bs] = ep_curr_slice
                self.ep_eq_tmp_func.x.array[h_start : h_start + self.ep_eq_old_func.function_space.dofmap.index_map_bs] = ep_eq_curr_slice
                self.v_staging[local_idx] = v_conv.copy() if hasattr(v_conv, 'copy') else v_conv
        
                # Scatter ghost values
                self.ep_eq_tmp_func.x.scatter_forward()
                self.ep_tmp_func.x.scatter_forward()
                self.macro_stress_field.x.scatter_forward()
                self.macro_tangent_field.x.scatter_forward()
        
                # Ensure linear algebra layer sees the updated ghost values
                for f in [self.macro_stress_field, self.macro_tangent_field]:
                    if hasattr(f.x, "petsc_vec"):
                        f.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD)

        if self.verbose: print('Finished updating quadrature', flush = True)
        
        # Check if all micromodels have been succesfully solve
        return all(microConvergence_all)
           
    def advance(self, u_macro):
        self.u_macro_converged.x.array[:] = u_macro.x.array[:]
        self.u_macro_converged.x.scatter_forward()
        
        self.v_history = [v.copy() if v is not None else None for v in self.v_staging]
        self.v_staging = [None] * self.local_cells_count
        
        self.ep_eq_old_func.x.array[:] = self.ep_eq_tmp_func.x.array[:]
        self.ep_old_func.x.array[:]    = self.ep_tmp_func.x.array[:]
        
        self.ep_eq_old_func.x.scatter_forward()
        self.ep_old_func.x.scatter_forward()
        
        if self.verbose: print('Updating old variables with ', self.ep_old_func.x.array[:], flush=True)

    def rollback(self):
        """Discard staged state variables on iteration failure."""
        
        self.u_macro.x.array[:] = self.u_macro_converged.x.array[:]
        self.u_macro.x.scatter_forward()
        
        self.v_staging = [None] * self.local_cells_count
        self.ep_tmp_func.x.array[:] = self.ep_old_func.x.array[:]
        self.ep_eq_tmp_func.x.array[:] = self.ep_eq_old_func.x.array[:]
        self.ep_tmp_func.x.scatter_forward()
        self.ep_eq_tmp_func.x.scatter_forward()
        
        
class plottingDomain:
    def __init__(self):
        print("Initialized robust plotting function.")

    def update(self, macro_domain, gdim, u_macro, 
               macro_qmap, macro_stress_field, iStep, fixed_bounds):
                   
        import numpy as np
        from dolfinx import fem, plot
        import pyvista as pv

        comm = macro_domain.comm
        rank = comm.Get_rank()

        # Extract VTK topology and cell types from DOLFINx
        topology, cell_types, x_coords = plot.vtk_mesh(macro_domain, gdim)
        #print(topology)
        
        # Ensure coordinates are 3D for PyVista
        if x_coords.shape[1] == 2:
            coords_3d = np.zeros((x_coords.shape[0], 3), dtype=np.float64)
            coords_3d[:, :2] = x_coords
        else:
            coords_3d = x_coords

        # Extract displacement field and map to geometry nodes
        V_plot = fem.functionspace(macro_domain, ("Lagrange", 1, (macro_domain.geometry.dim,)))
        u_plot = fem.Function(V_plot)
        u_plot.interpolate(u_macro)
        
        dim = macro_domain.geometry.dim
        n_nodes = len(coords_3d)
        local_u = np.zeros((n_nodes, 3), dtype=np.float64)
        
        u_array = u_plot.x.array.reshape(-1, dim)
        n_assign = min(len(u_array), n_nodes)
        local_u[:n_assign, :dim] = u_array[:n_assign]

        # Extract cell stress values
        macro_stress_field.x.scatter_forward()
        stress_vals = macro_stress_field.x.array

        if rank == 0:
            # Construct UnstructuredGrid explicitly passing cell types
            grid = pv.UnstructuredGrid(topology, cell_types, coords_3d)
            grid.point_data["Displacement"] = local_u
            
            if len(stress_vals) >= grid.n_cells:
                grid.cell_data["Stress_xx"] = stress_vals[:grid.n_cells]

            # Warp grid by displacement
            deformed_grid = grid.warp_by_vector("Displacement", factor=1.0)

            # Setup Off-screen Plotter
            plotter = pv.Plotter(off_screen=True)

            # Using style='wireframe' or explicit edge rendering without auto-triangulation flags
            plotter.add_mesh(
                grid, 
                style="wireframe", 
                color="black", 
                line_width=2,
                label="Undeformed Mesh"
            )

            # Add primary deformed mesh
            # Note: show_edges=True draws true cell boundaries without splitting quads into triangles
            plotter.add_mesh(
                deformed_grid, 
                scalars="Stress_xx" if "Stress_xx" in grid.cell_data else None, 
                cmap="coolwarm", 
                clim = [0, 80],
                show_edges=True,
                edge_color="blue",
                line_width=2,
                opacity=0.85
            )


            # Camera setup
            plotter.view_xy()
            plotter.camera.SetParallelProjection(True)
            plotter.reset_camera(bounds=fixed_bounds)
            plotter.screenshot(f"macro_stress_{iStep:02d}.png")
            plotter.close()

            
            print(f"  [SUCCESS] Domain rendered successfully ({grid.n_cells} cells).\n", flush=True)

# class plottingDomain:
#     def __init__(self):
#         print("Initialized plotting function.")
        
#     def update(self, macro_domain, gdim, u_macro, 
#                macro_qmap, macro_stress_field, iStep, fixed_bounds):
                   
#         import numpy as np
#         from dolfinx import fem
#         import pyvista as pv
        
#         comm = macro_domain.comm
#         rank = comm.Get_rank()
#         size = comm.Get_size()

#         # Extract local coordinates (including ghost nodes to keep cells complete)
#         geom_imap = macro_domain.geometry.index_map()
#         n_all_nodes = geom_imap.size_local + geom_imap.num_ghosts
#         local_coords = macro_domain.geometry.x[:n_all_nodes].copy()

#         # Map local topology connectivity array indices into unique global geometry indices
#         geom_global_ids = geom_imap.local_to_global(np.arange(n_all_nodes, dtype=np.int32))
        
#         # Use geometry dofmap to query cell node configurations directly
#         geom_dofmap = macro_domain.geometry.dofmap
#         n_local_cells = macro_domain.topology.index_map(gdim).size_local

#         cell_conn_global = []   
#         cell_types_local = []

#         for c in range(n_local_cells):
#             # FIXED: Extract direct geometry array rows using the geometry dofmap
#             local_node_ids = geom_dofmap[c]
#             global_node_ids = geom_global_ids[local_node_ids]
            
#             cell_conn_global.append(len(local_node_ids))
#             cell_conn_global.extend(global_node_ids.tolist())
#             cell_types_local.append(5)  # VTK_TRIANGLE = 5

#         # Extract displacement using the exact layout map matching the geometry array
#         V_plot = fem.functionspace(macro_domain, ("Lagrange", 1, (macro_domain.geometry.dim,)))
#         u_plot = fem.Function(V_plot)
#         u_plot.interpolate(u_macro)
        
#         # Map flat dof vector into the physical geometry node order positions safely
#         dim = macro_domain.geometry.dim
#         local_u = np.zeros((n_all_nodes, dim))
        
#         # Look up displacement fields utilizing the index-matched coordinate registries
#         n_dofs_available = len(u_plot.x.array) // dim
#         for i in range(n_all_nodes):
#             if i < n_dofs_available:
#                 local_u[i] = u_plot.x.array[i * dim : (i + 1) * dim]
#             else:
#                 local_u[i] = 0.0

#         # Extract cell-based stresses owned locally
#         macro_stress_field.x.scatter_forward()
#         bs = macro_qmap.stress_bs
#         local_stress_xx = macro_stress_field.x.array[:n_local_cells * bs][0::bs]

#         # Package parallel data structures into rank-isolated collections
#         rank_data = {
#             "coords": local_coords,
#             "global_ids": geom_global_ids,
#             "conn": np.array(cell_conn_global, dtype=np.int64),
#             "types": np.array(cell_types_local, dtype=np.uint8),
#             "u": local_u,
#             "stress": np.array(local_stress_xx, dtype=np.float64)
#         }
        
#         all_ranks_data = comm.gather(rank_data, root=0)

#         if rank == 0:
#             # Consolidate unique coordinate points using global map registries
#             global_id_registry = {}
#             for r in range(size):
#                 g_ids = all_ranks_data[r]["global_ids"]
#                 coords = all_ranks_data[r]["coords"]
#                 u_vals = all_ranks_data[r]["u"]
#                 for i, gid in enumerate(g_ids):
#                     if gid not in global_id_registry:
#                         global_id_registry[gid] = (coords[i], u_vals[i])

#             # Sort dictionary entries numerically by global ID key sequence
#             sorted_gids = sorted(global_id_registry.keys())
#             total_unique_nodes = len(sorted_gids)
            
#             final_coords = np.zeros((total_unique_nodes, 3))
#             final_u = np.zeros((total_unique_nodes, dim))
            
#             # Create a localized translation mapping helper [Global ID -> Compact Array Index]
#             gid_to_compact_idx = {}
#             for compact_idx, gid in enumerate(sorted_gids):
#                 final_coords[compact_idx] = global_id_registry[gid][0]
#                 final_u[compact_idx] = global_id_registry[gid][1]
#                 gid_to_compact_idx[gid] = compact_idx

#             # Remap gathered cell connectivity strings to new compact indices
#             merged_conn = []
#             merged_types = []
#             merged_stress = []

#             for r in range(size):
#                 r_conn = all_ranks_data[r]["conn"]
#                 merged_types.append(all_ranks_data[r]["types"])
#                 merged_stress.append(all_ranks_data[r]["stress"])
                
#                 idx = 0
#                 while idx < len(r_conn):
#                     n_pts = r_conn[idx]
#                     merged_conn.append(n_pts)
#                     for k in range(n_pts):
#                         original_gid = r_conn[idx + 1 + k]
#                         merged_conn.append(gid_to_compact_idx[original_gid])
#                     idx += 1 + n_pts

#             final_conn = np.array(merged_conn, dtype=np.int64)
#             final_types = np.concatenate(merged_types)
#             final_stress = np.concatenate(merged_stress)

#             # Pad displacements to 3D for PyVista Warp operation
#             if dim == 2:
#                 u_padded = np.zeros((total_unique_nodes, 3))
#                 u_padded[:, :2] = final_u[:, :2]
#             else:
#                 u_padded = final_u

#             # Instantiate standard memory unstructured grid directly
#             grid = pv.UnstructuredGrid(final_conn, final_types, final_coords)
#             grid.point_data["Displacement"] = u_padded
#             grid.cell_data["Stress_xx"] = final_stress[:grid.n_cells]

#             # Render Screen Captures
#             plotter = pv.Plotter(off_screen=True)
#             deformed_grid = grid.warp_by_vector("Displacement", factor=1.0)
            
#             plotter.add_mesh(deformed_grid, scalars="Stress_xx", cmap="coolwarm", 
#                              clim=[0, 50], show_edges=True)
#             plotter.add_mesh(grid, color="black", style="wireframe", opacity=0.25)
            
#             plotter.view_xy()
#             plotter.camera.SetParallelProjection(True)
#             plotter.reset_camera(bounds=fixed_bounds)
#             plotter.screenshot(f"macro_stress_{iStep:02d}.png")
#             plotter.close()
            
#             print(f"  [SUCCESS] Full domain rendered.\n", flush=True)