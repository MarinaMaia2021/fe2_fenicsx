#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 09:38:17 2026

@author: malvesmaia
"""

class Macromodel:
    def __init__(self, L = 1, H = 1, nx = 1, ny = 1,
                 verbose = False,
                 max_it=20, rtol=1e-4, atol=1e-5):
        self.L = L
        self.H = H
        self.nx = nx
        self.ny = ny
        self.gdim = 2
        self.fdim = 1
        
    def _create_mesh(self):        
        from dolfinx import mesh
        from mpi4py import MPI
        import numpy as np
        
        self.domain = mesh.create_rectangle(
            MPI.COMM_WORLD, 
            [np.array([0.0, 0.0]), np.array([self.L, self.H])], 
            [self.nx, self.ny], 
            mesh.CellType.quadrilateral,
            ghost_mode=mesh.GhostMode.shared_facet 
        )
    
        self.domain.topology.create_connectivity(self.fdim, self.gdim)
        
    def _create_mesh_with_gmsh(self, msh_filename="macro_mesh.msh"):
            """Generates a structured quad mesh using Gmsh, exports it as MSH 2.2 ASCII,
            and loads it into DOLFINx.
            """
            import gmsh
            from dolfinx.io.gmsh import model_to_mesh
            from mpi4py import MPI

            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)  
            gmsh.model.add("macro_domain")
    
            p1 = gmsh.model.geo.addPoint(0.0, 0.0, 0.0)
            p2 = gmsh.model.geo.addPoint(self.L, 0.0, 0.0)
            p3 = gmsh.model.geo.addPoint(self.L, self.H, 0.0)
            p4 = gmsh.model.geo.addPoint(0.0, self.H, 0.0)
    
            l1 = gmsh.model.geo.addLine(p1, p2)
            l2 = gmsh.model.geo.addLine(p2, p3)
            l3 = gmsh.model.geo.addLine(p3, p4)
            l4 = gmsh.model.geo.addLine(p4, p1)
    
            cl = gmsh.model.geo.addCurveLoop([l1, l2, l3, l4])
            s = gmsh.model.geo.addPlaneSurface([cl])
    
            # Transfinite curves define node density per edge
            gmsh.model.geo.mesh.setTransfiniteCurve(l1, self.nx + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(l3, self.nx + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(l2, self.ny + 1)
            gmsh.model.geo.mesh.setTransfiniteCurve(l4, self.ny + 1)
    
            # Transfinite surface forces structured quad layout
            gmsh.model.geo.mesh.setTransfiniteSurface(s)
            gmsh.model.geo.mesh.setRecombine(2, s)  # Convert triangles to quads
    
            gmsh.model.geo.synchronize()
    
            # Add physical surface so DOLFINx reads the 2D cell region
            gmsh.model.addPhysicalGroup(self.gdim, [s], tag=1, name="domain")
    
            # Generate 2D mesh
            gmsh.model.mesh.generate(self.gdim)
    
            # Generate mesh from model
            mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=self.gdim)
            self.domain = mesh_data.mesh
            self.cells = mesh_data.cell_tags
            self.facets = mesh_data.facet_tags
            
            # Print mesh (for debugging)
            gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
            gmsh.write("macro_mesh.msh")
            
            gmsh.finalize()    
            
    def _setup_functions(self):
        from dolfinx import fem
        import ufl
        
        self.V = fem.functionspace(self.domain, ("P", 1, (self.gdim,)))
        self.u = fem.Function(self.V, name="Macro_Displacement")

        self.W_tensor = fem.functionspace(self.domain, ("DG", 0, (3,)))
        self.stress_field = fem.Function(self.W_tensor, name="Macro_Stress")

        self.W_tangent = fem.functionspace(self.domain, ("DG", 0, (9,)))
        self.tangent_field = fem.Function(self.W_tangent, name="Macro_Tangent")
        
        # Declare test/trial functions
        self.u_test = ufl.TestFunction(self.V)
        self.u_trial = ufl.TrialFunction(self.V)
                
    def _setup_bc(self):
        from dolfinx import fem, default_scalar_type
        import numpy as np
        
        self.applied_pull = fem.Constant(self.domain, default_scalar_type(0.0))

        # Initialize Dirichlet conditions
        self.bcs = []

        # Get all dofs at x = 0 
        dofs_l = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0))
        # Constrain dx and dy on dofs at x = 0 
        self.bcs.append(fem.dirichletbc(np.array([0.0, 0.0], dtype=default_scalar_type), dofs_l, self.V))

        # Get all x dofs 
        V_mx, _ = self.V.sub(0).collapse()
        # Get all y dofs
        V_my, _ = self.V.sub(1).collapse()
        # Get dx dofs only at x = L
        dofs_rx, _ = fem.locate_dofs_geometrical((self.V.sub(0), V_mx), lambda x: np.isclose(x[0], self.L))
        # Get dy dofs only at x = L and y = 0
        dofs_ry, _ = fem.locate_dofs_geometrical((self.V.sub(1), V_my), lambda x: np.isclose(x[0], self.L) & np.isclose(x[1], 0.0))

        # Add it to the boundary conditions set
        self.bcs.append(fem.dirichletbc(fem.Constant(self.domain, default_scalar_type(0.0)), dofs_ry, self.V.sub(1)))
        
        # Add constraints to the boundary conditions set
        self.bcs.append(fem.dirichletbc(self.applied_pull, dofs_rx, self.V.sub(0)))
 