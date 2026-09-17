from mpi4py import MPI
import numpy as np
import contextlib
import ufl

from dolfinx.io.gmsh import model_to_mesh
import dolfinx_mpc
from dolfinx_mpc import NonlinearProblem 
import gmsh
from petsc4py import PETSc

# Import your JAX material parameters for mapping parity
import jax_j2

class MicroRVE:        
    def __init__(self, num_macro_cells = 1):    
        from dolfinx import fem
        
        # --- 1. RVE Geometric Layout Properties ---
        self.Lx, self.Ly = 1.0, 1.0
        self.corners = np.array([[0.0, 0.0], [self.Lx, 0.0], [self.Lx, self.Ly], [0, self.Ly]])
        self.a1 = self.corners[1, :] - self.corners[0, :]  
        self.a2 = self.corners[3, :] - self.corners[0, :]  
        
        
        # --- 2. Build Micro Mesh with exactly 2 elements and periodic boundary topology ---
        gdim, fdim = 2, 1
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        
        # Create a simple rectangular surface (no inclusions)
        unit_cell_tag = occ.add_rectangle(0.0, 0.0, 0.0, self.Lx, self.Ly)
        occ.synchronize()
        
        # Extract boundary edges explicitly for periodic constraint mapping
        # Bottom edge: (0,0) to (Lx,0) | Top edge: (0,Ly) to (Lx,Ly)
        # Left edge: (0,0) to (0,Ly)   | Right edge: (Lx,0) to (Lx,Ly)
        bottom_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, self.Lx + 0.01, 0.01, 0.01, fdim)
        right_edges  = gmsh.model.getEntitiesInBoundingBox(self.Lx - 0.01, -0.01, -0.01, self.Lx + 0.01, self.Ly + 0.01, 0.01, fdim)
        top_edges    = gmsh.model.getEntitiesInBoundingBox(-0.01, self.Ly - 0.01, -0.01, self.Lx + 0.01, self.Ly + 0.01, 0.01, fdim)
        left_edges   = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, 0.01, self.Ly + 0.01, 0.01, fdim)

        bottom_tags = [tag for _, tag in bottom_edges]
        right_tags  = [tag for _, tag in right_edges]
        top_tags    = [tag for _, tag in top_edges]
        left_tags   = [tag for _, tag in left_edges]

        # Enforce exactly 1 element segment along the X and Y bounds
        # For a 1x1 grid of squares split diagonally, this creates exactly 2 triangles.
        all_edges = bottom_tags + right_tags + top_tags + left_tags
        gmsh.model.mesh.setTransfiniteCurve(all_edges[0], 2) # 2 nodes = 1 element segment
        for edge in all_edges:
            gmsh.model.mesh.setTransfiniteCurve(edge, 2)
            
        gmsh.model.mesh.setTransfiniteSurface(unit_cell_tag, "Left", [1, 2, 3, 4])

        # Enforce periodic matching between parallel edges
        translation_right = [1, 0, 0, self.Lx,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
        for l_tag, r_tag in zip(left_tags, right_tags):
            gmsh.model.mesh.setPeriodic(fdim, [r_tag], [l_tag], translation_right)

        translation_top = [1, 0, 0, 0,  0, 1, 0, self.Ly,  0, 0, 1, 0,  0, 0, 0, 1]
        for b_tag, t_tag in zip(bottom_tags, top_tags):
            gmsh.model.mesh.setPeriodic(fdim, [t_tag], [b_tag], translation_top)
        
        # Physical group for the single "Matrix" material
        gmsh.model.addPhysicalGroup(gdim, [unit_cell_tag], 1, name="Matrix")
        
        # Bound tags for boundary condition operators
        gmsh.model.addPhysicalGroup(fdim, bottom_tags, 1, name="bottom")
        gmsh.model.addPhysicalGroup(fdim, right_tags, 2, name="right")
        gmsh.model.addPhysicalGroup(fdim, top_tags, 3, name="top")
        gmsh.model.addPhysicalGroup(fdim, left_tags, 4, name="left")
        
        # Generate the structured mesh 
        gmsh.model.mesh.generate(gdim)
        
        # Convert seamlessly to DOLFINx objects
        mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=gdim)
        self.mesh = mesh_data.mesh
        self.cells = mesh_data.cell_tags
        self.facets = mesh_data.facet_tags
        gmsh.finalize()

        self.vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=self.mesh)))
        
        # --- 3. Micro Material Fields (Aligned directly with jax_j2.create_material constants) ---
        self.V0 = fem.functionspace(self.mesh, ("DG", 0))
        
        # Matrix Properties (Phase 1) vs Fibers/Inclusions (Phase 2)
        # Fibers stay strictly elastic as requested
        self.E_field = self._create_field({1: 3.13e3, 2: 74000.0}) 
        self.nu_field = self._create_field({1: 0.37, 2: 0.2})
        self.phase_mask = self._create_field({1: 1.0, 2: 1.0}) # 1 for Matrix, 0 for Fiber
        
        # Pull jax_j2 exponential hardening curve constants for Phase 1 (Matrix)
        self.sig0 = fem.Constant(self.mesh, 31.2)
        self.sigu = fem.Constant(self.mesh, 64.8)
        self.b_param = fem.Constant(self.mesh, 1.0 / 0.003407)
        
        # Elastic Compliance Operators
        self.C11 = self.E_field / (1.0 - self.nu_field**2)
        self.C12 = self.nu_field * self.E_field / (1.0 - self.nu_field**2)
        self.C33 = self.E_field / (2.0 * (1.0 + self.nu_field)) # Shear Modulus G
        
        # --- 4. History Tracking Layout Space Setup ---
        self.W_scalar = fem.functionspace(self.mesh, ("DG", 0))
        self.W_tensor = fem.functionspace(self.mesh, ("DG", 0, (6,))) # Extended to 6 components to align with jax_j2.HistState
        
        self.alpha_old = fem.Function(self.W_scalar) # Equivalent plastic strain (eps_p_eq)
        self.ep_old = fem.Function(self.W_tensor)    # Plastic strain tensor components (eps_plastic)
        
        self.alpha_curr = fem.Function(self.W_scalar)
        self.ep_curr = fem.Function(self.W_tensor)
        
        self.E_xx_macro = fem.Function(self.V0)
        self.E_yy_macro = fem.Function(self.V0)
        self.E_xy_macro = fem.Function(self.V0)
        
        # --- 5. Define Micro Structural Displacement Spaces ---
        self.V = fem.functionspace(self.mesh, ("P", 2, (gdim,)))
        self.v = fem.Function(self.V, name="Periodic_fluctuation")
        self.u_ = ufl.TestFunction(self.V)
        
        # --- 6. Periodic Boundary Constraints Configuration ---
        def periodic_relation_left_right(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - self.a1[0] 
            out_x[1] = x[1] - self.a1[1]
            out_x[2] = x[2]
            return out_x

        def periodic_relation_bottom_top(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - self.a2[0]
            out_x[1] = x[1] - self.a2[1]
            out_x[2] = x[2]
            return out_x

        dof_00 = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
        self.bcs = [fem.dirichletbc(np.array([0.0, 0.0], dtype=np.float64), dof_00, self.V)]

        V_x, _ = self.V.sub(0).collapse()
        dof_01_x, _ = fem.locate_dofs_geometrical((self.V.sub(0), V_x), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], self.Ly))
        self.bcs.append(fem.dirichletbc(fem.Constant(self.mesh, 0.0), dof_01_x, V_x))

        self.mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        self.mpc.create_periodic_constraint_topological(self.V, self.facets, 2, periodic_relation_left_right, self.bcs)
        self.mpc.create_periodic_constraint_topological(self.V, self.facets, 3, periodic_relation_bottom_top, self.bcs)
        self.mpc.finalize()

        # --- 7. Adaptive J2 Constitutive Integration (UFL implementation of jax_j2) ---
        Eps_macro = ufl.as_tensor([[self.E_xx_macro, self.E_xy_macro], [self.E_xy_macro, self.E_yy_macro]])
        ee = Eps_macro + ufl.sym(ufl.grad(self.v))
        
        # Compute Out-Of-Plane Elastic Thickness Strain component via Plane Stress formulation matching jax_j2
        eps_el_zz_tr = -self.nu_field / (1.0 - self.nu_field) * (
            (ee[0, 0] - self.ep_old[0]) + (ee[1, 1] - self.ep_old[1])
        )

        # 6D Engineering Trial Elastic Strain vector mapping format
        eps_el_tr = ufl.as_vector([
            ee[0, 0] - self.ep_old[0],          # eps_xx
            ee[1, 1] - self.ep_old[1],          # eps_yy
            eps_el_zz_tr,                       # eps_zz (Out of plane thickness calculation)
            2.0 * ee[0, 1] - self.ep_old[3],    # gamma_xy
            0.0,                                # gamma_yz
            0.0                                 # gamma_xz
        ])
        
        # Compute Trial Stress Components via Elastic Hooke Matrix
        s_tr_xx = self.C11 * eps_el_tr[0] + self.C12 * eps_el_tr[1]
        s_tr_yy = self.C12 * eps_el_tr[0] + self.C11 * eps_el_tr[1]
        s_tr_xy = self.C33 * eps_el_tr[3]
        
        # Deviatoric Stress Projections (Yield criteria paths mirroring eval_xi / update_single_point)
        A11_tr = (s_tr_xx + s_tr_yy) ** 2
        A22_tr = (s_tr_yy - s_tr_xx) ** 2
        A33_tr = s_tr_xy ** 2
        xi_tr = A11_tr / 6.0 + 0.5 * A22_tr + 2.0 * A33_tr
        
        # Exponential Hardening Curve Function from jax_j2.sigma_C
        sigma_Y_old = self.sig0 + (self.sigu - self.sig0) * (1.0 - ufl.exp(-self.b_param * self.alpha_old))
        f_tr = 0.5 * xi_tr - (sigma_Y_old ** 2) / 3.0
        
        # Return Mapping Linearization Operator denominator
        # Derivative of yield stress function at current step: H = sigma_C_deriv
        H_mod_local = (self.sigu - self.sig0) * self.b_param * ufl.exp(-self.b_param * self.alpha_old)
        fac = self.E_field / (1.0 - self.nu_field)
        
        # Approximate plastic increment calculation over step
        xi_tr_safe = ufl.conditional(ufl.gt(xi_tr, 1e-8), xi_tr, 1e-8)
        denom = xi_tr_safe / 2.0 + (2.0 * sigma_Y_old * H_mod_local / 3.0) * ufl.sqrt(xi_tr_safe / 6.0)
        dgamma_approx = ufl.conditional(ufl.gt(f_tr, 0.0), f_tr / denom, 0.0)
        dgamma = self.phase_mask * dgamma_approx
        
        # Update Flow Vectors 
        A_mat_00 = 3. * (1. - self.nu_field) / (3. * (1. - self.nu_field) + self.E_field * dgamma)
        A_mat_22 = 1. / (1. + 2. * self.C33 * dgamma)
        
        A1 = (A_mat_00 + A_mat_22) / 2.0
        A12 = (A_mat_00 - A_mat_22) / 2.0
        
        # Compute plastic update stresses or keep elastic steps
        s_p_xx = A1 * s_tr_xx + A12 * s_tr_yy
        s_p_yy = A12 * s_tr_xx + A1 * s_tr_yy
        s_p_xy = A_mat_22 * s_tr_xy
        
        self.stress_v = ufl.as_vector([
            ufl.conditional(ufl.gt(dgamma, 0.0), s_p_xx, s_tr_xx),
            ufl.conditional(ufl.gt(dgamma, 0.0), s_p_yy, s_tr_yy),
            ufl.conditional(ufl.gt(dgamma, 0.0), s_p_xy, s_tr_xy)
        ])
                                       
        self.sigma_tensor = ufl.as_tensor([[self.stress_v[0], self.stress_v[2]], [self.stress_v[2], self.stress_v[1]]])
        self.F_form = ufl.inner(self.sigma_tensor, ufl.sym(ufl.grad(self.u_))) * ufl.dx
        self.J_form = ufl.derivative(self.F_form, self.v)
        
        # Plastic Incremental Strain Tensor Updates matching ep3_to_ep6 mapping paths
        depsp_3d_xx = dgamma * (2./3. * self.stress_v[0] - 1./3. * self.stress_v[1])
        depsp_3d_yy = dgamma * (-1./3. * self.stress_v[0] + 2./3. * self.stress_v[1])
        depsp_3d_xy = dgamma * (2.0 * self.stress_v[2])
        
        # Assign values back to 6D state format
        self.alpha_expr = fem.Expression(self.alpha_old + dgamma * ufl.sqrt(2.0 * xi_tr_safe / 3.0), self.W_scalar.element.interpolation_points)
        self.ep_expr = fem.Expression(ufl.as_vector([
            self.ep_old[0] + depsp_3d_xx,
            self.ep_old[1] + depsp_3d_yy,
            self.ep_old[2] + (-depsp_3d_xx - depsp_3d_yy), # out of plane thickness strain state component
            self.ep_old[3] + depsp_3d_xy,
            0.0,
            0.0
        ]), self.W_tensor.element.interpolation_points)
        
        self.Eps_ = fem.Constant(self.mesh, np.zeros((2, 2)))

        # --- 8. AUTODIFF TANGENT PERTURBATION FORMS ---
        self.dv = ufl.TrialFunction(self.V) 
        self.dE_xx = fem.Constant(self.mesh, 0.0)
        self.dE_yy = fem.Constant(self.mesh, 0.0)
        self.dE_xy = fem.Constant(self.mesh, 0.0)
        
        dF_dE = ufl.derivative(self.F_form, self.E_xx_macro, self.dE_xx) \
              + ufl.derivative(self.F_form, self.E_yy_macro, self.dE_yy) \
              + ufl.derivative(self.F_form, self.E_xy_macro, self.dE_xy)
              
        self.L_pert = -dF_dE 
        
        self.micro_problem = NonlinearProblem(
            self.F_form, self.v, self.mpc, bcs=self.bcs, J=self.J_form
        )
        
        snes = self.micro_problem.solver
        rank = MPI.COMM_WORLD.Get_rank()
        prefix = f"rve_rank{rank}_"
        snes.setOptionsPrefix(prefix)
        
        opts = PETSc.Options()
        opts[f"{prefix}snes_type"] = "newtonls"
        opts[f"{prefix}snes_atol"] = 1e-3
        opts[f"{prefix}snes_rtol"] = 1e-3
        opts[f"{prefix}snes_max_it"] = 18
        opts[f"{prefix}snes_ksp_type"] = "preonly"
        opts[f"{prefix}snes_pc_type"] = "lu"
        snes.setFromOptions()
                
    def _create_field(self, prop_dict):
        from dolfinx import fem
        k = fem.Function(self.V0)
        for tag, val in prop_dict.items():
            cs = self.cells.find(tag)
            k.x.array[cs] = np.full_like(cs, val, dtype=np.float64)
        return k

    def evaluate_homogenized_properties(self, E_macro_vector, v_init=None):
        from dolfinx import fem
        
        """Pushes current macro strains, solves for micro-fluctuations, and extracts Tangent via Autodiff."""
        self.E_xx_macro.x.array[:] = E_macro_vector[0]
        self.E_yy_macro.x.array[:] = E_macro_vector[1]
        self.E_xy_macro.x.array[:] = E_macro_vector[2] / 2.0
        
        self.E_xx_macro.x.scatter_forward()
        self.E_yy_macro.x.scatter_forward()
        self.E_xy_macro.x.scatter_forward()
        
        if v_init is not None:
           self.v.x.array[:] = v_init
        else:
           self.v.x.array[:] = 0.0
  
    #    self.v.x.array[:] = 0.0

        self.v.x.scatter_forward()    
        self.micro_problem.solve()

        self.v.x.scatter_forward()    
        self.alpha_curr.interpolate(self.alpha_expr)
        self.ep_curr.interpolate(self.ep_expr)
        
        # --- A. Compute Homogenized Stress Vector ---
        Sigma_out = np.zeros(3)
        for idx, (i, j) in enumerate([(0,0), (1,1), (0,1)]):
            test_mat = np.zeros((2, 2)); test_mat[i, j] = 1.0; self.Eps_.value = test_mat
            val = fem.assemble_scalar(fem.form(ufl.inner(self.sigma_tensor, self.Eps_) * ufl.dx)) / self.vol
            Sigma_out[idx] = val
                    
        # --- B. AUTODIFF TANGENT STIFFNESS MATRIX GENERATION ---
        C_tangent = np.zeros((3, 3))
        
        with contextlib.redirect_stdout(None):
            A_petsc = dolfinx_mpc.assemble_matrix(fem.form(self.J_form), constraint=self.mpc, bcs=self.bcs)
            A_petsc.assemble()
            
            ksp = PETSc.KSP().create(MPI.COMM_SELF)
            ksp.setOperators(A_petsc)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            ksp.getPC().setType(PETSc.PC.Type.LU)
            ksp.setOptionsPrefix(f"micro_rank_{MPI.COMM_WORLD.Get_rank()}_")
            ksp.setFromOptions()
            
            dv_sol = fem.Function(self.V)
            dv_petsc = dv_sol.x.petsc_vec if hasattr(dv_sol.x, "petsc_vec") else dv_sol.x.petsc_vector
            
            voigt_perturbations = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1)]
            
            for col_idx, (dE_xx_val, dE_yy_val, dE_xy_val) in enumerate(voigt_perturbations):
                self.dE_xx.value = dE_xx_val
                self.dE_yy.value = dE_yy_val
                self.dE_xy.value = dE_xy_val
                
                b_form = fem.form(self.L_pert)
                b_mpc = dolfinx_mpc.assemble_vector(b_form, constraint=self.mpc)
                dolfinx_mpc.apply_lifting(b_mpc, [fem.form(self.J_form)], [self.bcs], constraint=self.mpc)
                fem.petsc.set_bc(b_mpc, self.bcs)
                
                b_petsc = b_mpc.petsc_vec if hasattr(b_mpc, "petsc_vec") else b_mpc.petsc_vector if hasattr(b_mpc, "petsc_vector") else b_mpc
                
                ksp.solve(b_petsc, dv_petsc)
                dv_sol.x.scatter_forward()
                
                dsigma_form = ufl.derivative(self.sigma_tensor, self.v, dv_sol) \
                             + ufl.derivative(self.sigma_tensor, self.E_xx_macro, self.dE_xx) \
                             + ufl.derivative(self.sigma_tensor, self.E_yy_macro, self.dE_yy) \
                             + ufl.derivative(self.sigma_tensor, self.E_xy_macro, self.dE_xy)
                
                for row_idx, (i, j) in enumerate([(0,0), (1,1), (0,1)]):
                    test_mat = np.zeros((2, 2)); test_mat[i, j] = 1.0; self.Eps_.value = test_mat
                    val_modulus = fem.assemble_scalar(fem.form(ufl.inner(dsigma_form, self.Eps_) * ufl.dx)) / self.vol
                    C_tangent[row_idx, col_idx] = val_modulus
          
            ksp.destroy()
            A_petsc.destroy()

        # print("\n" + "="*50)
        # print("DEBUG: Computed Homogenized Tangent Matrix C_tangent:")
        # print("="*50)
        # np.set_printoptions(precision=4, suppress=True)
        # print(C_tangent)
        # print("="*50 + "\n", flush=True)
        
        # if np.isnan(C_tangent).any():
        #     import sys
        #     print("FATAL: Macro tangent field is corrupted (NaNs encountered). exiting.", flush=True)
        #     sys.exit(1)
        # elif np.allclose(C_tangent, 0.0):
        #     print("WARNING: C_tangent matrix consists entirely of zeros!", flush=True)
    


        if np.isnan(Sigma_out).any() or np.isnan(C_tangent).any():
            print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] NaN values detected in RVE computation.")

        return Sigma_out, C_tangent, self.v.x.array.copy()