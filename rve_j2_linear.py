"""
RVE homogenization where the unit cell is governed by the
non-linear (Voce hardening) plane-stress J2 model implemented in j2_jax.py
--------------------------------------------------

The j2_jax.py model performs an *iterative* Newton-Raphson return mapping
(jax.lax.scan over STATIC_MAX_ITER_YIELD steps) with non-linear hardening.
That control flow cannot be expressed as a UFL expression, so it cannot be
embedded in the weak form symbolically. Instead we use the standard
"external constitutive law on a Quadrature function space" pattern used in
the dolfinx plasticity demos:
    
  1. Total strain (voigt) is interpolated onto a Quadrature space
     at every Newton iteration.
  2. j2_jax.update_vectorized(...) is called (vmapped + jitted) to get the
     updated stress and trial plastic history at every quadrature point.
  3. jax.jacfwd is used to autodiff *through* the return-mapping Newton
     solve and get the exact consistent tangent d(sigma)/d(eps) at every
     quadrature point -- no manual tangent derivation needed.
  4. Stress and tangent are written into Quadrature Functions; the FE
     residual/Jacobian forms are then *linear* algebraic operations on
     those Functions, so dolfinx_mpc can assemble them normally.
  5. Because the constitutive update lives outside UFL, the non-linear
     solve is a hand-rolled Newton loop (not PETSc SNES) using
     dolfinx_mpc.assemble_vector / assemble_matrix directly.
  6. The homogenized tangent no longer needs ufl.derivative(): since
     eps_tot = Eps_macro + sym(grad(v)) is affine in the macro strain, the
     macro-strain perturbation enters the linear(ized) system exactly the
     same way grad(dv) does. We reuse the converged, factorized Jacobian
     and solve 3 perturbation RHS vectors built from Ct_q directly.
"""

from mpi4py import MPI
import numpy as np
import contextlib
import ufl
import basix
import basix.ufl
from dolfinx import fem, default_scalar_type
from dolfinx.io.gmsh import model_to_mesh
import dolfinx_mpc
import gmsh
import jax
import jax.numpy as jnp
from jax import vmap
from petsc4py import PETSc

# PETSc here is built in double precision (default_scalar_type == float64),
# so JAX must compute in float64 too, or the Newton return-map tolerance
# (material.return_map_tol = 1e-7) is meaningless and the homogenized
# tangent will be noisy.
jax.config.update("jax_enable_x64", True)

from j2_jax import create_material, update_single_point, constitutive_update_batch, HistState
                    
# JAX helpers

def _stress_only(eps_new, eps_p_hist, eps_p_eq_hist, material):
    stress, _, _ = update_single_point(eps_new, eps_p_hist, eps_p_eq_hist, material)
    return stress

_tangent_single = jax.jacfwd(_stress_only, argnums=0)

update_and_history_batched = jax.jit(
    vmap(update_single_point, in_axes=(0, 0, 0, None))
)

tangent_batched = jax.jit(
    vmap(_tangent_single, in_axes=(0, 0, 0, None))
)

class Micromodel:
    def __init__(self,
        num_macro_cells,
        E=3.13e3,
        nu=0.37,
        sig0=31.2,
        sigu=64.8,
        b=1/0.003407,
        quadrature_degree=1,
        newton_tol = 5e-4,
        rel_newton_tol = 1-3,
        newton_max_it=20,
    ):
        # Geometric layout properties 
        self.Lx, self.Ly = 1.0, 1.0
        self.corners = np.array([[0.0, 0.0], [self.Lx, 0.0], [self.Lx, self.Ly], [0, self.Ly]])
        self.a1 = self.corners[1, :] - self.corners[0, :]
        self.a2 = self.corners[3, :] - self.corners[0, :]
        
        # Build micromodel mesh with 2 elements and periodic boundary topology 
        gdim, fdim = 2, 1
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.option.setNumber("Mesh.Algorithm", 6)
        occ = gmsh.model.occ

        unit_cell_tag = occ.add_rectangle(0.0, 0.0, 0.0, self.Lx, self.Ly)
        occ.synchronize()

        bottom_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, self.Lx + 0.01, 0.01, 0.01, fdim)
        right_edges = gmsh.model.getEntitiesInBoundingBox(self.Lx - 0.01, -0.01, -0.01, self.Lx + 0.01, self.Ly + 0.01, 0.01, fdim)
        top_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, self.Ly - 0.01, -0.01, self.Lx + 0.01, self.Ly + 0.01, 0.01, fdim)
        left_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, 0.01, self.Ly + 0.01, 0.01, fdim)

        bottom_tags = [tag for _, tag in bottom_edges]
        right_tags = [tag for _, tag in right_edges]
        top_tags = [tag for _, tag in top_edges]
        left_tags = [tag for _, tag in left_edges]

        all_edges = bottom_tags + right_tags + top_tags + left_tags
        
        for edge in all_edges:
            gmsh.model.mesh.setTransfiniteCurve(edge, 2)
        
        gmsh.model.mesh.setTransfiniteSurface(unit_cell_tag, "Left", [1, 2, 3, 4])

        translation_right = [1, 0, 0, self.Lx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for l_tag, r_tag in zip(left_tags, right_tags):
            gmsh.model.mesh.setPeriodic(fdim, [r_tag], [l_tag], translation_right)

        translation_top = [1, 0, 0, 0, 0, 1, 0, self.Ly, 0, 0, 1, 0, 0, 0, 0, 1]
        for b_tag, t_tag in zip(bottom_tags, top_tags):
            gmsh.model.mesh.setPeriodic(fdim, [t_tag], [b_tag], translation_top)

        # Single physical group (entire RVE = J2 material)
        gmsh.model.addPhysicalGroup(gdim, [unit_cell_tag], 1, name="Matrix")
        gmsh.model.addPhysicalGroup(fdim, bottom_tags, 1, name="bottom")
        gmsh.model.addPhysicalGroup(fdim, right_tags, 2, name="right")
        gmsh.model.addPhysicalGroup(fdim, top_tags, 3, name="top")
        gmsh.model.addPhysicalGroup(fdim, left_tags, 4, name="left")

        # Generate model
        gmsh.model.mesh.generate(gdim)

        # Generate mesh from model
        mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=gdim)
        self.mesh = mesh_data.mesh
        self.cells = mesh_data.cell_tags
        self.facets = mesh_data.facet_tags
        gmsh.finalize()

        # Compute volume of micromodel
        self.vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=self.mesh)))

        # Create material
        self.material = create_material(E=E, nu=nu, sig0=sig0, sigu=sigu, b=b)
        self.newton_tol = newton_tol
        self.rel_newton_tol = rel_newton_tol
        self.newton_max_it = newton_max_it

        # Create quadratures (stress, tangent and history variables)
        cell_name = self.mesh.topology.cell_name()
        q_degree = quadrature_degree
        
        Qe_vec3 = basix.ufl.quadrature_element(cell_name, value_shape=(3,), degree=q_degree)
        Qe_mat3 = basix.ufl.quadrature_element(cell_name, value_shape=(3, 3), degree=q_degree)
        Qe_vec6 = basix.ufl.quadrature_element(cell_name, value_shape=(6,), degree=q_degree)
        Qe_sc = basix.ufl.quadrature_element(cell_name, value_shape=(1,), degree=q_degree)

        self.Qv = fem.functionspace(self.mesh, Qe_vec3)  # stress
        self.QT = fem.functionspace(self.mesh, Qe_mat3)  # tangent
        self.Qh = fem.functionspace(self.mesh, Qe_vec6)  # plastic strains 
        self.Qs = fem.functionspace(self.mesh, Qe_sc)    # equivalent plastic strain

        self.dx_q = ufl.Measure(
            "dx", domain=self.mesh,
            metadata={"quadrature_degree": q_degree, "quadrature_scheme": "default"},
        )

        # Quadrature point
        # Current total strain
        self.eps_q = fem.Function(self.Qv, name="eps_tot")
        # Current stress 
        self.sigma_q = fem.Function(self.Qv, name="sigma")
        # Tangent stiffness matrix
        self.Ct_q = fem.Function(self.QT, name="Ct")
        # History (previously converged)
        self.ep_old = fem.Function(self.Qh, name="ep_old")
        self.ep_eq_old = fem.Function(self.Qs, name="ep_eq_old")
        self.ep_old.x.array[:] = 0.0
        self.ep_eq_old.x.array[:] = 0.0
        # History (current)
        self.ep_curr = fem.Function(self.Qh, name="ep_curr")
        self.ep_eq_curr = fem.Function(self.Qs, name="ep_eq_curr")
        
        # Number of quadrature points
        self.n_qp = self.ep_eq_old.x.array.shape[0]

        # Macro strains 
        self.V0 = fem.functionspace(self.mesh, ("DG", 0))
        self.E_xx_macro = fem.Function(self.V0)
        self.E_yy_macro = fem.Function(self.V0)
        self.E_xy_macro = fem.Function(self.V0)

        # Displacement fluctuation space
        self.V = fem.functionspace(self.mesh, ("P", 1, (gdim,)))
        self.u_ = ufl.TestFunction(self.V)
        self.du = ufl.TrialFunction(self.V)

        self.u = fem.Function(self.V, name="Displacement")
        self.v = fem.Function(self.V, name="Periodic_fluctuation")

        # Periodic boundary constraints (PBC)       
        # Constrain corner node at x = 0 y = 0
        self.bcs = []
        dof_00 = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
        
        def periodic_boundary(x):
            return np.isclose(x[0], 0.0)
    
        def periodic_relation(x):
            out_x = np.zeros(x.shape)
            out_x[0] = 1.0 - x[0]
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x
    
        # Apply PBC
        self.mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        self.mpc.create_periodic_constraint_geometrical(self.V, periodic_boundary, periodic_relation, [])
        self.mpc.finalize()

        # Strain definition
        def voigt(g):
            s = ufl.sym(g)
            return ufl.as_vector([s[0, 0], s[1, 1], 2.0 * s[0, 1]])

        # Define macroscopic strain (average + periodic)
        Eps_macro = ufl.as_tensor([[self.E_xx_macro, self.E_xy_macro], [self.E_xy_macro, self.E_yy_macro]])
        self.eps_tot_form = ufl.as_vector(
            [Eps_macro[0, 0], Eps_macro[1, 1], 2.0 * Eps_macro[0, 1]]
        ) + voigt(ufl.grad(self.v))
        self.eps_tot_expr = fem.Expression(self.eps_tot_form, self.Qv.element.interpolation_points)
       
        # To use in the weak form definition
        eps_test = voigt(ufl.grad(self.u_))
        eps_trial = voigt(ufl.grad(self.du))

        # Residual (virtual work) and tangent forms
        self.F_form = ufl.inner(self.sigma_q, eps_test) * self.dx_q
        self.J_form = ufl.inner(ufl.dot(self.Ct_q, eps_trial), eps_test) * self.dx_q

        # Voigt unit perturbation vectors for macro strain components
        self._voigt_basis = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, .5]),
        ]

        # PETSc objects used across Newton iterations
        rank = MPI.COMM_WORLD.Get_rank()
        self._prefix = f"rve_rank{rank}_"
        

    # ------------------------------------------------------------------
    # Constitutive update
    # ------------------------------------------------------------------
    def _update_constitutive_fields(self):
        self.v.x.scatter_forward()                  # Updating fluctuation field
        self.eps_q.interpolate(self.eps_tot_expr)   # Compute strain at 
                                                    # integration point using expression within ()

        # Reshape jaxnumpy arrays
        eps_j = self.eps_q.x.array.reshape(self.n_qp, 3)
        ep_hist_j = self.ep_old.x.array.reshape(self.n_qp, 6)
        ep_eq_hist_j = self.ep_eq_old.x.array.reshape(self.n_qp)

        # Call material model to obtain stress, history variables and tangent         
       # print('Get stresses', flush = True)
        stress_j, new_ep_j, new_ep_eq_j = update_and_history_batched(
            eps_j, ep_hist_j, ep_eq_hist_j, self.material)
        
     #   print('Get tangent', flush = True)
        Ct_j = tangent_batched(eps_j, ep_hist_j, ep_eq_hist_j, self.material)

        # Check if there is any NaN
        if jnp.any(jnp.isnan(stress_j)):
            print('\nNan was detected.')
            np.set_printoptions(formatter={'float': lambda x: "{0:0.1f}".format(x)})
            print(stress_j)
            np.set_printoptions(formatter={'float': lambda x: "{0:0.5f}".format(x)})
            print(new_ep_j)
            print(new_ep_eq_j)
            raise Exception("Nan")
            
       # print(Ct_j)         
         # Reshaping stuff
        self.sigma_q.x.array[:] = stress_j.reshape(-1) 
        self.Ct_q.x.array[:] =  Ct_j.reshape(-1) 

        # Trial plastic state at the *current* (last Newton iteration's)
        self._trial_new_ep = new_ep_j.reshape(-1)
        self._trial_new_ep_eq = new_ep_eq_j.reshape(-1)

    # ------------------------------------------------------------------
    # Newton solve for the micro-fluctuation field v
    # ------------------------------------------------------------------
    def _solve_micro_newton(self):
        # Wrap F and J in form objects
        F_compiled = fem.form(self.F_form)
        J_compiled = fem.form(self.J_form)

        du = fem.Function(self.V)
        
        # Access underlying PETSc vector for the increment
        du_petsc = du.x.petsc_vec if hasattr(du.x, "petsc_vec") else du.x.petsc_vector

        for it in range(self.newton_max_it):
            # Update sigma_q and Ct_q based on the current fluctuation 'v'
            self._update_constitutive_fields()
            
            # Assemble residual vector and jacobian matrix with MPC constraints
            b_mpc = dolfinx_mpc.assemble_vector(F_compiled, constraint=self.mpc)
            
            # Apply lifting for Dirichlet BCs (scale=0.0 because it's a residual update)
            dolfinx_mpc.apply_lifting(b_mpc, [J_compiled], [self.bcs], constraint=self.mpc, scale=0.0)
            b_mpc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b_mpc, self.bcs, alpha=0.0)
            
            # Handling different versions of petsc vec/vectors 
            if hasattr(b_mpc, "petsc_vec"):
                b_petsc = b_mpc.petsc_vec
            elif hasattr(b_mpc, "petsc_vector"):
                b_petsc = b_mpc.petsc_vector
            else:
                b_petsc = b_mpc
                
            # Compute residual norm for convergence tracking
           # res_norm = b_petsc.norm(0)
            res_norm = b_petsc.norm(1)
            if it == 0:
                res_norm0 = res_norm if res_norm != 0 else 1.0
                
            # Print current state information
        #    max_alpha = np.max(self._trial_new_ep_eq) if len(self._trial_new_ep_eq) > 0 else 0.0
        #    print(f"micro it {it:.0f} force residual: {res_norm:.3e}", flush=True)
              
            # Convergence check 
            if res_norm < self.newton_tol or (res_norm / res_norm0) < self.rel_newton_tol:
                 self.ep_curr.x.array[:] = self._trial_new_ep
                 self.ep_eq_curr.x.array[:] = self._trial_new_ep_eq
                 return True, it
            
            # Assemble micromodel tangent matrix
            A_petsc = dolfinx_mpc.assemble_matrix(J_compiled, constraint=self.mpc, bcs=self.bcs)
            A_petsc.assemble()

            # Set up PETSc linear solver and solve for increment 'du'
            ksp = PETSc.KSP().create(self.mesh.comm)
            ksp.setOperators(A_petsc)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            
            du.x.array[:] = 0.0
            ksp.solve(b_petsc, du_petsc)
            
            # # 5. Apply MPC restrictions to the incremental correction step
            # self.v.x.array[:] -= du.x.array[:]
            # self.v.x.scatter_forward()

            # self.mpc.homogenize(du)
            # self.mpc.backsubstitution(du)
            # du.x.scatter_forward()
            
            # Check convergence
            reason = ksp.getConvergedReason()
            if reason < 0: # Negative values indicate divergence or failure
                ksp.destroy()
                A_petsc.destroy()
                return False, it
            
            # Clean up PETSc structures immediately to save memory
            ksp.destroy()
            A_petsc.destroy()
                
            # Homogenize
            self.mpc.homogenize(du)
            self.mpc.backsubstitution(du)
            du.x.scatter_forward()
            
            # self.v.x.array[:] -= du.x.array[:]
            # self.v.x.scatter_forward()
            
            self.v.x.petsc_vec.axpy(-1.0, du.x.petsc_vec)
            self.v.x.scatter_forward()
            
        return False, self.newton_max_it

    def advance(self):
        """Manages history of material model in combination with macroscopic
        quadrature. Similar to commit in jive."""
        self.ep_old.x.array[:] = self.ep_curr.x.array
        self.ep_eq_old.x.array[:] = self.ep_eq_curr.x.array

    def evaluate_homogenized_properties(self, E_macro_vector, ep_old_slice, 
                                        ep_eq_old_slice, v_init=None):
        """Pushes current macro strains, solves for micro-fluctuations via
        the JAX-based J2 return map, and extracts the homogenized tangent
        by reusing the converged consistent-tangent operator."""

        # Update macroscopic strains
        self.ep_old.x.array[:] = ep_old_slice
        self.ep_eq_old.x.array[:] = ep_eq_old_slice
    
        self.E_xx_macro.x.array[:] = E_macro_vector[0]
        self.E_yy_macro.x.array[:] = E_macro_vector[1]
        self.E_xy_macro.x.array[:] = E_macro_vector[2] 

        self.E_xx_macro.x.scatter_forward()
        self.E_yy_macro.x.scatter_forward()
        self.E_xy_macro.x.scatter_forward()
        
        self.ep_old.x.scatter_forward()
        self.ep_eq_old.x.scatter_forward()
        
        # Update microscopic displacement field based on passed converged solution
        if v_init is not None:
            self.v.x.array[:] = v_init
        else:
            self.v.x.array[:] = 0.0
                        
        # Solve microscopic problem
        converged, n_it = self._solve_micro_newton()
        
        # Check convergence
        if not converged:
            print(f"micro Newton solve did not converge "
                  f"in {n_it} iterations.")

        # Compute homogenized stress vector 
        Sigma_out = np.zeros(3)
        for idx, e_i in enumerate(self._voigt_basis):
            e_const = fem.Constant(self.mesh, np.asarray(e_i, dtype=default_scalar_type))
            val = fem.assemble_scalar(fem.form(ufl.inner(self.sigma_q, e_const) * self.dx_q)) / self.vol
            Sigma_out[idx] = val

        # Compute homogenized tangent
        C_tangent = np.zeros((3, 3))
        with contextlib.redirect_stdout(None):
            J_compiled = fem.form(self.J_form)
            A_petsc = dolfinx_mpc.assemble_matrix(J_compiled, constraint=self.mpc, bcs=self.bcs)
            A_petsc.assemble()
            
            # Define solver parameters
            ksp = PETSc.KSP().create(self.mesh.comm)
            ksp.setOperators(A_petsc)
            ksp.setType(PETSc.KSP.Type.PREONLY)
            pc = ksp.getPC()
            pc.setType(PETSc.PC.Type.LU)
            pc.setFactorSolverType("mumps")
            ksp.setOptionsPrefix(f"micro_rank_{MPI.COMM_WORLD.Get_rank()}_")
            ksp.setFromOptions()

            # Fluctuation field
            dv_sol = fem.Function(self.V)
            dv_petsc = dv_sol.x.petsc_vec if hasattr(dv_sol.x, "petsc_vec") else dv_sol.x.petsc_vector

            for col_idx, e_col in enumerate(self._voigt_basis):
                dv_sol.x.array[:] = 0.0
                
                e_col_const = fem.Constant(self.mesh, np.asarray(e_col, dtype=default_scalar_type))

                # RHS = -(d residual / d E_macro)_col = -inner(Ct_q . e_col, eps_test) dx
                eps_test = ufl.as_vector(
                    [ufl.sym(ufl.grad(self.u_))[0, 0],
                     ufl.sym(ufl.grad(self.u_))[1, 1],
                     2.0 * ufl.sym(ufl.grad(self.u_))[0, 1]]
                )
                L_pert = -ufl.inner(ufl.dot(self.Ct_q, e_col_const), eps_test) * self.dx_q
                b_form = fem.form(L_pert)

                b_mpc = dolfinx_mpc.assemble_vector(b_form, constraint=self.mpc)
                dolfinx_mpc.apply_lifting(b_mpc, [J_compiled], [self.bcs], constraint=self.mpc, scale=0.0)
                b_mpc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
              #  fem.petsc.set_bc(b_mpc, self.bcs, alpha=0.0)

                b_petsc = (
                    b_mpc.petsc_vec if hasattr(b_mpc, "petsc_vec")
                    else b_mpc.petsc_vector if hasattr(b_mpc, "petsc_vector")
                    else b_mpc
                )

                ksp.solve(b_petsc, dv_petsc)
                
                # Homogenized
                self.mpc.homogenize(dv_sol)
                self.mpc.backsubstitution(dv_sol)
                
                dv_sol.x.scatter_forward()

                # Total derivative of homogenized stress w.r.t. macro strain
                # component col = average response to e_col 
                dsigma_dE_col = ufl.dot(self.Ct_q, e_col_const) 
              
                for row_idx, e_row in enumerate(self._voigt_basis):
                    e_row_const = fem.Constant(self.mesh, np.asarray(e_row, dtype=default_scalar_type))
                    val_modulus = fem.assemble_scalar(
                        fem.form(ufl.inner(dsigma_dE_col, e_row_const) * self.dx_q)
                    ) / self.vol
                    C_tangent[row_idx, col_idx] = val_modulus

            # To save memory
            ksp.destroy()
            A_petsc.destroy()

        return Sigma_out, C_tangent, self.v.x.array.copy(), self.ep_curr.x.array.copy(), self.ep_eq_curr.x.array.copy()
    
        # print("\n" + "="*50)
        # print("DEBUG: Computed Homogenized Tangent Matrix C_tangent:")
        # print("="*50)
        # np.set_printoptions(precision=4, suppress=True)
        # print(C_tangent)
        # print("="*50 + "\n", flush=True)
        
        # if np.isnan(Sigma_out).any() or np.isnan(C_tangent).any():
        #     print(Sigma_out)
        #     print(C_tangent)
        #     print(f"[Rank {MPI.COMM_WORLD.Get_rank()}] NaN values detected in RVE computation.")

     