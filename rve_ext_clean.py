import numpy as np
from mpi4py import MPI
import ufl
import basix.ufl
from dolfinx import fem, default_scalar_type
from dolfinx.io.gmsh import model_to_mesh
import dolfinx_mpc
import gmsh
from petsc4py import PETSc

import jax
import jax.numpy as jnp
from jax import vmap

from auxFunctions import _setup_solver
import jax_j2  # Single source of truth material/return-mapping definition file

# Enforce Double Precision for JAX to match PETSc float64 solver requirements
jax.config.update("jax_enable_x64", True)


def _matrix_stress_only(eps_3d, ep_hist, ep_eq_hist, jax_mat):
    """
    Evaluates matrix 3D plane-stress response from JAX material mapping.
    Maps input 3D Voigt strains [eps_xx, eps_yy, gamma_xy] to 6D JAX inputs using 
    the plane-stress strain constraint: eps_zz = -nu / (1 - nu) * (eps_xx + eps_yy).
    Outputs reduced 3D Voigt stress [sig_xx, sig_yy, sig_xy].
    """
    nu_m = jax_mat.nu
    eps_6d = jnp.array([
        eps_3d[0],
        eps_3d[1],
        -(nu_m / (1.0 - nu_m)) * (eps_3d[0] + eps_3d[1]),
        eps_3d[2],
        0.0,
        0.0
    ], dtype=jnp.float64)
    
    stress_6d, _, _ = jax_j2.update_single_point(eps_6d, ep_hist, ep_eq_hist, jax_mat)
    
    # Return 3D Voigt stresses: [sig_xx, sig_yy, sig_xy]
    return jnp.array([stress_6d[0], stress_6d[1], stress_6d[3]], dtype=jnp.float64)


# Consistent 3x3 plane-stress tangent via autodiff through the 3D-to-6D-to-3D pipeline
_matrix_tangent_single = jax.jacfwd(_matrix_stress_only, argnums=0)

_matrix_update_batched = jax.jit(
    vmap(
        lambda eps_3d, ep, ep_eq, mat: jax_j2.update_single_point(
            jnp.array([
                eps_3d[0], 
                eps_3d[1], 
                -(mat.nu / (1.0 - mat.nu)) * (eps_3d[0] + eps_3d[1]), 
                eps_3d[2], 
                0.0, 
                0.0
            ], dtype=jnp.float64),
            ep, ep_eq, mat
        ),
        in_axes=(0, 0, 0, None)
    )
)

_matrix_tangent_batched = jax.jit(
    vmap(_matrix_tangent_single, in_axes=(0, 0, 0, None))
)


class ExternalJAXMatrixPhase:
    """Manages internal history states and macro-point tracking for the matrix phase."""
    def __init__(self, num_quad_points_per_rve, num_macro_points_total, material_properties = None):
        
        self.E_m = material_properties[0]
        self.nu_m = material_properties[1]
        self.sig0_m = material_properties[2]
        self.sigu_m = material_properties[3]
        self.b_m = material_properties[4]
        
        self.jax_material = jax_j2.create_material(self.E_m, self.nu_m, self.sig0_m, self.sigu_m, self.b_m)            
            
        self.h_states_registry = [
            jax_j2.init_history(num_quad_points_per_rve)
            for _ in range(num_macro_points_total)
        ]
        self.current_macro_pt_id = 0
        self._cached_h_state = None        

    def advance_macro_point(self):
        if self._cached_h_state is not None:
            self.h_states_registry[self.current_macro_pt_id] = self._cached_h_state


class Micromodel:
    def __init__(self, num_macro_cells=8, 
                 quadrature_degree=1, 
                 direct_solver=False, 
                 verbose=False,
                 strain_factor = 2):
        
        self.microSolverType = direct_solver
        self.verbose = verbose
        self.factor = strain_factor
        
        material_properties = self._setup_material_properties(
            E_m = 3.13e3,
            nu_m = 0.37,
            sig0_m = 31.2,
            sigu_m = 64.8,
            b_m = 1/0.003407,
            E_f = 74000.0,
            nu_f = 0.2
            )

        # Geometric layout properties
        self.Lx, self.Ly = 1.0, 1.0
        self.R = 0.15 * self.Lx
        self.h = 0.1 * self.Lx
        
        self.corners = np.array([[0.0, 0.0], [self.Lx, 0.0], [self.Lx, self.Ly], [0.0, self.Ly]])
        self.fibers_center = np.vstack([self.corners, np.array([0.5, 0.5])])

        self.a1 = self.corners[1, :] - self.corners[0, :]
        self.a2 = self.corners[3, :] - self.corners[0, :]
        
        # Build Mesh & Physical Groups via Gmsh
        gdim, fdim = 2, 1
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        occ = gmsh.model.occ
        
        points = [occ.add_point(*corner, 0) for corner in self.corners]
        lines = [occ.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
        loop = occ.add_curve_loop(lines)
        unit_cell = occ.add_plane_surface([loop])
        inclusions = [occ.add_disk(*corner, 0, self.R, self.R) for corner in self.fibers_center]
        vol_dimTag = (gdim, unit_cell)
        
        out = occ.intersect([vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False)
        incl_dimTags = out[0]
        occ.synchronize()
        occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
        occ.synchronize()
                
        bottom_edges = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, self.Lx + 0.01, 0.01, 0.01, fdim)
        right_edges  = gmsh.model.getEntitiesInBoundingBox(self.Lx - 0.01, -0.01, -0.01, self.Lx + 0.01, self.Ly + 0.01, 0.01, fdim)
        top_edges    = gmsh.model.getEntitiesInBoundingBox(-0.01, self.Ly - 0.01, -0.01, self.Lx + 0.01, self.Ly + 0.01, 0.01, fdim)
        left_edges   = gmsh.model.getEntitiesInBoundingBox(-0.01, -0.01, -0.01, 0.01, self.Ly + 0.01, 0.01, fdim)

        bottom_tags = [tag for _, tag in bottom_edges]
        right_tags  = [tag for _, tag in right_edges]
        top_tags    = [tag for _, tag in top_edges]
        left_tags   = [tag for _, tag in left_edges]

        translation_right = [1, 0, 0, self.Lx,  0, 1, 0, 0,  0, 0, 1, 0,  0, 0, 0, 1]
        for l_tag, r_tag in zip(left_tags, right_tags):
            gmsh.model.mesh.setPeriodic(fdim, [r_tag], [l_tag], translation_right)

        translation_top = [1, 0, 0, 0,  0, 1, 0, self.Ly,  0, 0, 1, 0,  0, 0, 0, 1]
        for b_tag, t_tag in zip(bottom_tags, top_tags):
            gmsh.model.mesh.setPeriodic(fdim, [t_tag], [b_tag], translation_top)
        
        gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
        gmsh.model.addPhysicalGroup(gdim, [tag for _, tag in incl_dimTags], 2, name="Inclusions")
        gmsh.model.addPhysicalGroup(fdim, bottom_tags, 1, name="bottom")
        gmsh.model.addPhysicalGroup(fdim, right_tags, 2, name="right")
        gmsh.model.addPhysicalGroup(fdim, top_tags, 3, name="top")
        gmsh.model.addPhysicalGroup(fdim, left_tags, 4, name="left")
        
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", self.h)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", self.h)
        
        gmsh.model.mesh.generate(gdim)
        
        mesh_data = model_to_mesh(gmsh.model, MPI.COMM_SELF, 0, gdim=gdim)
        self.mesh = mesh_data.mesh
        self.cells = mesh_data.cell_tags
        self.facets = mesh_data.facet_tags
        gmsh.finalize()
        
        self.vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=self.mesh)))
        
        # Identification of Matrix and Inclusion cells
        self.matrix_cells = self.cells.find(1)
        self.fiber_cells = self.cells.find(2)
        
        # Fiber: Linear Isotropic Plane-Stress Constants
        E_f, nu_f = self.E_f, self.nu_f
        c_11_f = E_f / (1.0 - nu_f**2)
        c_12_f = nu_f * E_f / (1.0 - nu_f**2)
        c_33_f = E_f / (2.0 * (1.0 + nu_f))
        
        self.C_fiber = np.array([
            [c_11_f, c_12_f, 0.0],
            [c_12_f, c_11_f, 0.0],
            [0.0,    0.0,    c_33_f]
        ], dtype=np.float64)

        # Quadrature Space Allocations
        cell_name = self.mesh.topology.cell_name()
        q_degree = quadrature_degree
        
        Qe_vec3 = basix.ufl.quadrature_element(cell_name, value_shape=(3,), degree=q_degree)
        Qe_mat3 = basix.ufl.quadrature_element(cell_name, value_shape=(3, 3), degree=q_degree)
        Qe_vec6 = basix.ufl.quadrature_element(cell_name, value_shape=(6,), degree=q_degree)
        Qe_sc   = basix.ufl.quadrature_element(cell_name, value_shape=(1,), degree=q_degree)

        self.Qv = fem.functionspace(self.mesh, Qe_vec3)
        self.QT = fem.functionspace(self.mesh, Qe_mat3)
        self.Qh = fem.functionspace(self.mesh, Qe_vec6)  # Plastic strain state tensor (6D in JAX)
        self.Qs = fem.functionspace(self.mesh, Qe_sc)    # Equivalent plastic strain (1D)

        self.dx_q = ufl.Measure("dx", domain=self.mesh, metadata={"quadrature_degree": q_degree, "quadrature_scheme": "default"})

        self.eps_q   = fem.Function(self.Qv, name="eps_tot")
        self.sigma_q = fem.Function(self.Qv, name="sigma")
        self.Ct_q    = fem.Function(self.QT, name="Ct")
        
        self.ep_old  = fem.Function(self.Qh, name="ep_old")
        self.ep_eq_old = fem.Function(self.Qs, name="ep_eq_old")
        self.ep_curr = fem.Function(self.Qh, name="ep_curr")
        self.ep_eq_curr = fem.Function(self.Qs, name="ep_eq_curr")

        self.n_qp = self.ep_eq_old.x.array.shape[0]
    
        # Extract Matrix QP Indices for Selective JAX Updating
        matrix_qp_indices = []
        dofmap = self.Qv.dofmap
        for cell_idx in self.matrix_cells:
            cell_dofs = dofmap.cell_dofs(cell_idx)
            matrix_qp_indices.extend(cell_dofs)
        self.matrix_qp_indices = np.array(matrix_qp_indices, dtype=np.int32)
        
        num_matrix_points = len(self.matrix_qp_indices)
        self.jax_operator = ExternalJAXMatrixPhase(num_matrix_points,
                                                   num_macro_cells, 
                                                   material_properties)

        # Macroscopic strain functions
        self.V0 = fem.functionspace(self.mesh, ("DG", 0))
        self.E_xx_macro = fem.Function(self.V0)
        self.E_yy_macro = fem.Function(self.V0)
        self.E_xy_macro = fem.Function(self.V0)

        # Displacement and MPC Setup
        self.V = fem.functionspace(self.mesh, ("P", 1, (gdim,)))
        self.u_ = ufl.TestFunction(self.V)
        self.du = ufl.TrialFunction(self.V)
        self.v  = fem.Function(self.V, name="Periodic_fluctuation")

        dof_00 = fem.locate_dofs_geometrical(self.V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0.0))
        self.bcs = [fem.dirichletbc(np.array([0.0, 0.0], dtype=np.float64), dof_00, self.V)]
        
        V_x, _ = self.V.sub(0).collapse()
        dof_01_x, _ = fem.locate_dofs_geometrical((self.V.sub(0), V_x), lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], self.Ly))
        self.bcs.append(fem.dirichletbc(fem.Constant(self.mesh, 0.0), dof_01_x, V_x))

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

        self.mpc = dolfinx_mpc.MultiPointConstraint(self.V)
        self.mpc.create_periodic_constraint_topological(self.V, self.facets, 2, periodic_relation_left_right, self.bcs)
        self.mpc.create_periodic_constraint_topological(self.V, self.facets, 3, periodic_relation_bottom_top, self.bcs)
        self.mpc.finalize()

        # UFL Kinematics & Residual Formulation
        def voigt(g):
            s = ufl.sym(g)
            return ufl.as_vector([s[0, 0], s[1, 1], self.factor * s[0, 1]])

        Eps_macro = ufl.as_tensor([[self.E_xx_macro, self.E_xy_macro], [self.E_xy_macro, self.E_yy_macro]])
        self.eps_tot_form = ufl.as_vector([Eps_macro[0, 0], Eps_macro[1, 1], self.factor * Eps_macro[0, 1]]) + voigt(ufl.grad(self.v))
        self.eps_tot_expr = fem.Expression(self.eps_tot_form, self.Qv.element.interpolation_points)

        eps_test = voigt(ufl.grad(self.u_))
        eps_trial = voigt(ufl.grad(self.du))

        self.F_form = ufl.inner(self.sigma_q, eps_test) * self.dx_q
        self.J_form = ufl.inner(ufl.dot(self.Ct_q, eps_trial), eps_test) * self.dx_q

        self._voigt_basis = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]

    def _setup_material_properties(self, E_m = 3.13e3, nu_m = 0.37, 
                                   sig0_m = 31.2, sigu_m = 64.8, b_m = 1/0.003407,
                                   E_f = 74000.0, nu_f = 0.2
                                   ):
        self.E_m = E_m
        self.nu_m = nu_m
        self.sig0_m = sig0_m
        self.sigu_m = sigu_m
        self.b_m = b_m
        self.E_f = E_f
        self.nu_f = nu_f
        return [self.E_m, self.nu_m, self.sig0_m, self.sigu_m, self.b_m, self.E_f, self.nu_f]

    def _update_constitutive_fields(self):
        """Updates internal multi-material fields considering 3D vs 6D dimension mappings."""
        self.v.x.scatter_forward()
        self.eps_q.interpolate(self.eps_tot_expr)

        eps_all = self.eps_q.x.array.reshape(self.n_qp, 3)
        sigma_all = np.zeros((self.n_qp, 3), dtype=np.float64)
        Ct_all = np.zeros((self.n_qp, 3, 3), dtype=np.float64)

        ep_curr_all = self.ep_old.x.array.reshape(self.n_qp, 6).copy()
        ep_eq_curr_all = self.ep_eq_old.x.array.reshape(self.n_qp).copy()

        # 1. Update Inclusions (Pure Linear Elastic)
        for i in range(self.n_qp):
            sigma_all[i] = self.C_fiber @ eps_all[i]
            Ct_all[i] = self.C_fiber

        # 2. Update Matrix Phase (JAX J2 Return Mapping)
        if len(self.matrix_qp_indices) > 0:
            m_idx = self.matrix_qp_indices
            eps_m_3d = eps_all[m_idx]
            ep_m     = self.ep_old.x.array.reshape(self.n_qp, 6)[m_idx]
            ep_eq_m  = self.ep_eq_old.x.array.reshape(self.n_qp)[m_idx]

            # Execute batch JAX return mapping & autodiff consistent tangent
            stress_m_6d, new_ep_m, new_ep_eq_m = _matrix_update_batched(
                eps_m_3d, ep_m, ep_eq_m, self.jax_operator.jax_material
            )
            Ct_m_3x3 = _matrix_tangent_batched(
                eps_m_3d, ep_m, ep_eq_m, self.jax_operator.jax_material
            )

            # Reduce JAX 6D stress tensor array to 3D Voigt components [sig_xx, sig_yy, sig_xy]
            stress_m_6d_np = np.array(stress_m_6d, dtype=np.float64)
            stress_m_3d_np = np.column_stack((
                stress_m_6d_np[:, 0],
                stress_m_6d_np[:, 1],
                stress_m_6d_np[:, 3]
            ))

            sigma_all[m_idx] = stress_m_3d_np
            Ct_all[m_idx]    = np.array(Ct_m_3x3, dtype=np.float64)

            ep_curr_all[m_idx]    = np.array(new_ep_m, dtype=np.float64)
            ep_eq_curr_all[m_idx] = np.array(new_ep_eq_m, dtype=np.float64)

        self.sigma_q.x.array[:] = sigma_all.reshape(-1)
        self.Ct_q.x.array[:]    = Ct_all.reshape(-1)

        self._trial_new_ep    = ep_curr_all.reshape(-1)
        self._trial_new_ep_eq = ep_eq_curr_all.reshape(-1)

    def _solve_micro_newton(self):
        """Solves micro-fluctuation equilibrium problem."""
        F_compiled = fem.form(self.F_form)
        J_compiled = fem.form(self.J_form)

        du = fem.Function(self.V)
        du_petsc = du.x.petsc_vec if hasattr(du.x, "petsc_vec") else du.x.petsc_vector

        self._update_constitutive_fields()

        b_mpc = dolfinx_mpc.assemble_vector(F_compiled, constraint=self.mpc)
        b_mpc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        dolfinx_mpc.apply_lifting(b_mpc, [J_compiled], [self.bcs], constraint=self.mpc, scale=0.0)
        fem.petsc.set_bc(b_mpc, self.bcs, alpha=0.0)

        b_petsc = (
            b_mpc.petsc_vec if hasattr(b_mpc, "petsc_vec")
            else b_mpc.petsc_vector if hasattr(b_mpc, "petsc_vector")
            else b_mpc
        )

        A_petsc = dolfinx_mpc.assemble_matrix(J_compiled, constraint=self.mpc, bcs=self.bcs)
        A_petsc.assemble()

        solver = _setup_solver(self.mesh, directSolver=self.microSolverType, tag='micro_')
        solver.setOperators(A_petsc)

        du.x.array[:] = 0.0
        solver.solve(b_petsc, du_petsc)

        solver.destroy()
        A_petsc.destroy()

        self.mpc.homogenize(du)
        self.mpc.backsubstitution(du)
        du.x.scatter_forward()

        self.v.x.petsc_vec.axpy(-1.0, du.x.petsc_vec)
        self.v.x.scatter_forward()

        self.ep_curr.x.array[:] = self._trial_new_ep
        self.ep_eq_curr.x.array[:] = self._trial_new_ep_eq
        return True

    def advance(self):
        """Commits micro history states upon macroscopic step convergence."""
        self.ep_old.x.array[:] = self.ep_curr.x.array
        self.ep_eq_old.x.array[:] = self.ep_eq_curr.x.array
        self.jax_operator.advance_macro_point()
        if self.verbose:
            print("Updated history of two-phase micromodel.")

    def evaluate_homogenized_properties(self, E_macro_vector, ep_old_slice, ep_eq_old_slice, v_init=None, macro_cell_id=0):
        """Pushes macro strains, computes micro solution, and extracts homogenized tangent."""
        self.jax_operator.current_macro_pt_id = macro_cell_id
        
        self.ep_old.x.array[:] = ep_old_slice
        self.ep_eq_old.x.array[:] = ep_eq_old_slice
        self.ep_old.x.scatter_forward()
        self.ep_eq_old.x.scatter_forward()

        self.E_xx_macro.x.array[:] = E_macro_vector[0]
        self.E_yy_macro.x.array[:] = E_macro_vector[1]
        self.E_xy_macro.x.array[:] = E_macro_vector[2]
        self.E_xx_macro.x.scatter_forward()
        self.E_yy_macro.x.scatter_forward()
        self.E_xy_macro.x.scatter_forward()

        if v_init is not None:
            self.v.x.array[:] = v_init
        else:
            self.v.x.array[:] = 0.0

        microConverged = self._solve_micro_newton()
        
        if not microConverged:
            return False, np.zeros(3), np.zeros((3, 3)), self.v.x.array.copy(), self.ep_curr.x.array.copy(), self.ep_eq_curr.x.array.copy()

        Sigma_out = self.compute_stress()
        C_tangent = self.compute_tangent()

        return True, Sigma_out, C_tangent, self.v.x.array.copy(), self.ep_curr.x.array.copy(), self.ep_eq_curr.x.array.copy()

    def compute_stress(self):
        """Computes homogenized macro stress vector."""
        if self.verbose:
            print("Computing homogenized stress.")
        Sigma_out = np.zeros(3)
        for idx, e_i in enumerate(self._voigt_basis):
            e_const = fem.Constant(self.mesh, np.asarray(e_i, dtype=default_scalar_type))
            val = fem.assemble_scalar(fem.form(ufl.inner(self.sigma_q, e_const) * self.dx_q)) / self.vol
            Sigma_out[idx] = val
        return Sigma_out

    def compute_tangent(self):
        """Assembles homogenized 3x3 macroscopic consistent tangent stiffness matrix."""
        if self.verbose:
            print("Computing homogenized tangent stiffness matrix.", flush=True)

        C_tangent = np.zeros((3, 3))
        J_compiled = fem.form(self.J_form)
        A_petsc = dolfinx_mpc.assemble_matrix(J_compiled, constraint=self.mpc, bcs=self.bcs)
        A_petsc.assemble()

        ksp = _setup_solver(self.mesh, directSolver=self.microSolverType, tag='micro_')
        ksp.setOperators(A_petsc)

        dv_sol = fem.Function(self.V)
        dv_petsc = dv_sol.x.petsc_vec if hasattr(dv_sol.x, "petsc_vec") else dv_sol.x.petsc_vector

        for row_idx, e_col in enumerate(self._voigt_basis):
            dv_sol.x.array[:] = 0.0
            e_col_const = fem.Constant(self.mesh, np.asarray(e_col, dtype=default_scalar_type))

            eps_test = ufl.as_vector(
                [ufl.sym(ufl.grad(self.u_))[0, 0],
                 ufl.sym(ufl.grad(self.u_))[1, 1],
                 self.factor * ufl.sym(ufl.grad(self.u_))[0, 1]]
            )
            L_pert = -ufl.inner(ufl.dot(self.Ct_q, e_col_const), eps_test) * self.dx_q
            b_form = fem.form(L_pert)

            b_mpc = dolfinx_mpc.assemble_vector(b_form, constraint=self.mpc)
            b_mpc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b_mpc, self.bcs, alpha=0.0)

            b_petsc = (
                b_mpc.petsc_vec if hasattr(b_mpc, "petsc_vec")
                else b_mpc.petsc_vector if hasattr(b_mpc, "petsc_vector")
                else b_mpc
            )

            ksp.solve(b_petsc, dv_petsc)

            self.mpc.homogenize(dv_sol)
            self.mpc.backsubstitution(dv_sol)
            dv_sol.x.scatter_forward()

            dsigma_dE_col = ufl.dot(self.Ct_q, e_col_const)

            for col_idx, e_row in enumerate(self._voigt_basis):
                e_row_const = fem.Constant(self.mesh, np.asarray(e_row, dtype=default_scalar_type))
                val_modulus = fem.assemble_scalar(
                    fem.form(ufl.inner(dsigma_dE_col, e_row_const) * self.dx_q)
                ) / self.vol
                C_tangent[row_idx, col_idx] = val_modulus

        ksp.destroy()
        A_petsc.destroy()

        return C_tangent