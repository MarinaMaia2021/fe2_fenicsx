#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 15:18:04 2026

@author: malvesmaia
"""

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

# Small padding used when selecting boundary edges by bounding box (mesh units)
_BBOX_TOL = 0.01


# ==============================================================================
# JAX material kernels (matrix phase: J2 plasticity under plane-stress)
# ==============================================================================

def _plane_stress_strain_6d(eps_3d, nu):
    """
    Embeds a 3-component plane-stress Voigt strain [eps_xx, eps_yy, gamma_xy]
    into the 6-component 3D strain convention expected by `jax_j2`, enforcing
    the plane-stress condition: eps_zz = -nu / (1 - nu) * (eps_xx + eps_yy).
    """
    eps_zz = -(nu / (1.0 - nu)) * (eps_3d[0] + eps_3d[1])
    return jnp.array([eps_3d[0], eps_3d[1], eps_zz, eps_3d[2], 0.0, 0.0], dtype=jnp.float64)


def _voigt_stress_from_6d(stress_6d):
    """Extracts plane-stress Voigt stresses [sig_xx, sig_yy, sig_xy] from a 6D stress vector."""
    return jnp.array([stress_6d[0], stress_6d[1], stress_6d[3]], dtype=jnp.float64)


def _matrix_stress_only(eps_3d, ep_hist, ep_eq_hist, jax_mat):
    """Evaluates the matrix phase's plane-stress response for a single quadrature point."""
    eps_6d = _plane_stress_strain_6d(eps_3d, jax_mat.nu)
    stress_6d, _, _ = jax_j2.update_single_point(eps_6d, ep_hist, ep_eq_hist, jax_mat)
    return _voigt_stress_from_6d(stress_6d)


# Consistent 3x3 plane-stress tangent via autodiff through the 3D-to-6D-to-3D pipeline
_matrix_tangent_single = jax.jacfwd(_matrix_stress_only, argnums=0)

# Batched (vmap+jit) versions used to update/differentiate all matrix quadrature points at once
_matrix_update_batched = jax.jit(
    vmap(
        lambda eps_3d, ep, ep_eq, mat: jax_j2.update_single_point(
            _plane_stress_strain_6d(eps_3d, mat.nu), ep, ep_eq, mat
        ),
        in_axes=(0, 0, 0, None),
    )
)

_matrix_tangent_batched = jax.jit(
    vmap(_matrix_tangent_single, in_axes=(0, 0, 0, None))
)


def _as_petsc_vec(obj):
    """
    Returns the underlying PETSc vector for a dolfinx/dolfinx_mpc object,
    tolerating both the `petsc_vec` and legacy `petsc_vector` attribute names.
    """
    if hasattr(obj, "petsc_vec"):
        return obj.petsc_vec
    if hasattr(obj, "petsc_vector"):
        return obj.petsc_vector
    return obj


class ExternalJAXMatrixPhase:
    """Manages internal history states and macro-point tracking for the matrix phase."""

    def __init__(self, num_quad_points_per_rve, num_macro_points_total, material_properties=None):
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
                 strain_factor=2.0):

        self.microSolverType = direct_solver
        self.verbose = verbose
        self.factor = 1.0

        material_properties = self._setup_material_properties(
            E_m=3.13e3,
            nu_m=0.37,
            sig0_m=31.2,
            sigu_m=64.8,
            b_m=1 / 0.003407,
            E_f=74000.0,
            nu_f=0.2,
        )

        self._setup_geometry()
        gdim, fdim = self._build_periodic_unit_cell_mesh()

        self.vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=self.mesh)))
        self.matrix_cells = self.cells.find(1)
        self.fiber_cells = self.cells.find(2)

        self._setup_fiber_elastic_constants()
        self._setup_quadrature_fields(quadrature_degree)
        self._setup_jax_matrix_operator(num_macro_cells, material_properties)
        self._setup_macro_strain_fields()
        self._setup_displacement_space_and_constraints(gdim)
        self._setup_weak_forms()

        # Cartesian Voigt basis {e_xx, e_yy, e_xy}, used to probe homogenized
        # stress/tangent one strain component at a time.
        self._voigt_basis = [
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
        ]

    # --------------------------------------------------------------------
    # Construction helpers
    # --------------------------------------------------------------------

    def _setup_material_properties(self, E_m=3.13e3, nu_m=0.37,
                                    sig0_m=31.2, sigu_m=64.8, b_m=1 / 0.003407,
                                    E_f=74000.0, nu_f=0.2):
        """Stores matrix/fiber material constants on self and returns them as a list."""
        self.E_m, self.nu_m, self.sig0_m, self.sigu_m, self.b_m = E_m, nu_m, sig0_m, sigu_m, b_m
        self.E_f, self.nu_f = E_f, nu_f
        return [self.E_m, self.nu_m, self.sig0_m, self.sigu_m, self.b_m, self.E_f, self.nu_f]

    def _setup_geometry(self):
        """Defines the unit-cell layout: a square domain with fibers at the corners and center."""
        self.Lx, self.Ly = 1.0, 1.0
        self.R = 0.15 * self.Lx
        self.h = 0.1 * self.Lx

        self.corners = np.array([[0.0, 0.0], [self.Lx, 0.0], [self.Lx, self.Ly], [0.0, self.Ly]])
        self.fibers_center = np.vstack([self.corners, np.array([0.5, 0.5])])

        # Lattice vectors defining the periodic cell (used by the periodicity relations below)
        self.a1 = self.corners[1, :] - self.corners[0, :]
        self.a2 = self.corners[3, :] - self.corners[0, :]

    def _build_periodic_unit_cell_mesh(self):
        """
        Builds the periodic unit-cell mesh (matrix square with corner + center fiber
        inclusions) using Gmsh, tags matrix/fiber volumes and boundary edges, and
        registers the left-right / bottom-top periodicity used later by the MPC.

        Sets self.mesh, self.cells, self.facets. Returns (gdim, fdim).
        """
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

        bottom_tags, right_tags, top_tags, left_tags = self._tag_boundary_edges(fdim)

        translation_right = [1, 0, 0, self.Lx, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        for l_tag, r_tag in zip(left_tags, right_tags):
            gmsh.model.mesh.setPeriodic(fdim, [r_tag], [l_tag], translation_right)

        translation_top = [1, 0, 0, 0, 0, 1, 0, self.Ly, 0, 0, 1, 0, 0, 0, 0, 1]
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
        
        self.dim = self.mesh.topology.dim
        
        # Print mesh (for debugging)
        gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
        gmsh.write("micro_mesh.msh")

        gmsh.finalize()

        return gdim, fdim

    def _tag_boundary_edges(self, fdim):
        """Identifies the four outer edges of the unit square by bounding box, for periodicity/BCs."""
        tol = _BBOX_TOL
        bottom_edges = gmsh.model.getEntitiesInBoundingBox(-tol, -tol, -tol, self.Lx + tol, tol, tol, fdim)
        right_edges = gmsh.model.getEntitiesInBoundingBox(self.Lx - tol, -tol, -tol, self.Lx + tol, self.Ly + tol, tol, fdim)
        top_edges = gmsh.model.getEntitiesInBoundingBox(-tol, self.Ly - tol, -tol, self.Lx + tol, self.Ly + tol, tol, fdim)
        left_edges = gmsh.model.getEntitiesInBoundingBox(-tol, -tol, -tol, tol, self.Ly + tol, tol, fdim)

        bottom_tags = [tag for _, tag in bottom_edges]
        right_tags = [tag for _, tag in right_edges]
        top_tags = [tag for _, tag in top_edges]
        left_tags = [tag for _, tag in left_edges]
        return bottom_tags, right_tags, top_tags, left_tags

    def _setup_fiber_elastic_constants(self):
        """Builds the fiber's linear-isotropic plane-stress stiffness matrix (self.C_fiber)."""
        E_f, nu_f = self.E_f, self.nu_f
        c_11_f = E_f / (1.0 - nu_f ** 2)
        c_12_f = nu_f * E_f / (1.0 - nu_f ** 2)
        c_33_f = E_f / (2.0 * (1.0 + nu_f))

        self.C_fiber = np.array([
            [c_11_f, c_12_f, 0.0],
            [c_12_f, c_11_f, 0.0],
            [0.0, 0.0, c_33_f],
        ], dtype=np.float64)

    def _setup_quadrature_fields(self, quadrature_degree):
        """
        Allocates the quadrature-point function spaces and state fields (total strain,
        stress, tangent, and plastic-history variables), and records which quadrature
        points belong to the matrix phase (as opposed to the fibers).
        """
        cell_name = self.mesh.topology.cell_name()
        q_degree = quadrature_degree

        Qe_vec3 = basix.ufl.quadrature_element(cell_name, value_shape=(3,), degree=q_degree)
        Qe_mat3 = basix.ufl.quadrature_element(cell_name, value_shape=(3, 3), degree=q_degree)
        Qe_vec6 = basix.ufl.quadrature_element(cell_name, value_shape=(6,), degree=q_degree)
        Qe_sc = basix.ufl.quadrature_element(cell_name, value_shape=(1,), degree=q_degree)

        self.Qv = fem.functionspace(self.mesh, Qe_vec3)
        self.QT = fem.functionspace(self.mesh, Qe_mat3)
        self.Qh = fem.functionspace(self.mesh, Qe_vec6)  # Plastic strain state tensor (6D in JAX)
        self.Qs = fem.functionspace(self.mesh, Qe_sc)    # Equivalent plastic strain (1D)

        self.dx_q = ufl.Measure(
            "dx", domain=self.mesh,
            metadata={"quadrature_degree": q_degree, "quadrature_scheme": "default"},
        )

        self.eps_q = fem.Function(self.Qv, name="eps_tot")
        self.sigma_q = fem.Function(self.Qv, name="sigma")
        self.Ct_q = fem.Function(self.QT, name="Ct")

        self.ep_old = fem.Function(self.Qh, name="ep_old")
        self.ep_eq_old = fem.Function(self.Qs, name="ep_eq_old")
        self.ep_curr = fem.Function(self.Qh, name="ep_curr")
        self.ep_eq_curr = fem.Function(self.Qs, name="ep_eq_curr")

        self.n_qp = self.ep_eq_old.x.array.shape[0]

        # Extract matrix QP indices for selective JAX updating (fibers stay linear-elastic)
        matrix_qp_indices = []
        dofmap = self.Qv.dofmap
        for cell_idx in self.matrix_cells:
            matrix_qp_indices.extend(dofmap.cell_dofs(cell_idx))
        self.matrix_qp_indices = np.array(matrix_qp_indices, dtype=np.int32)

    def _setup_jax_matrix_operator(self, num_macro_cells, material_properties):
        """Creates the JAX-based matrix-phase operator that owns the plastic history registry."""
        num_matrix_points = len(self.matrix_qp_indices)
        self.jax_operator = ExternalJAXMatrixPhase(num_matrix_points, num_macro_cells, material_properties)

    def _setup_macro_strain_fields(self):
        """Cell-wise (DG0) fields used to impose the macroscopic strain onto this RVE."""
        self.V0 = fem.functionspace(self.mesh, ("DG", 0))
        self.E_xx_macro = fem.Function(self.V0)
        self.E_yy_macro = fem.Function(self.V0)
        self.E_xy_macro = fem.Function(self.V0)

    def _setup_displacement_space_and_constraints(self, gdim):
        """
        Sets up the displacement fluctuation space, pins rigid-body motion with two
        Dirichlet conditions, and builds the periodic multi-point constraint (MPC)
        that ties the left/right and bottom/top boundaries together.
        """
        self.V = fem.functionspace(self.mesh, ("P", 1, (gdim,)))
        self.u_ = ufl.TestFunction(self.V)
        self.du = ufl.TrialFunction(self.V)
        self.v = fem.Function(self.V, name="Periodic_fluctuation")

        # Pin the origin fully, and pin the x-displacement at (0, Ly) to remove rotation
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
        
    #    self._setup_boundary_conditions()

      # FROM JOEP
    # def _setup_boundary_conditions(self):
    #         """Setup Dirichlet BC at all corners, and periodic constraints on edges excluding corners"""
    #         L = self.Lx
    
    #         def bot_left(x):
    #             return np.isclose(x[0], 0) & np.isclose(x[1], 0)
    
    #         def bot_right(x):
    #             return np.isclose(x[0], L) & np.isclose(x[1], 0.0)
    
    #         def top_right(x):
    #             return np.isclose(x[0], L) & np.isclose(x[1], L)
    
    #         def top_left(x):
    #             return np.isclose(x[0], 0.0) & np.isclose(x[1], L)
    
    #         # Fix all corners
    #         dofs_bl = fem.locate_dofs_geometrical(self.V, bot_left)
    #         dofs_br = fem.locate_dofs_geometrical(self.V, bot_right)
    #         dofs_tr = fem.locate_dofs_geometrical(self.V, top_right)
    #         dofs_tl = fem.locate_dofs_geometrical(self.V, top_left)
    #         self.bcs = [
    #             fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_bl, self.V),
    #             fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_br, self.V),
    #             fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_tr, self.V),
    #             fem.dirichletbc(np.zeros(self.dim, dtype=default_scalar_type), dofs_tl, self.V),
    #         ]
    
    #         # Periodic relation
    #         def periodic_relation(x):
    #             """Map right/top boundaries to left/bottom boundaries."""
    #             out_x = x[0].copy()
    #             out_y = x[1].copy()
    #             out_z = x[2].copy()
    #             out_x[np.isclose(x[0], L)] = 0.0
    #             out_y[np.isclose(x[1], L)] = 0.0
    #             return np.array([out_x, out_y, out_z])
    
    #         def boundary_locator(x):
    #             """Identify right and top boundaries, excluding corners."""
    #             on_right = np.isclose(x[0], L)
    #             on_top = np.isclose(x[1], L)
    #             on_left = np.isclose(x[0], 0.0)
    #             on_bottom = np.isclose(x[1], 0.0)
    
    #             # Exclude corners: (L,0), (0,L), (L,L)
    #             right_edge = on_right & ~on_top & ~on_bottom
    #             top_edge = on_top & ~on_left & ~on_right
    
    #             return right_edge | top_edge
        
    #         # Setup multi-point constraints for periodicity
    #         self.mpc = dolfinx_mpc.MultiPointConstraint(self.V)
    #         self.mpc.create_periodic_constraint_geometrical(
    #             self.V, boundary_locator, periodic_relation, self.bcs
    #         )
    #         self.mpc.finalize()

    def _setup_weak_forms(self):
        """Builds the Voigt strain measures and the residual/tangent weak forms."""

        def voigt(g):
            s = ufl.sym(g)
            return ufl.as_vector([s[0, 0], s[1, 1], self.factor * s[0, 1]])

        Eps_macro = ufl.as_tensor([[self.E_xx_macro, self.E_xy_macro], [self.E_xy_macro, self.E_yy_macro]])
        self.eps_tot_form = ufl.as_vector(
            [Eps_macro[0, 0], Eps_macro[1, 1], self.factor * Eps_macro[0, 1]]
        ) + voigt(ufl.grad(self.v))
        self.eps_tot_expr = fem.Expression(self.eps_tot_form, self.Qv.element.interpolation_points)

        eps_test = voigt(ufl.grad(self.u_))
        eps_trial = voigt(ufl.grad(self.du))

        self.F_form = ufl.inner(self.sigma_q, eps_test) * self.dx_q
        self.J_form = ufl.inner(ufl.dot(self.Ct_q, eps_trial), eps_test) * self.dx_q

    # --------------------------------------------------------------------
    # Constitutive update
    # --------------------------------------------------------------------

    def _update_constitutive_fields(self):
        """Updates stress/tangent at every quadrature point (fibers: elastic, matrix: JAX J2)."""
        self.v.x.scatter_forward()
        self.eps_q.interpolate(self.eps_tot_expr)

        eps_all = self.eps_q.x.array.reshape(self.n_qp, 3)
        sigma_all = np.zeros((self.n_qp, 3), dtype=np.float64)
        Ct_all = np.zeros((self.n_qp, 3, 3), dtype=np.float64)

        ep_curr_all = self.ep_old.x.array.reshape(self.n_qp, 6).copy()
        ep_eq_curr_all = self.ep_eq_old.x.array.reshape(self.n_qp).copy()

        # 1. Inclusions: pure linear elastic
        sigma_all[:] = eps_all @ self.C_fiber.T
        Ct_all[:] = self.C_fiber

        # 2. Matrix: JAX J2 return mapping (overwrites the elastic guess at matrix points)
        if len(self.matrix_qp_indices) > 0:
            m_idx = self.matrix_qp_indices
            eps_m_3d = eps_all[m_idx]
            ep_m = self.ep_old.x.array.reshape(self.n_qp, 6)[m_idx]
            ep_eq_m = self.ep_eq_old.x.array.reshape(self.n_qp)[m_idx]

            # Batch JAX return mapping and its autodiff-consistent tangent
            stress_m_6d, new_ep_m, new_ep_eq_m = _matrix_update_batched(
                eps_m_3d, ep_m, ep_eq_m, self.jax_operator.jax_material
            )
            Ct_m_3x3 = _matrix_tangent_batched(
                eps_m_3d, ep_m, ep_eq_m, self.jax_operator.jax_material
            )

            # Reduce JAX's 6D stress to plane-stress Voigt components [sig_xx, sig_yy, sig_xy]
            stress_m_6d_np = np.array(stress_m_6d, dtype=np.float64)
            sigma_all[m_idx] = stress_m_6d_np[:, [0, 1, 3]]
            Ct_all[m_idx] = np.array(Ct_m_3x3, dtype=np.float64)

            ep_curr_all[m_idx] = np.array(new_ep_m, dtype=np.float64)
            ep_eq_curr_all[m_idx] = np.array(new_ep_eq_m, dtype=np.float64)

        self.sigma_q.x.array[:] = sigma_all.reshape(-1)
        self.Ct_q.x.array[:] = Ct_all.reshape(-1)

        self._trial_new_ep = ep_curr_all.reshape(-1)
        self._trial_new_ep_eq = ep_eq_curr_all.reshape(-1)

    def _solve_micro_newton(self):
        """Solves the micro-fluctuation equilibrium problem for the current macro strain."""
        F_compiled = fem.form(self.F_form)
        J_compiled = fem.form(self.J_form)

        du = fem.Function(self.V)
        du_petsc = _as_petsc_vec(du.x)
        
        # Update sigma_q and Ct_q based on the current fluctuation 'v'
        self._update_constitutive_fields()

        # Assemble residual vector and jacobian matrix with MPC constraints
        b_mpc = dolfinx_mpc.assemble_vector(F_compiled, constraint=self.mpc)
        b_mpc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        
        # Apply lifting for Dirichlet BCs (scale=0.0 because it's a residual update)
        dolfinx_mpc.apply_lifting(b_mpc, [J_compiled], [self.bcs], constraint=self.mpc, scale=0.0)
        
        # Apply BCs            
        fem.petsc.set_bc(b_mpc, self.bcs, alpha=0.0)
        b_petsc = _as_petsc_vec(b_mpc)

        # Assemble micromodel tangent matrix
        A_petsc = dolfinx_mpc.assemble_matrix(J_compiled, constraint=self.mpc, bcs=self.bcs)
        A_petsc.assemble()

        # Set up PETSc linear solver and solve for increment 'du'
        solver = _setup_solver(self.mesh, directSolver=self.microSolverType, tag='micro_')
        solver.setOperators(A_petsc)

        du.x.array[:] = 0.0
        solver.solve(b_petsc, du_petsc)

        # Clean up PETSc structures immediately to save memory
        solver.destroy()
        A_petsc.destroy()

        # Homogenize
        self.mpc.homogenize(du)
        self.mpc.backsubstitution(du)
        du.x.scatter_forward()

        self.v.x.petsc_vec.axpy(-1.0, du.x.petsc_vec)
        self.v.x.scatter_forward()

        self.ep_curr.x.array[:] = self._trial_new_ep
        self.ep_eq_curr.x.array[:] = self._trial_new_ep_eq
        return True

    # --------------------------------------------------------------------
    # History advance + macro-scale coupling
    # --------------------------------------------------------------------

    def advance(self):
        """Commits micro history states upon macroscopic step convergence."""
        self.ep_old.x.array[:] = self.ep_curr.x.array
        self.ep_eq_old.x.array[:] = self.ep_eq_curr.x.array
        self.jax_operator.advance_macro_point()
        if self.verbose:
            print("Updated history of two-phase micromodel.")

    def evaluate_homogenized_properties(self, E_macro_vector, ep_old_slice, ep_eq_old_slice, v_init=None, macro_cell_id=0):
        """Pushes macro strains, computes the micro solution, and extracts the homogenized tangent."""
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

        self.v.x.array[:] = v_init if v_init is not None else 0.0

        micro_converged = self._solve_micro_newton()
        if not micro_converged:
            return False, np.zeros(3), np.zeros((3, 3)), self.v.x.array.copy(), self.ep_curr.x.array.copy(), self.ep_eq_curr.x.array.copy()

        Sigma_out = self.compute_stress()
        C_tangent = self.compute_tangent()
        
        return True, Sigma_out, C_tangent, self.v.x.array.copy(), self.ep_curr.x.array.copy(), self.ep_eq_curr.x.array.copy()

    def compute_stress(self):
        """Computes the homogenized macro stress vector [sig_xx, sig_yy, sig_xy]."""
        if self.verbose:
            print("Computing homogenized stress.")

        Sigma_out = np.zeros(3)
        for idx, e_i in enumerate(self._voigt_basis):
            e_const = fem.Constant(self.mesh, np.asarray(e_i, dtype=default_scalar_type))
            Sigma_out[idx] = fem.assemble_scalar(
                fem.form(ufl.inner(self.sigma_q, e_const) * self.dx_q)
            ) / self.vol
        return Sigma_out

    def compute_tangent(self):
        """
        Assembles the homogenized 3x3 macroscopic consistent tangent stiffness matrix by
        solving one perturbed equilibrium problem per Voigt strain direction.
        """
        if self.verbose:
            print("Computing homogenized tangent stiffness matrix.", flush=True)

        C_tangent = np.zeros((3, 3))
        J_compiled = fem.form(self.J_form)
        A_petsc = dolfinx_mpc.assemble_matrix(J_compiled, constraint=self.mpc, bcs=self.bcs)
        A_petsc.assemble()

        ksp = _setup_solver(self.mesh, directSolver=self.microSolverType, tag='micro_')
        ksp.setOperators(A_petsc)

        dv_sol = fem.Function(self.V)
        dv_petsc = _as_petsc_vec(dv_sol.x)

        # Test-function strain (Voigt), independent of the perturbation direction below
        grad_u_sym = ufl.sym(ufl.grad(self.u_))
        eps_test = ufl.as_vector([grad_u_sym[0, 0], grad_u_sym[1, 1], self.factor * grad_u_sym[0, 1]])

        for row_idx, e_col in enumerate(self._voigt_basis):
            dv_sol.x.array[:] = 0.0
            e_col_const = fem.Constant(self.mesh, np.asarray(e_col, dtype=default_scalar_type))

            L_pert = -ufl.inner(ufl.dot(self.Ct_q, e_col_const), eps_test) * self.dx_q
            b_form = fem.form(L_pert)

            b_mpc = dolfinx_mpc.assemble_vector(b_form, constraint=self.mpc)
            b_mpc.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(b_mpc, self.bcs, alpha=0.0)
            b_petsc = _as_petsc_vec(b_mpc)

            ksp.solve(b_petsc, dv_petsc)

            self.mpc.homogenize(dv_sol)
            self.mpc.backsubstitution(dv_sol)
            dv_sol.x.scatter_forward()

            dsigma_dE_col = ufl.dot(self.Ct_q, e_col_const)

            for col_idx, e_row in enumerate(self._voigt_basis):
                e_row_const = fem.Constant(self.mesh, np.asarray(e_row, dtype=default_scalar_type))
                C_tangent[row_idx, col_idx] = fem.assemble_scalar(
                    fem.form(ufl.inner(dsigma_dE_col, e_row_const) * self.dx_q)
                ) / self.vol

        ksp.destroy()
        A_petsc.destroy()
        return C_tangent