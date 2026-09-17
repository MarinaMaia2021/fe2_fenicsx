# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import numpy as np
from mpi4py import MPI
import gmsh
import ufl
from dolfinx import fem, io
import dolfinx.fem.petsc
from dolfinx.io.gmsh import model_to_mesh
import dolfinx_mpc.utils
# CHANGED: Import only NonlinearProblem from dolfinx_mpc
from dolfinx_mpc import NonlinearProblem

Lx = 1.0
Ly = 1.0 
c = 0.5 * Lx
R = 0.1 * Lx
h = 0.08 * Lx

corners = np.array([[0.0, 0.0], [Lx, 0.0], [Lx, Ly], [0, Ly]])
a1 = corners[1, :] - corners[0, :]  
a2 = corners[3, :] - corners[0, :]  
fibers_center = np.vstack([corners, np.array([0.5, 0.5])])

# --- Mesh Generation ---
gdim = 2  
fdim = 1  
gmsh.initialize()

occ = gmsh.model.occ
mesh_comm = MPI.COMM_WORLD
model_rank = 0
if model_rank == 0:
    points = [occ.add_point(*corner, 0) for corner in fibers_center]
    lines = [occ.add_line(points[i], points[(i + 1) % 4]) for i in range(4)]
    loop = occ.add_curve_loop(lines)
    unit_cell = occ.add_plane_surface([loop])
    inclusions = [occ.add_disk(*corner, 0, R, R) for corner in corners]
    vol_dimTag = (gdim, unit_cell)
    out = occ.intersect(
        [vol_dimTag], [(gdim, incl) for incl in inclusions], removeObject=False
    )
    incl_dimTags = out[0]
    occ.synchronize()
    occ.cut([vol_dimTag], incl_dimTags, removeTool=False)
    occ.synchronize()

    gmsh.model.addPhysicalGroup(gdim, [vol_dimTag[1]], 1, name="Matrix")
    gmsh.model.addPhysicalGroup(gdim, [tag for _, tag in incl_dimTags], 2, name="Inclusions")
    gmsh.model.addPhysicalGroup(fdim, [7, 20, 10], 1, name="bottom")
    gmsh.model.addPhysicalGroup(fdim, [9, 19, 16], 2, name="right")
    gmsh.model.addPhysicalGroup(fdim, [15, 18, 12], 3, name="top")
    gmsh.model.addPhysicalGroup(fdim, [11, 17, 5], 4, name="left")
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    gmsh.model.mesh.generate(gdim)

mesh_data = model_to_mesh(gmsh.model, mesh_comm, model_rank, gdim=gdim)
domain = mesh_data.mesh
cells = mesh_data.cell_tags
facets = mesh_data.facet_tags
gmsh.finalize()

vol = fem.assemble_scalar(fem.form(1 * ufl.dx(domain=domain)))
print("Volume:", vol)

# --- Material Fields ---
def create_piecewise_constant_field(domain, cell_markers, property_dict, name=None):
    V0 = fem.functionspace(domain, ("DG", 0))
    k = fem.Function(V0, name=name)
    for tag, value in property_dict.items():
        cells = cell_markers.find(tag)
        k.x.array[cells] = np.full_like(cells, value, dtype=np.float64)
    return k

E = create_piecewise_constant_field(domain, cells, {1: 50e3, 2: 210e3}, name="YoungModulus")
nu = create_piecewise_constant_field(domain, cells, {1: 0.2, 2: 0.3}, name="PoissonRatio")

lmbda = E * nu / (1 + nu) / (1 - 2 * nu)
mu = E / 2 / (1 + nu)

# --- Variational Formulation ---
Eps = fem.Constant(domain, np.zeros((2, 2)))
Eps_ = fem.Constant(domain, np.zeros((2, 2)))
y = ufl.SpatialCoordinate(domain)

def epsilon(v):
    return ufl.sym(ufl.grad(v))

def sigma(v):
    eps = Eps + epsilon(v)
    return lmbda * ufl.tr(eps) * ufl.Identity(gdim) + 2 * mu * eps

V = fem.functionspace(domain, ("P", 2, (gdim,)))
u_ = ufl.TestFunction(V)

# Fluctuations solution unknown container
v = fem.Function(V, name="Periodic_fluctuation") 

# Residual Form: F(v; u_) = 0
F_form = ufl.inner(sigma(v), epsilon(u_)) * ufl.dx

# Jacobian Form: J(v; dv, u_) = dF/dv
dv = ufl.TrialFunction(V)
J_form = ufl.derivative(F_form, v, dv)

# --- Boundary Conditions & MPC ---
point_dof = fem.locate_dofs_geometrical(V, lambda x: np.isclose(x[0], 0.0) & np.isclose(x[1], 0))
bcs = [fem.dirichletbc(np.zeros((gdim,)), point_dof, V)]

def periodic_relation_left_right(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a1[0]
    out_x[1] = x[1] - a1[1]
    out_x[2] = x[2]
    return out_x

def periodic_relation_bottom_top(x):
    out_x = np.zeros(x.shape)
    out_x[0] = x[0] - a2[0]
    out_x[1] = x[1] - a2[1]
    out_x[2] = x[2]
    return out_x

mpc = dolfinx_mpc.MultiPointConstraint(V)
mpc.create_periodic_constraint_topological(V, facets, 2, periodic_relation_left_right, bcs)
mpc.create_periodic_constraint_topological(V, facets, 3, periodic_relation_bottom_top, bcs)
mpc.finalize()

# Reallocate functions to match the constrained mpc space
v = fem.Function(mpc.function_space, name="Periodic_fluctuation")
u = fem.Function(mpc.function_space, name="Displacement")

# CHANGED: Use NonlinearProblem to explicitly handle both forms, constraints, and tolerances via PETSc options
problem = NonlinearProblem(F_form, v, mpc, bcs=bcs, J=J_form, petsc_options={
    "snes_type": "newtonls",
    "snes_atol": 1e-8,
    "snes_rtol": 1e-8,
    "snes_max_it": 50
})

# --- Visualization Prep ---
import pyvista
from dolfinx import plot
pyvista.set_jupyter_backend("static")

W0 = fem.functionspace(domain, ("DG", 0))

def build_deformation_meshes(u, v, scale=1.0):
    topology, cell_types, geometry = plot.vtk_mesh(u.function_space)
    undeformed = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    undeformed.cell_data["material"] = cells.values

    disp = np.zeros((geometry.shape[0], 3))
    disp[:, :2] = u.x.array.reshape(-1, 2)

    stress_expr = fem.Expression(sigma(v)[0, 0], W0.element.interpolation_points)
    sigma_xx = fem.Function(W0)
    sigma_xx.interpolate(stress_expr)
    
    deformed = pyvista.UnstructuredGrid(topology, cell_types, geometry.copy())
    deformed.point_data["u"] = disp
    deformed.set_active_vectors("u")
    deformed.cell_data["stress"] = sigma_xx.x.array
    warped = deformed.warp_by_vector("u", factor=scale)
    return undeformed, warped

macro_strains = [
    np.array([[0.00, 0.00], [0.00, 0.00]]),
    np.array([[0.01, 0.00], [0.00, 0.00]]),
    np.array([[0.02, 0.00], [0.00, 0.00]]),
    np.array([[0.03, 0.00], [0.00, 0.00]]),
    np.array([[0.04, 0.00], [0.00, 0.00]]),
    np.array([[0.05, 0.00], [0.00, 0.00]]),
    np.array([[0.06, 0.00], [0.00, 0.00]])
]

plotter = pyvista.Plotter(off_screen=True)
plotter.open_gif("deformation.gif")

xmin, ymin = 0.0, 0.0
xmax, ymax = np.max(corners[:, 0]), np.max(corners[:, 1])
margin = 0.15 * max(xmax - xmin, ymax - ymin)
fixed_bounds = (xmin - margin, xmax + margin, ymin - margin, ymax + margin, -1, 1)

# --- Incremental Loading Loop ---
for iStep, strain in enumerate(macro_strains):
    print(f"\nStep {iStep}")
    Eps.value = strain

    # CHANGED: Solve via the direct problem object. 
    # It packages the underlying PETSc SNES routine automatically.
    problem.solve()
    
    # Calculate average macroscopic stresses
    Sigma = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            test_strain = np.zeros((2, 2))
            test_strain[i, j] = 1.0
            Eps_.value = test_strain
            Sigma[i, j] = fem.assemble_scalar(
                fem.form(ufl.inner(sigma(v), Eps_) * ufl.dx)
            ) / vol
            
    print("Average stress:")
    print(Sigma)
       
    # Build full displacement field: u = E.y + v
    u.interpolate(
        fem.Expression(
            ufl.dot(Eps, y),
            mpc.function_space.element.interpolation_points
        )
    )
    u.x.array[:] += v.x.array[:]
    
    undeformed, warped = build_deformation_meshes(u, v, scale=5.0)
    
    plotter.clear()
    plotter.add_mesh(
        undeformed, scalars="material", categories=True,
        cmap=["lightgray", "gray"], style="wireframe", line_width=.5,
        opacity=0, edge_opacity=0.5, show_edges=True
    )
    plotter.add_mesh(
        warped, scalars="stress", cmap="coolwarm", clim=[0, 3500],
        show_edges=False, line_width=0.25,
    )
    plotter.view_xy()
    plotter.camera.SetParallelProjection(True)
    plotter.reset_camera(bounds=fixed_bounds)
    plotter.add_text(f"Step {iStep}", font_size=12)
    plotter.show()
    plotter.write_frame()

plotter.close()