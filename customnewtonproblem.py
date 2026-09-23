#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  8 17:53:41 2026

@author: malvesmaia
"""
from auxFunctions import _setup_solver

class CustomNewtonProblem:
    def __init__(self, quadrature_map, 
                 F, J, u, bcs,
                 verbose = False,
                 directSolver = False,
                 max_it=30, rtol=1e-5, atol=1e-6):
        from dolfinx.fem import form, Function
        from dolfinx.fem.petsc import create_matrix, create_vector
        
        # Integration quadrature
        self.qmap = quadrature_map
        self.verbose = verbose
        self.macroSolverType = directSolver
        
        # Safe form compilation checking
        if isinstance(F, list):
            self.L = [form(f) if not hasattr(f, "function_space") else f for f in F]
        else:
            self.L = form(F) if not hasattr(F, "function_space") else F
            
        if isinstance(J, list):
            self.a = [form(j) if not hasattr(j, "function_space") else j for j in J]
        else:
            self.a = form(J) if not hasattr(J, "function_space") else J
            
        self.bcs = bcs
        self._F, self._J = None, None
        self.u = u
        self.du = Function(self.u.function_space, name="Increment")
        self.it = 0

        # Matrix preallocation via modern form objects
        if isinstance(self.a, list):
            self.A = create_matrix(self.a[0])
            self.b = create_vector(self.L[0].function_spaces)
        else:
            self.A = create_matrix(self.a)  
            self.b = create_vector(self.L.function_spaces)
        
        self.max_it = max_it
        self.rtol = rtol
        self.atol = atol
        self.internal_forces = None      # for retrieving internal forces easily
        self.comm = self.u.function_space.mesh.comm
        
        if self.verbose: print('Finished initiating Newton Raphson.')
                
    # Main solve function
    def solve(self):
        from dolfinx.common import Timer
        import petsc4py.PETSc as PETSc
        from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc
                  
        # Handling different version names                   
        if hasattr(self.u.x, "petsc_vec"):
            u_petsc = self.u.x.petsc_vec
        else:
            u_petsc = self.u.x.petsc_vector

        if hasattr(self.du.x, "petsc_vec"):
            du_petsc = self.du.x.petsc_vec
        else:
            du_petsc = self.du.x.petsc_vector
        
        it = 0  # number of iterations of the Newton solver
        allMicroConverged = False
        macro_converged = False
        
        while it < self.max_it:            
            with Timer("Constitutive update"):
                if self.verbose: print('Update integration points.')
                allMicroConverged = self.qmap.update(self.u)
                if self.verbose: print('End of update integration points')
            
            # Stop timestep if a micromodel does not not converge
            if not allMicroConverged:
                print("  [FAILURE] One or more micromodels did not converge.")
                return False, self.max_it    # TODO: Adjust step size of macroscale 
                                             # or implement substepping for specific
                                             # micromodel
                      
            # Assemble Jacobian and residual
            with self.b.localForm() as loc_b:
                loc_b.set(0)
                
            if isinstance(self.L, list):
                 for Li in self.L:
                     bi = assemble_vector(Li)
                     self.b.axpy(1.0, bi)
            else:
                 assemble_vector(self.b, self.L)
                 
            if self.verbose:
                raw_b = self.b.getArray(readonly=True).copy()
                print(f"\n[Macro] [Iter {it}] Residual vector (pre-lifting):", flush=True)
                print(raw_b, flush = True)
            
            # Compute b - alpha * J(u_D-u_(i-1))
            if isinstance(self.a, list):
                for ai in self.a:
                    apply_lifting(self.b, [ai], [self.bcs], x0=[u_petsc], alpha=-1.0)
            else:
                apply_lifting(self.b, [self.a], [self.bcs], x0=[u_petsc], alpha=-1.0)
                              
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE
            )       
            
            if self.verbose:
                raw_b_lifted = self.b.getArray(readonly=True).copy()
                print(f"\n[Macro] [Iter {it}] Raw residual (lifted):", flush=True)
                print(raw_b_lifted, flush = True)
            
            # Set dx|_bc = u_{i-1}-u_D
            set_bc(self.b, self.bcs, u_petsc, -1.0)        
            self.b.ghostUpdate(
                addv=PETSc.InsertMode.INSERT_VALUES, mode=PETSc.ScatterMode.FORWARD
            )
            
            if self.verbose:
                raw_b_lifted_bc = self.b.getArray(readonly=True).copy()
                print(f"\n[Macro] [Iter {it}] Raw residual (lifted+bc):", flush=True)
                print(raw_b_lifted_bc, flush = True)

            # Calculate norm of residual AFTER boundary conditions are applied
            norm_res = self.b.norm(PETSc.NormType.NORM_2)
            if it == 0:
                norm_res0 = norm_res
                prev_res = 0
                rel_norm_res = 1.0
            else:
                rel_norm_res = norm_res / norm_res0
    
            if not self.verbose:
                print(f"[Macro] [Iter {it:2d}] | Residual norm: {norm_res:.4e}"
                      f" Relative residual norm: {rel_norm_res:.4e}")
    
            if it > 0 and norm_res > 3*prev_res:
                # Divergence. Try with smaller step.
                print(f"[Macro] [DIVERGENCE] Residual norm: {norm_res:.4e}"
                      f" Previous residual norm: {prev_res:.4e}")
                return False, it

            if rel_norm_res < self.rtol or norm_res < self.atol:
                macro_converged = True
                break

            # Assemble tangent matrix with BCs
            self.A.zeroEntries()
            if isinstance(self.a, list):
                self.A.assemble()
                for ai in self.a:
                    Ai = assemble_matrix(ai, bcs=self.bcs)
                    Ai.assemble()
                    self.A.axpy(1.0, Ai)
            else:
                assemble_matrix(self.A, self.a, bcs=self.bcs)
                self.A.assemble()
                test = assemble_matrix(self.a)
                test.assemble()
            
            from scipy.sparse import csr_matrix
            indptr, indices, data = self.A.getValuesCSR()
            A_dense = csr_matrix((data, indices, indptr), shape=self.A.getSize()).toarray()
            if self.verbose:
                print(f"\n[Macro] [Iter {it}] Macro Stiffness Matrix self.A (shape {A_dense.shape}):")
                print(A_dense)
                print("-" * 50)    
                
            indptr, indices, data = test.getValuesCSR()
            A_test = csr_matrix((data, indices, indptr), shape=test.getSize()).toarray()
            if self.verbose:
                print(f"\n[Macro] [Iter {it}] Unconstrained Macro Stiffness Matrix self.A (shape {A_dense.shape}):")
                print(A_test)
                print("-" * 50)    

            # Compute negative residual vector
            self.b.scale(-1)

            # Solve linear problem K * du = -R
            solver = _setup_solver(self.qmap.macro_domain, directSolver=self.macroSolverType, tag = 'macro_', )
            solver.setOperators(self.A)
            
            with Timer("Linear solve"):
                solver.solve(self.b, du_petsc)  
                self.du.x.scatter_forward()
            
            solver.destroy()

            # Update displacement candidate: u_{k+1} = u_k + du
            self.u.x.array[:] += self.du.x.array[:]
            self.u.x.scatter_forward()
            
            if self.verbose:
                print('[Macro] Increment of displacement ', self.du.x.array, flush = True)
                print('[Macro] Total displacement ', self.u.x.array, flush = True)
            
            it += 1
            prev_res = norm_res
                    
        return macro_converged, it
