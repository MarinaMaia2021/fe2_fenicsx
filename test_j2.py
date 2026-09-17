import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from j2_jax import create_material, init_history, update_single_point, constitutive_update_batch

# 1. Setup Material and History
material = create_material()
# We will simulate a single integration point (batch_size = 1)
batch_size = 1
hist_state = init_history(batch_size)

# 2. Define the Loading Scenario (Uniaxial Strain Stretch)
# We increment eps_xx while keeping eps_yy and eps_xy at 0
import numpy as np
elastic_steps  = np.linspace(0.0,  0.015, 10)   # coarse, elastic
plastic_steps  = np.linspace(0.015, 0.05, 20)   # fine, plastic onset
unloading     = np.linspace(0.05, 0.04,  10)    # medium
reloading =  np.linspace(0.04, 0.06,  10)
eps_xx_path = np.concatenate([elastic_steps, plastic_steps, unloading, reloading])

# Prepare arrays to store results for plotting
stored_strains = []
stored_stresses = []
stored_tangents = []

# 3. Define a helper function to get only the stress for Jacobian computation
# jax.jacobian requires a function that outputs just the tensor we want to differentiate
def compute_stress_only(eps_new, eps_p_hist, eps_p_eq_hist, mat):
    stress, _, _ = update_single_point(eps_new, eps_p_hist, eps_p_eq_hist, mat)
    return stress

# Automatically compute the 3x3 tangent matrix (d_stress / d_eps_new)
tangent_matrix_fn = jax.jacobian(compute_stress_only, argnums=0)

print("Running loading simulation...")

# 4. Time-stepping loop (History must be updated sequentially)
for step, target_disp in enumerate(eps_xx_path):
    print(f'Step {step}')
    # Construct the 3D strain vector for this step [eps_xx, eps_yy, gamma_xy]
    eps_step = jnp.array([eps_xx_path[step], 0.0, 0.0])
    
    # Reshape for the batch function (batch_size, 3)
    eps_batch = eps_step.reshape(1, 3)
    
    # Extract the scalar/vector tracking for the single point to compute the tangent
    eps_p_single = hist_state.eps_plastic[0]
    eps_p_eq_single = hist_state.eps_p_eq[0]
    
    # Compute Tangent Stiffness Matrix (3x3) for this point
    C_tangent = tangent_matrix_fn(eps_step, eps_p_single, eps_p_eq_single, material)
    
    # Compute the material update and get the next history state
    stress_batch, hist_state = constitutive_update_batch(eps_batch, hist_state, material)
    
    # Store results (extracting out of the batch dimension)
    stored_strains.append(eps_step[0])
    stored_stresses.append(stress_batch[0, 0])  # sig_xx
    stored_tangents.append(C_tangent[1,0])      # d(sig_xx) / d(eps_xx)

# Convert lists to arrays for plotting
stored_strains = jnp.array(stored_strains)
stored_stresses = jnp.array(stored_stresses)
stored_tangents = jnp.array(stored_tangents)

# 5. Plotting the results
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot Stress vs Strain
color = 'tab:blue'
ax1.set_xlabel('Applied Strain $\epsilon_{xx}$')
ax1.set_ylabel('Stress $\sigma_{xx}$ (MPa)', color=color)
ax1.plot(stored_strains, stored_stresses, color=color, linewidth=2.5, label='Stress $\sigma_{xx}$')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, linestyle='--', alpha=0.6)

# Instantiate a second axes that shares the same x-axis for the Tangent Modulus
ax2 = ax1.twinx()  
color = 'tab:red'
ax2.set_ylabel('Tangent Modulus $C_{11}$', color=color)
ax2.plot(stored_strains, stored_tangents, color=color, linewidth=2, linestyle='--', label='Tangent $C_{11}$')
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.title('J2 Plasticity: Uniaxial Strain Loading Profile')
plt.show()