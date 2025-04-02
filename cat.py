import numpy as np
import math  
import matplotlib.pyplot as plt
from scipy.special import eval_genlaguerre  

def coherent_state(alpha, N):
    """Generate a coherent state |α⟩ in a truncated Fock basis."""
    n = np.arange(N)
    # Use vectorized operations and math.factorial
    coeffs = np.exp(-0.5 * np.abs(alpha)**2) * (alpha**n) / np.sqrt(np.array([math.factorial(k) for k in n]))
    return coeffs

def cat_state(alpha, N):
    """Generate a cat state (|α⟩ + |-α⟩)/√(2 + 2e^{-2|α|^2})."""
    return (coherent_state(alpha, N) + coherent_state(-alpha, N)) / np.sqrt(2 + 2*np.exp(-2*np.abs(alpha)**2))

def wigner_function(psi, x, p):
    """
    Compute Wigner function for state vector psi at phase space points (x,p).
    Uses direct summation over Fock states.
    """
    N = len(psi)
    W = np.zeros((len(x), len(p)), dtype=complex)
    
    for i, xi in enumerate(x):
        for j, pj in enumerate(p):
            z = (xi + 1j*pj)/np.sqrt(2)  # Complex phase space coordinate
            total = 0.0
            for n in range(N):
                for m in range(N):
                    # Matrix elements of displacement operator
                    if m >= n:
                        term = psi[n] * np.conj(psi[m]) * (-1)**n * \
                               np.exp(-0.5*np.abs(z)**2) * \
                               (np.conj(z))**(m-n) * \
                               eval_genlaguerre(n, m-n, np.abs(z)**2)
                    else:
                        term = np.conj(psi[m]) * psi[n] * (-1)**m * \
                               np.exp(-0.5*np.abs(z)**2) * \
                               z**(n-m) * \
                               eval_genlaguerre(m, n-m, np.abs(z)**2)
                    total += term
            W[i,j] = total * 2/np.pi  # Wigner normalization
    
    return np.real(W)

# ==================================
# Example Usage
# ==================================
N = 20       # Fock space truncation
alpha = 2.0  # Cat state parameter

# Create state
psi = cat_state(alpha, N)

# Phase space grid
x = np.linspace(-5, 5, 100)
p = np.linspace(-5, 5, 100)

# Calculate Wigner function (may take ~1 minute for 100x100 grid)
W = wigner_function(psi, x, p)

# Plotting
plt.figure(figsize=(8, 6))
plt.contourf(x, p, W.T, levels=100, cmap='RdBu_r')
plt.colorbar(label='Wigner Function Value')
plt.xlabel('Position (x)')
plt.ylabel('Momentum (p)')
plt.title('Wigner Function of Cat State ')
plt.show()