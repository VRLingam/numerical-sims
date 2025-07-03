import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

c0   = 299_792_458.0       # speed of light (m/s)
eps0 = 8.854e-12           # permittivity (F/m)
mu0  = 4*np.pi*1e-7        # permeability (H/m)

Nx, Ny = 200, 200          # grid size
dx = 1e-3                  # spatial step (m)
dt = dx / (2*c0)           # time step (s), Courant safety
nt = 500                   # number of time steps
x = np.arange(Nx) * dx
y = np.arange(Ny) * dx

eps_r = np.ones((Nx, Ny))
slab_i0, slab_i1 = int(0.4*Nx), int(0.7*Nx)
eps_r[slab_i0:slab_i1, :] = 4.0  # n=2 inside slab

Ez = np.zeros((Nx, Ny))
Hx = np.zeros((Nx, Ny-1))
Hy = np.zeros((Nx-1, Ny))

pulse_i, pulse_j = 50, Ny//2
pulse_width     = 20

fig, ax = plt.subplots(figsize=(6,5))
im = ax.imshow(Ez.T, 
               origin='lower',
               extent=[0, Nx*dx, 0, Ny*dx],
               cmap='RdBu', vmin=-0.02, vmax=0.02)
cbar = fig.colorbar(im, ax=ax)
cbar.set_label('$E_z$ (a.u.)')
ax.set_xlabel('x (m)')
ax.set_ylabel('y (m)')
ax.axvspan(slab_i0*dx, slab_i1*dx, color='gray', alpha=0.3, label='n=2 slab')
ax.legend(loc='upper right')
ax.set_title('t = 0.00 ns')

def update(n):
    global Ez, Hx, Hy
    
    Hx[:, :] -= (dt/(mu0 * dx)) * (Ez[:, 1:] - Ez[:, :-1])
    Hy[:, :] += (dt/(mu0 * dx)) * (Ez[1:, :] - Ez[:-1, :])
    
    curl_H = (Hy[1:, 1:-1] - Hy[:-1, 1:-1]) - (Hx[1:-1, 1:] - Hx[1:-1, :-1])
    Ez[1:-1, 1:-1] += (dt/(eps0 * eps_r[1:-1,1:-1] * dx)) * curl_H
    
    Ez[pulse_i, pulse_j] += np.exp(-0.5*((n - pulse_width)/pulse_width)**2)
    
    Ez[0, :] = Ez[-1, :] = Ez[:, 0] = Ez[:, -1] = 0
    
    im.set_data(Ez.T)
    ax.set_title(f't = {n*dt*1e9:5.2f} ns')
    return im,

anim = FuncAnimation(fig, update, frames=nt, interval=30, blit=True)

writer = FFMpegWriter(fps=20, metadata=dict(artist='You'), bitrate=1800)
anim.save('Ez_heatmap_2d.mp4', writer=writer)

print("Saved 'Ez_heatmap_2d.mp4' — a heatmap animation of Ez(x,y,t).")
