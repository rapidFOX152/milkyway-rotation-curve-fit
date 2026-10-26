#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Milky Way rotation-curve fit with NFW and Einasto dark-matter halos.

Reproduction of the Jan 2025 astrophysics workshop task (Galaxy Rotation Curve
Fitting). Data: Milky Way rotation curve from Jiao et al. 2023, baryonic
contribution from de Salas et al. 2019.

What the script does, in order:
  1. load the data and plot it                      -> fig0.png
  2. NFW best fit (simulated annealing on chi^2)    -> fig1_nfw_best.png
  3. NFW heaviest halo the data still allows        -> fig2_nfw_highmass.png
  4. Einasto best fit                               -> fig3_einasto_best.png
  5. Einasto heaviest halo the data still allows    -> fig4_einasto_highmass.png

Masses are M200 (baryons + dark matter inside r200) in units of 1e11 Msun.
Run from the folder that holds rc_mw_jiao23.txt and the two .npy files.
"""

import numpy as np
import astropy.units as u
import astropy.io.ascii as ascii
from galpy.potential import AnySphericalPotential, NFWPotential
from galpy.util import conversion
import matplotlib
matplotlib.use('Agg')          # only save figures, never open windows
import matplotlib.pyplot as plt
from scipy.special import gammaincc

SEED = 42                      # fixes the random walks so results are reproducible
np.random.seed(SEED)

# ---------------------------------------------------------------- setup ----
# Galactic constants in galpy's convention: solar radius ro and circular
# speed vo. massol converts galpy's natural mass unit into solar masses.
ro = 8.178 * u.kpc
vo = 247.3 * u.km / u.s
massol = conversion.mass_in_msol(vo, ro)
mass_bary = 0.61               # baryonic mass, x1e11 Msun

# Observed rotation curve: radius R (kpc), speed Vc and error eVc (km/s).
rc_J23 = ascii.read('rc_mw_jiao23.txt')
[r, v, ev] = np.array([rc_J23['R'], rc_J23['Vc'], rc_J23['eVc']])

# Baryon contribution to the circular speed: at the data radii, and on a
# fine grid (r_plot) for smooth curves in the figures.
v_bary = np.load('v_bary_J23.npy', allow_pickle=True)
v_bary_plot = np.load('v_bary_b2_plot.npy', allow_pickle=True)
r_plot = np.linspace(0.1, 30, 300)


# -------------------------------------------------------------- helpers ----
def vcirc_at(pot, radii):
    """Circular speed of a galpy potential at the given radii (kpc)."""
    return np.array([pot.vcirc(x * u.kpc) for x in radii])


def chisq_of(v_dm):
    """Chi^2 of baryons + dark matter (added in quadrature) against the data."""
    return np.sum(((v - np.sqrt(v_bary**2 + v_dm**2)) / ev)**2)


def prob_of(chisq, n_par):
    """Probability of getting a chi^2 at least this large by chance, given the
    degrees of freedom (data points minus fitted parameters). A tiny value
    means the model is statistically rejected."""
    return gammaincc((len(r) - n_par) / 2, chisq / 2)


def plot_rc(fname, dm_label, v_dm_plot, title):
    """Data, baryon, dark-matter and total curves in one figure."""
    v_tot_plot = np.sqrt(v_bary_plot**2 + v_dm_plot**2)
    fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
    ax.errorbar(r, v, yerr=ev, fmt='r.', label='MW RC Jiao', capsize=3)
    ax.plot(r_plot, v_bary_plot, label='baryon')
    ax.plot(r_plot, v_dm_plot, label=dm_label)
    ax.plot(r_plot, v_tot_plot, label='Total')
    ax.set(xlim=(0, 30), ylim=(0, 250),
           xlabel=r'$R_{gc}$ (kpc)', ylabel=r'$v_{circ}$ (km/s)')
    ax.set_title(title, fontsize=8)
    ax.grid()
    ax.legend(loc='lower right', fontsize=7)
    fig.savefig(fname)
    plt.close(fig)


def anneal(calc_vals, new_param, start, i_max, step_size, min_step=0.01):
    """Simulated annealing: find the parameters with the lowest chi^2.

    A random neighbour is always accepted if it fits better, and sometimes
    even if it fits worse (Metropolis rule). Early on the 'temperature' T is
    high, so the walk roams widely and can escape local minima; as T falls
    the walk settles down. The step size shrinks along with T.
    """
    cur_par = list(start)
    cur_mass, cur_chisq = calc_vals(cur_par)
    accepted = [[cur_mass, cur_chisq, *cur_par]]
    for i in range(i_max):
        if i % round(i_max / 10) == 0:
            print(round(100 * i / i_max), '% done')
        T = 1 - i / i_max
        step = min_step + (step_size - min_step) * T
        next_par = new_param(cur_par, step)
        next_mass, next_chisq = calc_vals(next_par)
        with np.errstate(over='ignore'):   # exp(huge) -> inf is fine here
            take = np.exp(-(next_chisq - cur_chisq) / T) > np.random.uniform(0, 1)
        if take:
            cur_par, cur_chisq, cur_mass = next_par, next_chisq, next_mass
            accepted.append([cur_mass, cur_chisq, *cur_par])
    return min(accepted, key=lambda row: row[1])      # row = [mass, chisq, *params]


def heavy_search(calc_vals, new_param, start, i_max, step_size,
                 p_min=0.01, min_step=0.01):
    """Find the heaviest halo the data cannot rule out.

    Starts from the best fit and only accepts a move if (a) the fit is still
    statistically acceptable (p >= p_min) and (b) the halo is heavier than the
    current one. The result sits near the edge of what the data allow.
    """
    cur_par = list(start)
    cur_mass, cur_chisq = calc_vals(cur_par)
    heavy = [[cur_mass, cur_chisq, *cur_par]]
    for i in range(i_max):
        if i % round(i_max / 10) == 0:
            print(round(100 * i / i_max), '% done')
        T = 1 - i / i_max
        step = min_step + (step_size - min_step) * T
        next_par = new_param(cur_par, step)
        next_mass, next_chisq = calc_vals(next_par)
        prob_ok = prob_of(next_chisq, len(next_par)) >= p_min
        if prob_ok and next_mass > cur_mass:
            cur_par, cur_chisq, cur_mass = next_par, next_chisq, next_mass
            heavy.append([cur_mass, cur_chisq, *cur_par])
    return max(heavy, key=lambda row: row[0])


# ------------------------------------------------------------ 1. the data ----
fig, ax = plt.subplots(figsize=(6, 4), dpi=200)
ax.errorbar(r, v, yerr=ev, fmt='r.', label='MW RC Jiao', capsize=3)
ax.plot(r_plot, v_bary_plot, label='baryon')
ax.set(xlim=(0, 30), ylim=(0, 250),
       xlabel=r'$R_{gc}$ (kpc)', ylabel=r'$v_{circ}$ (km/s)',
       title='MW RC from Jiao et al. 2023, Baryon from de Salas et al. 2019')
ax.title.set_fontsize(8)
ax.grid()
ax.legend(loc='lower right')
fig.savefig('fig0.png')
plt.close(fig)


# ------------------------------------------------------------------ NFW ----
def make_nfw(c, mvir):
    # galpy takes the virial mass in units of 1e12 Msun, and wrtcrit=True
    # defines it relative to the critical density.
    return NFWPotential(conc=c, mvir=mvir, wrtcrit=True, ro=ro, vo=vo)


def calc_vals_nfw(par):
    c, mvir = par
    mass = np.round(mass_bary + mvir * 10, 2)   # mvir*10 -> units of 1e11 Msun
    return [mass, chisq_of(vcirc_at(make_nfw(c, mvir), r))]


def new_nfw_param(par, size):
    """Random neighbour: each parameter moves uniformly within +-size (fraction)."""
    return [np.random.uniform(p - p * size, p + p * size) for p in par]


# ---------------------------------------------------- 2. NFW best fit ----
print('\n=== NFW best fit ===')
nfw_row = anneal(calc_vals_nfw, new_nfw_param, start=[17, 1],
                 i_max=1000, step_size=0.15)
nfw_mass, nfw_chisq, *nfw_par = nfw_row
nfw_pot = make_nfw(*nfw_par)
nfw_prob = prob_of(chisq_of(vcirc_at(nfw_pot, r)), 2)
print('best NFW [mass, chisq, c, mvir]:', nfw_row, ' p =', round(nfw_prob, 3))

plot_rc('fig1_nfw_best.png',
        'NFW DM\n' + r'[conc, $m_{vir}$ ($10^{11} M_{\odot}$)]' + '\n'
        + str([round(nfw_par[0], 3), round(10 * nfw_par[1], 2)]),
        vcirc_at(nfw_pot, r_plot),
        r'$M_{200}$ = %.2f$\times10^{11}M_{\odot}$, p = %.3f, chisq/dof = %.2f'
        % (nfw_mass, nfw_prob, nfw_chisq / (len(r) - 2)))


# ------------------------------------------- 3. NFW heaviest allowed halo ----
print('\n=== NFW heaviest halo with p >= 0.01 ===')
nfw_heavy = heavy_search(calc_vals_nfw, new_nfw_param, start=nfw_par,
                         i_max=10000, step_size=0.15, p_min=0.01)
nfw_h_mass, nfw_h_chisq, *nfw_h_par = nfw_heavy
nfw_h_prob = prob_of(nfw_h_chisq, 2)
print('heaviest NFW [mass, chisq, c, mvir]:', nfw_heavy, ' p =', round(nfw_h_prob, 3))

plot_rc('fig2_nfw_highmass.png', 'NFW DM (heaviest allowed)',
        vcirc_at(make_nfw(*nfw_h_par), r_plot),
        'Heaviest NFW, p>=0.01: M = %.2f x 1e11 Msun, chisq/dof = %.2f'
        % (nfw_h_mass, nfw_h_chisq / (len(r) - 2)))


# ------------------------------------------------------------- Einasto ----
def eval_ave_dens(Pot, rad):
    """Mean density inside radius rad (kpc), in Msun/pc^3."""
    return Pot.mass(R=rad * u.kpc, z=None) / ((4 / 3) * np.pi * (rad * 1000)**3)


def R_200(Pot):
    """Radius (kpc) inside which the mean density is 200 x the critical density
    (2.68e-5 Msun/pc^3, i.e. 2.68 in units of 1e-5). Found by scanning radii,
    then refining the scan around the closest match in two more passes."""
    def mismatch(radii):
        dens = np.array([eval_ave_dens(Pot, rr) for rr in radii])
        return abs(np.round(dens * 1e5 - 2.68, 2))

    radii = np.linspace(50, 300, 26)                       # coarse scan
    diff = mismatch(radii)
    rmin = radii[np.argmin(diff)]
    if diff.min() == 0:
        return rmin
    radii = np.round(np.linspace(rmin - 10, rmin + 10, 21), 2)   # finer scan
    diff = mismatch(radii)
    rmin = radii[np.argmin(diff)]
    if diff.min() == 0:
        return rmin
    radii = np.round(np.linspace(rmin - 1, rmin + 1, 21), 2)     # final scan
    return radii[np.argmin(mismatch(radii))]


def make_einasto(mdm, h, n):
    """Einasto halo: density ~ exp(-(r/h)^(1/n)). mdm sets the amplitude
    (x1e11 Msun), h is the scale radius in kpc, n is the shape index."""
    h_n = h / ro.value                    # kpc -> galpy natural units
    rho0 = np.array((mdm * 10**11) / massol / (np.pi * 4. * h_n**3))
    return AnySphericalPotential(dens=lambda x: np.exp(-(x / h_n)**(1 / n)),
                                 amp=rho0, ro=ro, vo=vo)


def calc_vals_ein(par):
    ein = make_einasto(*par)
    mass = mass_bary + np.round(ein.mass(R_200(ein) * u.kpc, z=None) / 10**11, 2)
    return [mass, chisq_of(vcirc_at(ein, r))]


def new_ein_param(par, size):
    """Random neighbour. The amplitude mdm is varied in log10 (it can span
    orders of magnitude); h and n are varied directly."""
    newpar = []
    for i, p in enumerate(par):
        if i == 0:
            lp = np.log10(p)
            newpar.append(10**np.random.uniform(lp - lp * size, lp + lp * size))
        else:
            newpar.append(np.random.uniform(p - p * size, p + p * size))
    return newpar


# -------------------------------------------------- 4. Einasto best fit ----
print('\n=== Einasto best fit ===')
ein_row = anneal(calc_vals_ein, new_ein_param, start=[3, 10, 1],
                 i_max=10000, step_size=0.25)
ein_mass, ein_chisq, *ein_par = ein_row
ein_pot = make_einasto(*ein_par)
ein_prob = prob_of(chisq_of(vcirc_at(ein_pot, r)), 3)
print('best Einasto [mass, chisq, mdm, h, n]:', ein_row, ' p =', round(ein_prob, 3))

plot_rc('fig3_einasto_best.png',
        'Einasto DM\n[mdm h n]\n' + str(np.round(ein_par, 3)),
        vcirc_at(ein_pot, r_plot),
        r'$M_{200}$ = %.2f$\times10^{11}M_{\odot}$, p = %.3f, chisq/dof = %.2f'
        % (ein_mass, ein_prob, ein_chisq / (len(r) - 3)))


# ---------------------------------------- 5. Einasto heaviest allowed halo ----
print('\n=== Einasto heaviest halo with p >= 0.01 ===')
ein_heavy = heavy_search(calc_vals_ein, new_ein_param, start=ein_par,
                         i_max=10000, step_size=0.25, p_min=0.01)
ein_h_mass, ein_h_chisq, *ein_h_par = ein_heavy
ein_h_prob = prob_of(ein_h_chisq, 3)
print('heaviest Einasto [mass, chisq, mdm, h, n]:', ein_heavy,
      ' p =', round(ein_h_prob, 3))

plot_rc('fig4_einasto_highmass.png', 'Einasto DM (heaviest allowed)',
        vcirc_at(make_einasto(*ein_h_par), r_plot),
        'Heaviest Einasto, p>=0.01: M = %.2f x 1e11 Msun, chisq/dof = %.2f'
        % (ein_h_mass, ein_h_chisq / (len(r) - 3)))


# -------------------------------------------------------------- summary ----
print('\n=== Summary (M200 in 1e11 Msun, seed = %d) ===' % SEED)
print('NFW      best fit : M = %.2f  chisq = %.2f  p = %.3f' % (nfw_mass, nfw_chisq, nfw_prob))
print('NFW      heaviest : M = %.2f  chisq = %.2f  p = %.3f' % (nfw_h_mass, nfw_h_chisq, nfw_h_prob))
print('Einasto  best fit : M = %.2f  chisq = %.2f  p = %.3f' % (ein_mass, ein_chisq, ein_prob))
print('Einasto  heaviest : M = %.2f  chisq = %.2f  p = %.3f' % (ein_h_mass, ein_h_chisq, ein_h_prob))
