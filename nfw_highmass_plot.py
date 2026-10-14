# ---- plot the heaviest acceptable NFW halo ----
c_h, mvir_h = best_heavy[2:]
nfw_h = NFWPotential(conc=c_h, mvir=mvir_h, wrtcrit=True, ro=ro, vo=vo)
v_nfw_h = np.array([nfw_h.vcirc(rad * u.kpc) for rad in r])
chisq_h = np.sum(((v - np.sqrt(v_bary**2 + v_nfw_h**2)) / ev)**2)
prob_h = np.round(prob_of(chisq_h, 2), 3)
print('chisq =', chisq_h, ' p =', prob_h)

v_nfw_h_plot = np.array([nfw_h.vcirc(rad * u.kpc) for rad in r_plot])
v_circ_h_plot = np.sqrt(v_bary_plot**2 + v_nfw_h_plot**2)

fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
ax.errorbar(r, v, yerr=ev, fmt='r.', label='MW RC Jiao', capsize=3)
ax.plot(r_plot, v_bary_plot, label='baryon')
ax.plot(r_plot, v_nfw_h_plot, label='NFW DM (heaviest allowed)')
ax.plot(r_plot, v_circ_h_plot, label='Total')
ax.set(xlim=(0, 30), ylim=(0, 250), xlabel='R (kpc)', ylabel='v_circ (km/s)')
ax.set_title('Heaviest NFW, p>=0.01: M = %.2f x 1e11 Msun, chisq/dof = %.2f'
             % (best_heavy[0], chisq_h / (len(r) - 2)), fontsize=8)
ax.grid(); ax.legend(loc='lower right')
fig.savefig('fig2_nfw_highmass.png')
