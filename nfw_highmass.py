# ---- high-mass NFW search: heaviest halo with p >= 0.01 ----
p_min = 0.01
i_max = 10000
step_size = 0.15
min_step = 0.01

def prob_of(chisq, n_par):
    return gammaincc((len(r) - n_par) / 2, chisq / 2)

cur_par = list(param_plot_best)
[cur_mass, cur_chisq] = calc_vals_nfw(cur_par)
heavy = [[cur_mass, cur_chisq, *cur_par]]

for i in range(i_max):
    if i % round(i_max / 10) == 0: print(round(100 * i / i_max), '% done')
    T = 1 - i / i_max
    step = min_step + (step_size - min_step) * T
    next_par = new_nfw_param(cur_par, step)
    [next_mass, next_chisq] = calc_vals_nfw(next_par)
    prob_ok = prob_of(next_chisq, len(next_par)) >= p_min
    accept = prob_ok and next_mass > cur_mass
    if accept:
        cur_par, cur_chisq, cur_mass = next_par, next_chisq, next_mass
        heavy.append([cur_mass, cur_chisq, *cur_par])

best_heavy = max(heavy, key=lambda row: row[0])
print('heaviest acceptable NFW [mass, chisq, c, mvir]:', best_heavy)
