"""
Module with the core plotting functions.
"""

import numpy as np
import matplotlib
# Use non-interactive backend for better compatibility with Streamlit and batch processing
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [16, 9]

def configure_plot(ax=None, title=None, xlabel=None, ylabel=None, grid=True, legend=True):
    """Helper function to configure plot appearance."""
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    ax.grid(grid)
    if legend:
        # Only show legend if there are labeled artists to avoid warnings
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="lower right")
    if title: ax.set_title(title)
    return ax

def save_plot(name, method, plot_2D, tau_boundary=None):
    """Helper function to save plots with consistent naming."""
    if method == "Bosch":
        save_name = f"{name}_{method}_tau_{int(tau_boundary * 1e12)}ps"
    else:
        save_name = f"{name}_{method}"
    plt.savefig(f"{save_name}_{plot_2D}.png", dpi=300)

def create_grid(var1, var2, var1_unit, var2_unit):
    """Helper function to create grids and scaled variables."""
    var1_grid, var2_grid = np.meshgrid(var1, var2)
    var1_grid, var2_grid = var1_grid.T * var1_unit, var2_grid.T * var2_unit
    return var1_grid, var2_grid, var1 * var1_unit, var2 * var2_unit

def plot_image(ax, data, extent, clabel, var1_grid, var2_grid, cmap="viridis",
               vmin=None, vmax=None, colorbar=True, contour=False, colorplot=True,
               contour_dict={}):
    """Helper function to handle 2D data visualization."""
    if colorplot:
        c = ax.imshow(data, origin='lower', cmap=cmap, aspect='auto', extent=extent, vmin=vmin, vmax=vmax)
        if colorbar:
            cbar = plt.colorbar(c, ax=ax)
            cbar.set_label(clabel)
    if contour:
        data[data == 0] = np.nan
        contours = ax.contour(var1_grid.T, var2_grid.T, data,
                              contour_dict['levels'], colors=contour_dict.get('colors', 'black'),
                              alpha=contour_dict.get('alpha', 1),
                              linestyles=contour_dict.get('linestyles', '-'))
        ax.clabel(contours, inline=True, fontsize=10, manual=contour_dict.get('manual_clabel', False))

def __plot(out, var1, var2, var1_unit, var2_unit, var1_label, var2_label, plot_2D, save, **kwargs):
    """Primary plotting function for 2D data."""
    (zero_freq_coup, robinson_coup, modes_coup, HOM_coup, converged_coup, PTBL_coup, bl, xi, R) = out
    not_converged = (~converged_coup[:,:,0] | ~converged_coup[:,:,1]) & ~robinson_coup[:,:,:].any(axis=2)

    opts = kwargs["other_kwargs"]
    opts['mode_coupling'] = kwargs["other_kwargs"].get("mode_coupling", True)
    opts['contour'] = kwargs["other_kwargs"].get("contour", True)
    opts['title'] = kwargs["other_kwargs"].get("title", True)
    opts['axes'] = kwargs["other_kwargs"].get("axes", False)
    opts['colorbar'] = kwargs["other_kwargs"].get("colorbar", True)
    opts['show_PTBL'] = kwargs["other_kwargs"].get("show_PTBL", True)
    opts['cbar_v'] = kwargs["other_kwargs"].get("cbar_v", [None, None])
    opts['show_legend'] = kwargs["other_kwargs"].get("show_legend", True)
    opts['n_contour'] = kwargs["other_kwargs"].get("n_contour", 15)
    opts['marker_size'] = kwargs["other_kwargs"].get("marker_size", 80)
    opts['alpha'] = kwargs["other_kwargs"].get("alpha", 0.7)
    opts['manual_clabel'] = kwargs["other_kwargs"].get("manual_clabel", False)
    opts['colorplot'] = kwargs["other_kwargs"].get("colorplot", True)
    opts['contour_alpha'] = kwargs["other_kwargs"].get("contour_alpha", 1)
    opts['contour_linestyles'] = kwargs["other_kwargs"].get("contour_linestyles", "-")
    opts['var1_label'] = kwargs["other_kwargs"].get("var1_label", var1_label)
    opts['var2_label'] = kwargs["other_kwargs"].get("var2_label", var2_label)
    var1_grid, var2_grid, var1_plot, var2_plot = create_grid(var1, var2, var1_unit, var2_unit)
    data_map = {"xi": (xi, r"$\xi$"), "bunch_length": (bl, "Bunch length [ps]"), "R": (R, "Touschek lifetime ratio")}
    if plot_2D not in data_map:
        raise ValueError("Invalid plot_2D. Must be one of ['xi', 'bunch_length', 'R']")
    data, clabel = data_map[plot_2D]
    data = data.T

    ax = opts.get("axes") or configure_plot(
        title=opts['title'],
        xlabel=var1_label,
        ylabel=var2_label
    )

    # Plot image and contours
    plot_image(
        ax, data, [var1_plot.min(), var1_plot.max(), var2_plot.min(), var2_plot.max()],
        clabel,
        var1_grid, var2_grid,
        vmin=opts["cbar_v"][0], vmax=opts["cbar_v"][1],
        colorbar=opts['colorbar'],
        contour=opts['contour'],
        contour_dict={"levels": opts['n_contour'], "alpha": opts['contour_alpha'], "linestyles": opts['contour_linestyles']},
        colorplot=opts.get("colorplot", True)
        )
    # Plot scatter markers
    mode_coupling = opts['mode_coupling']
    scatter_opts = [ 
        (HOM_coup, '^', 'CBI driven by HOMs', "tab:purple") if HOM_coup.any() else None,
        (robinson_coup[:, :, 0], 'o', 'Dipole Robinson instability', "tab:blue"),
        (robinson_coup[:, :, 1], 'v', 'Quadrupole Robinson instability', "tab:orange"),
        (robinson_coup[:, :, 2] | robinson_coup[:, :, 3], '*', "Fast mode-coupling instability", "tab:red") if mode_coupling else None,
        (robinson_coup[:, :, 2], '>', "Sextupole Robinson instability", "tab:olive") if not mode_coupling else None,
        (robinson_coup[:, :, 3], '<', "Octupole Robinson instability", "tab:brown") if not mode_coupling else None,
        (zero_freq_coup, 'd', 'Zero-frequency instability', "tab:pink") if zero_freq_coup.any() else None,
        (PTBL_coup, 'X', 'PTBL', "tab:green") if opts.get("show_PTBL", True) else None,
        (not_converged, '1', 'Not converged', "tab:gray") if not_converged.any() else None
        ]
    for condition, marker, label, color in filter(None, scatter_opts):
        ax.scatter(var1_grid[condition], var2_grid[condition], marker=marker, label=label, alpha=opts.get("alpha", 0.7), s=opts.get("marker_size", 80), color=color)

    if opts['show_legend']:
        # Only show legend if there are labeled artists to avoid warnings
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            ax.legend(loc="lower right")
    if opts['title']:
        ax.set_title(kwargs["name"])
    if save:
        save_plot(kwargs["name"], kwargs["method"], plot_2D, kwargs.get("tau_boundary"))

def __plot_modes(out, psi_HC_vals, mode_coupling):
    """Plots mode coupling dynamics."""
    (_, _, modes_coup, _, _, _, _, _, _) = out
    labels = ["Coupled dipole mode", "Coupled quadrupole mode"] if mode_coupling else \
             ["Dipole mode", "Quadrupole mode", "Sextupole mode", "Octupole mode"]
    for i, label in enumerate(labels):
        plt.plot(psi_HC_vals, modes_coup[:, i] / (2 * np.pi), label=label)
    configure_plot(title="Mode Coupling", xlabel="Tuning angle [°]", ylabel="Frequency [Hz]")
    plt.gca().invert_xaxis()

def __plot_opti(out, var1, var2, var1_unit, var2_unit, var1_label, var2_label, plot_2D, save, **kwargs):
    """Optimized plotting for specific optimization outputs."""
    (psi, bl, xi, R) = out
    data_map = {"xi": (xi, "xi"), "bunch_length": (bl, "Bunch length [ps]"), "R": (R, "Touschek lifetime ratio"), "psi": (psi, "psi [°]")}
    if plot_2D not in data_map:
        raise ValueError("Invalid plot_2D. Must be one of ['xi', 'bunch_length', 'R', 'psi']")
    data, clabel = data_map[plot_2D]
    data = data.T
    
    opts = kwargs["other_kwargs"]
    opts['contour'] = kwargs["other_kwargs"].get("contour", True)
    opts['title'] = kwargs["other_kwargs"].get("title", True)
    opts['axes'] = kwargs["other_kwargs"].get("axes", False)
    opts['colorbar'] = kwargs["other_kwargs"].get("colorbar", True)
    opts['cbar_v'] = kwargs["other_kwargs"].get("cbar_v", [None, None])
    opts['show_legend'] = kwargs["other_kwargs"].get("show_legend", True)
    opts['n_contour'] = kwargs["other_kwargs"].get("n_contour", 15)
    opts['marker_size'] = kwargs["other_kwargs"].get("marker_size", 80)
    opts['alpha'] = kwargs["other_kwargs"].get("alpha", 0.7)
    opts['manual_clabel'] = kwargs["other_kwargs"].get("manual_clabel", False)
    opts['colorplot'] = kwargs["other_kwargs"].get("colorplot", True)
    opts['contour_alpha'] = kwargs["other_kwargs"].get("contour_alpha", 1)
    opts['contour_linestyles'] = kwargs["other_kwargs"].get("contour_linestyles", "-")
    opts['var1_label'] = kwargs["other_kwargs"].get("var1_label", var1_label)
    opts['var2_label'] = kwargs["other_kwargs"].get("var2_label", var2_label)

    _, _, var1_plot, var2_plot = create_grid(var1, var2, var1_unit, var2_unit)
    ax = opts['axes'] or configure_plot(
        title=opts['title'],
        xlabel=var1_label,
        ylabel=var2_label
    )

    # Plot image and contours
    plot_image(
        ax, data, [var1_plot.min(), var1_plot.max(), var2_plot.min(), var2_plot.max()],
        clabel, var1_plot, var2_plot, cmap="plasma", vmin=opts['cbar_v'][0], vmax=opts['cbar_v'][1],
        contour=opts['contour'],
        contour_dict={"levels": opts['n_contour'], "alpha": opts['alpha'], "linestyles": opts['contour_linestyles']},
        colorbar=opts['colorbar'],
        colorplot=opts['colorplot'],
    )

    if save:
        save_plot(kwargs["name"], kwargs["method"], plot_2D, kwargs.get("tau_boundary"))
