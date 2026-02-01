"""
Visualization functions for ALBuMS results using Plotly.
"""
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


def plot_2d_heatmap(x_vals, y_vals, z_vals, 
                    x_label="X", y_label="Y", z_label="Z",
                    title="2D Heatmap", colorscale="RdBu_r"):
    """
    Create an interactive 2D heatmap with Plotly.
    
    Parameters
    ----------
    x_vals : array-like
        X-axis values
    y_vals : array-like
        Y-axis values
    z_vals : 2D array
        Z values (heatmap data)
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    z_label : str
        Z-axis label (colorbar)
    title : str
        Plot title
    colorscale : str
        Plotly colorscale name
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    # Ensure arrays are numpy arrays
    x_vals = np.asarray(x_vals, dtype=float)
    y_vals = np.asarray(y_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)
    
    # Ensure z_vals is 2D
    if z_vals.ndim == 1:
        # If 1D, reshape based on x and y dimensions
        z_vals = z_vals.reshape(len(y_vals), len(x_vals))
    
    # Make sure dimensions match: z should be (len(y), len(x))
    if z_vals.shape[0] != len(y_vals) or z_vals.shape[1] != len(x_vals):
        # Try transposing
        if z_vals.shape[1] == len(y_vals) and z_vals.shape[0] == len(x_vals):
            z_vals = z_vals.T
    
    fig = go.Figure(data=go.Heatmap(
        x=x_vals,
        y=y_vals,
        z=z_vals,
        colorscale=colorscale,
        colorbar=dict(title=z_label),
        hovertemplate=f'{x_label}: %{{x}}<br>{y_label}: %{{y}}<br>{z_label}: %{{z:.4f}}<extra></extra>'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        template="plotly_dark",
        height=600,
        font=dict(size=12)
    )
    
    return fig


def plot_stability_map(psi_vals, current_vals, growth_rates,
                       title="Stability Map"):
    """
    Create a stability map showing growth rates.
    
    Parameters
    ----------
    psi_vals : array-like
        Psi values in degrees
    current_vals : array-like
        Current values in A
    growth_rates : 2D array
        Growth rate values
    title : str
        Plot title
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    # Create contour plot
    fig = go.Figure(data=go.Contour(
        x=psi_vals,
        y=current_vals,
        z=growth_rates,
        colorscale="RdYlGn_r",
        colorbar=dict(title="Growth Rate (1/s)"),
        contours=dict(
            start=np.min(growth_rates),
            end=np.max(growth_rates),
            size=(np.max(growth_rates) - np.min(growth_rates)) / 20
        ),
        hovertemplate='Psi: %{x}°<br>Current: %{y} A<br>Growth Rate: %{z:.2e} 1/s<extra></extra>'
    ))
    
    # Add stability boundary (zero growth rate)
    fig.add_contour(
        x=psi_vals,
        y=current_vals,
        z=growth_rates,
        contours=dict(
            start=0,
            end=0,
            size=1,
            coloring='lines'
        ),
        line=dict(color='white', width=3),
        showscale=False,
        name='Stability Boundary'
    )
    
    fig.update_layout(
        title=title,
        xaxis_title="Psi (degrees)",
        yaxis_title="Current (A)",
        template="plotly_dark",
        height=600,
        font=dict(size=12)
    )
    
    return fig


def plot_mode_frequencies(psi_vals, mode_frequencies, mode_labels=None,
                         title="Robinson Mode Frequencies"):
    """
    Plot mode frequencies vs psi.
    
    Parameters
    ----------
    psi_vals : array-like
        Psi values in degrees
    mode_frequencies : list of arrays
        List of frequency arrays for each mode
    mode_labels : list of str, optional
        Labels for each mode
    title : str
        Plot title
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if mode_labels is None:
        mode_labels = [f"Mode {i}" for i in range(len(mode_frequencies))]
    
    colors = px.colors.qualitative.Plotly
    
    for i, (freqs, label) in enumerate(zip(mode_frequencies, mode_labels)):
        fig.add_trace(go.Scatter(
            x=psi_vals,
            y=freqs,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate=f'{label}<br>Psi: %{{x}}°<br>Frequency: %{{y:.4f}} Hz<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Psi (degrees)",
        yaxis_title="Frequency (Hz)",
        template="plotly_dark",
        height=500,
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_growth_rates(psi_vals, growth_rates, mode_labels=None,
                     title="Growth Rates vs Psi"):
    """
    Plot growth rates vs psi.
    
    Parameters
    ----------
    psi_vals : array-like
        Psi values in degrees
    growth_rates : list of arrays
        List of growth rate arrays for each mode
    mode_labels : list of str, optional
        Labels for each mode
    title : str
        Plot title
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    if mode_labels is None:
        mode_labels = [f"Mode {i}" for i in range(len(growth_rates))]
    
    colors = px.colors.qualitative.Plotly
    
    for i, (rates, label) in enumerate(zip(growth_rates, mode_labels)):
        fig.add_trace(go.Scatter(
            x=psi_vals,
            y=rates,
            mode='lines+markers',
            name=label,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(size=4),
            hovertemplate=f'{label}<br>Psi: %{{x}}°<br>Growth Rate: %{{y:.2e}} 1/s<extra></extra>'
        ))
    
    # Add zero line
    fig.add_hline(y=0, line_dash="dash", line_color="white", 
                  annotation_text="Stability Threshold")
    
    fig.update_layout(
        title=title,
        xaxis_title="Psi (degrees)",
        yaxis_title="Growth Rate (1/s)",
        template="plotly_dark",
        height=500,
        font=dict(size=12),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )
    
    return fig


def plot_optimization_result(psi0, optimal_psi, bounds, r_factor=None):
    """
    Visualize optimization result.
    
    Parameters
    ----------
    psi0 : float
        Initial psi value in degrees
    optimal_psi : float
        Optimal psi value in degrees
    bounds : tuple
        (min, max) bounds in degrees
    r_factor : float, optional
        R-factor at optimal psi
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    # Plot bounds region
    fig.add_vrect(
        x0=bounds[0], x1=bounds[1],
        fillcolor="gray", opacity=0.2,
        layer="below", line_width=0,
        annotation_text="Search Region"
    )
    
    # Plot initial point
    fig.add_trace(go.Scatter(
        x=[psi0],
        y=[0],
        mode='markers',
        name='Initial',
        marker=dict(size=15, color='orange', symbol='x'),
        hovertemplate=f'Initial Psi: {psi0:.2f}°<extra></extra>'
    ))
    
    # Plot optimal point
    fig.add_trace(go.Scatter(
        x=[optimal_psi],
        y=[0],
        mode='markers',
        name='Optimal',
        marker=dict(size=20, color='lime', symbol='star'),
        hovertemplate=f'Optimal Psi: {optimal_psi:.2f}°<br>R-factor: {r_factor:.4f}<extra></extra>' if r_factor else f'Optimal Psi: {optimal_psi:.2f}°<extra></extra>'
    ))
    
    fig.update_layout(
        title="Optimization Result",
        xaxis_title="Psi (degrees)",
        yaxis=dict(visible=False),
        template="plotly_dark",
        height=300,
        font=dict(size=12),
        showlegend=True
    )
    
    return fig


def plot_r_factor_vs_psi(psi_vals, r_factors, optimal_psi=None,
                        title="R-Factor vs Psi"):
    """
    Plot R-factor as a function of psi.
    
    Parameters
    ----------
    psi_vals : array-like
        Psi values in degrees
    r_factors : array-like
        R-factor values
    optimal_psi : float, optional
        Optimal psi value to highlight
    title : str
        Plot title
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=psi_vals,
        y=r_factors,
        mode='lines+markers',
        name='R-factor',
        line=dict(color='cyan', width=2),
        marker=dict(size=6),
        hovertemplate='Psi: %{x}°<br>R-factor: %{y:.4f}<extra></extra>'
    ))
    
    if optimal_psi is not None:
        # Find closest psi value
        idx = np.argmin(np.abs(psi_vals - optimal_psi))
        fig.add_trace(go.Scatter(
            x=[psi_vals[idx]],
            y=[r_factors[idx]],
            mode='markers',
            name='Optimal',
            marker=dict(size=15, color='lime', symbol='star'),
            hovertemplate=f'Optimal: {psi_vals[idx]:.2f}°<br>R-factor: {r_factors[idx]:.4f}<extra></extra>'
        ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Psi (degrees)",
        yaxis_title="R-factor",
        template="plotly_dark",
        height=500,
        font=dict(size=12)
    )
    
    return fig
    return fig


def plot_stability_regions(psi_vals, y_vals, results, 
                          x_label="Tuning angle [°]", 
                          y_label="Current [mA]", 
                          title="Stability Map",
                          mode_coupling=True):
    """
    Create a detailed stability map with markers for different instability types.
    Matches the visual style of the original Matplotlib implementation but using Plotly.
    
    Parameters
    ----------
    psi_vals : array-like
        Psi values in degrees (X-axis)
    y_vals : array-like
        Y-axis values (Current in A or R/Q in Ohm)
    results : dict
        Dictionary containing scan results arrays:
        - xi: 2D array of xi values (background contour)
        - robinson_coup: 3D array [nx, ny, 4] for Robinson modes
        - HOM_coup: 2D boolean array for HOM instability
        - zero_freq_coup: 2D boolean array for zero frequency instability
        - PTBL_coup: 2D boolean array for PTBL
        - converged_coup: 3D boolean array for convergence
    x_label : str
        X-axis label
    y_label : str
        Y-axis label
    title : str
        Plot title
    mode_coupling : bool
        Whether mode coupling was enabled (affects marker logic)
        
    Returns
    -------
    fig : plotly.graph_objects.Figure
        Plotly figure object
    """
    import plotly.graph_objects as go
    import numpy as np

    # Unpack results
    xi = np.asarray(results.get('xi', np.zeros((len(psi_vals), len(y_vals))))).T
    robinson_coup = np.asarray(results.get('robinson_coup', np.zeros((len(psi_vals), len(y_vals), 4)))).transpose(1, 0, 2)
    HOM_coup = np.asarray(results.get('HOM_coup', np.zeros((len(psi_vals), len(y_vals))))).T
    zero_freq_coup = np.asarray(results.get('zero_freq_coup', np.zeros((len(psi_vals), len(y_vals))))).T
    PTBL_coup = np.asarray(results.get('PTBL_coup', np.zeros((len(psi_vals), len(y_vals))))).T
    
    # Handle convergence
    # Default to all True if not present
    cov = results.get('converged_coup')
    if cov is None:
        converged_coup = np.ones((len(y_vals), len(psi_vals), 2), dtype=bool)
    else:
        converged_coup = np.asarray(cov).transpose(1, 0, 2)
    
    # Calculate not_converged condition
    # Logic from albums/plot_func.py:
    # not_converged = (~converged_coup[:,:,0] | ~converged_coup[:,:,1]) & ~robinson_coup[:,:,:].any(axis=2)
    # Note: dimensions need to be handled carefully. 
    # Here we use transposed arrays (Y, X) to match Plotly's (y, x) expectation for grids
    
    # Check dimensions
    ny, nx = xi.shape
    
    # Ensure all arrays match this shape
    if HOM_coup.shape != (ny, nx): HOM_coup = np.resize(HOM_coup, (ny, nx))
    if PTBL_coup.shape != (ny, nx): PTBL_coup = np.resize(PTBL_coup, (ny, nx))
    if zero_freq_coup.shape != (ny, nx): zero_freq_coup = np.resize(zero_freq_coup, (ny, nx))
    
    # Robinson coup is (ny, nx, 4)
    # Converged coup is (ny, nx, N)
    
    robinson_any = robinson_coup.any(axis=2)
    if converged_coup.shape[2] >= 2:
        not_converged = (~converged_coup[:,:,0] | ~converged_coup[:,:,1]) & ~robinson_any
    else:
        not_converged = np.zeros((ny, nx), dtype=bool)

    # Convert y_vals to mA if the label says so
    y_plot = y_vals * 1000 if "mA" in y_label and np.max(y_vals) < 100 else y_vals
    y_label_plot = y_label
    
    if "Current" in y_label and np.max(y_vals) < 100:
        y_label_plot = "Current [mA]"
    
    # Create grids for scatter plots
    X, Y = np.meshgrid(psi_vals, y_plot)
    
    # Initialize figure
    fig = go.Figure()
    
    # 1. Background Contours (Xi) -- using Heatmap + Contours
    # Mask Xi values where not converged to avoid misleading contours
    xi_masked = xi.copy()
    xi_masked[not_converged] = np.nan  # Hide non-converged regions in contours
    
    # Also check for unrealistic Xi values
    xi_masked[~np.isfinite(xi_masked)] = np.nan
    xi_masked[xi_masked < 0] = np.nan
    xi_masked[xi_masked > 10] = np.nan  # Cap unrealistic values
    
    # Only show contours if we have enough valid data
    valid_ratio = np.sum(np.isfinite(xi_masked)) / xi_masked.size
    
    if valid_ratio > 0.1:  # At least 10% valid points
        # Trace 1: Standard Xi levels from 0.2 to 1.0 with 0.1 step
        fig.add_trace(go.Contour(
            z=xi_masked,
            x=psi_vals,
            y=y_plot,
            colorscale="Greys",
            opacity=0.5,
            showscale=False,
            contours=dict(
                coloring='lines',
                showlabels=True,
                labelfont=dict(size=10, color='gray'),
                start=0.2,
                end=1.0,
                size=0.1,
            ),
            line=dict(width=1, dash='dot'),
            name="Xi Isolines (0.2-1.0)",
            hoverinfo='skip',
            connectgaps=False  # Don't connect across gaps
        ))
        
        # Trace 2: Specific Xi level 1.05 (Stability Limit) - red line
        # Only draw if we have Xi values near 1.05
        if np.nanmax(xi_masked) > 1.0:
            fig.add_trace(go.Contour(
                z=xi_masked,
                x=psi_vals,
                y=y_plot,
                colorscale=[[0, 'red'], [1, 'red']], # Fixed red color
                showscale=False,
                contours=dict(
                    coloring='lines',
                    showlabels=True,
                    labelfont=dict(size=10, color='red'),
                    start=1.05,
                    end=1.05,
                    size=0.1, # Dummy size
                ),
                line=dict(width=3, color='red'), # Thicker red line for visibility
                name="Xi = 1.05 (PTBL threshold)",
                hoverinfo='skip',
                connectgaps=False
            ))
    else:
        # Not enough converged data for meaningful contours
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="⚠️ Low convergence: Xi contours unreliable",
            showarrow=False,
            font=dict(size=14, color="orange"),
            bgcolor="rgba(0,0,0,0.5)"
        )
    
    # Note: The red lines in the plot represent the ξ=1.05 threshold
    # This is the theoretical PTBL (Periodic Transient Beam Loading) instability boundary
    # Scattered/jagged red lines indicate convergence issues
    
    # 2. Add Scatter Markers for each condition
    
    # Define markers conditions and styles
    # (Condition Array, Marker Symbol, Legend Label, Color)
    
    markers_config = [
        (HOM_coup, 'triangle-up', 'CBI driven by HOMs', 'purple'),
        (robinson_coup[:, :, 0], 'circle', 'Dipole Robinson instability', 'blue'),
        (robinson_coup[:, :, 1], 'triangle-down', 'Quadrupole Robinson instability', 'orange'),
        (robinson_coup[:, :, 2] | robinson_coup[:, :, 3], 'star', "Fast mode-coupling instability", 'red') if mode_coupling else None,
        (zero_freq_coup, 'diamond', 'Zero-frequency instability', 'pink'),
        (PTBL_coup, 'x', 'PTBL or l=1 CBI', 'green'),
        (not_converged, 'y-down', 'Not converged', 'gray')
    ]
    
    # Flatten arrays for scatter plotting to optimize (only plot True points)
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    for item in markers_config:
        if item is None: continue
        
        condition, symbol, label, color = item
        
        if condition.shape != X.shape:
             # Try to match shape
             continue
             
        mask = condition.flatten()
        if not np.any(mask):
            continue
            
        fig.add_trace(go.Scatter(
            x=X_flat[mask],
            y=Y_flat[mask],
            mode='markers',
            marker=dict(
                symbol=symbol,
                size=8,
                color=color,
                line=dict(width=1, color='white') if symbol != 'x' else dict(width=2)
            ),
            name=label,
            hovertemplate=f"{label}<br>{x_label}: %{{x:.2f}}<br>{y_label_plot}: %{{y:.2f}}<extra></extra>"
        ))
        
    # Add a dummy trace for "Stable beam" if we want to show it explicitly?
    # In the image, blue circles are "Stable beam", but here 'circle' is Dipole Robinson?
    # Wait, check check plot_func.py:
    # (robinson_coup[:, :, 0], 'o', 'Dipole Robinson instability', "tab:blue")
    # In the image, Blue Circle = "Stable beam". 
    # Ah, in `plot_func.py`, it marks robinson_coup mode 0 with blue circles.
    # Usually mode 0 is the dipole mode. If it's unstable, it's Dipole Robinson.
    # Maybe the "Stable beam" is where NO markers are present? 
    # The image shows blue circles for "Stable beam".
    # But `plot_func.py` labels them "Dipole Robinson instability". 
    # Maybe the user's image is slightly different or the label in `plot_func.py` is for when it IS unstable?
    # `robinson_coup` contains booleans (True/False). 
    # So `robinson_coup[:,:,0]` True means Dipole Mode is Unstable.
    # So where are the Stable Beam markers?
    # In `check_stability.py` or similar usage, maybe they plot stable points?
    # In the image, "Stable beam" markers are blue open circles.
    # ALBuMS `plot_func.py` doesn't explicitly plot "Stable beam". It only plots instabilities.
    # Points with NO instability are blank.
    # The user's image has "Stable beam" markers (blue circles).
    # And "Dipole Robinson" is not in the legend of the user's image. 
    # Wait, maybe `robinson_coup` stores something else?
    # `robinson_coup` is boolean.
    
    # Let's look at the image again.
    # "Stable beam" (Blue circle).
    # "CBI driven by HOMs" (Purple triangle).
    # "Fast mode coupling instability" (Red star).
    # "PTBL" (Green X).
    # "Quadrupole Robinson" (Orange inverted triangle).
    # "Not converged" (Y).
    
    # The user's image legend doesn't list Dipole Robinson.
    # My hypothesis: The user's image might be from a specific configuration where Mode 0 is always stable or not of interest, 
    # OR the "Stable beam" markers are added separately for points where everything is False.
    
    # I will add a "Stable beam" condition:
    # All instabilities are False AND converged is True.
    all_instabilities = (
        HOM_coup | 
        robinson_coup.any(axis=2) | 
        zero_freq_coup | 
        PTBL_coup
    )
    stable_mask = (~all_instabilities) & (~not_converged) & converged_coup.any(axis=2).any(axis=0 if converged_coup.ndim==3 else -1) # converged check is tricky
    # Actually simpler: if not not_converged and not any instability.
    
    is_stable = (~all_instabilities) & (~not_converged)
    
    if np.any(is_stable):
         fig.add_trace(go.Scatter(
            x=X_flat[is_stable.flatten()],
            y=Y_flat[is_stable.flatten()],
            mode='markers',
            marker=dict(
                symbol='circle-open', # Open circle for stable
                size=6,
                color='blue',
                line=dict(width=1)
            ),
            name="Stable beam",
            hovertemplate=f"Stable Beam<br>{x_label}: %{{x:.2f}}<br>{y_label_plot}: %{{y:.2f}}<extra></extra>"
        ))

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label_plot,
        template="plotly_white",
        height=600,
        font=dict(size=12),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig
