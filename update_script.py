with open("generate_complete_2d_figure.py", "r") as f:
    text = f.read()

# Replace the figure creation and colorbar logic
old_block = """    # ------------------------------------------------------------------
    # Render Master 2x3 Figure: 5 Heatmaps + 1 Quantitative Bar Chart
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(16, 9.8), dpi=300)
    fig.patch.set_facecolor('white')

    panel_coords = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1)]
    panel_titles = [
        ("A. Deep Ensembles (Gaussian NLL)", '#E66101'),
        ("B. Variational BNN (BBB)", '#5E3C99'),
        ("C. BootDQN + Randomized Priors", '#B27B00'),
        ("D. MC Dropout (p=0.10)", '#2B83BA'),
        ("E. DP-BNN (Ours: Measure Prior)", '#B2182B')
    ]

    im_shared = None
    for idx, (m, (r, c)) in enumerate(zip(model_order, panel_coords)):
        ax = axes[r, c]
        im_shared = ax.imshow(
            bald_maps[m],
            extent=[-2.5, 2.5, -2.5, 2.5],
            origin='lower',
            cmap='plasma',
            vmin=0.0,
            vmax=1.8,
            interpolation='bicubic'
        )
        # Overlay Cavity (cyan dashed circle) and Annulus (black dotted)
        ax.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.2, label='Cavity ($r \leq 1.0$)'))
        ax.add_patch(plt.Circle((0, 0), 1.3, color='white', fill=False, linestyle=':', lw=1.5, alpha=0.8))
        ax.add_patch(plt.Circle((0, 0), 2.45, color='white', fill=False, linestyle=':', lw=1.5, alpha=0.8))

        title_text, title_col = panel_titles[idx]
        ax.set_title(title_text, fontsize=11, fontweight='bold', color=title_col, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=10)
        ax.set_ylabel(r"$x_2$", fontsize=10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)

    # Colorbar on right of heatmaps
    plt.tight_layout(rect=[0, 0, 0.98, 0.96])
    cbar_ax = fig.add_axes([0.655, 0.55, 0.012, 0.38])
    cbar = fig.colorbar(im_shared, cax=cbar_ax, orientation='vertical')
    cbar.set_label("BALD Epistemic Uncertainty (nats)", fontsize=9.5, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.5)

    # Panel F: Bar Chart in axes[1, 2]
    ax_bar = axes[1, 2]"""

new_block = """    # ------------------------------------------------------------------
    # Render Master 2x3 Figure with Clean Colorbar Placement
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(16.8, 9.8), dpi=300)
    fig.patch.set_facecolor('white')

    # GridSpec with 2 rows, 3 columns for subplots, and 1 dedicated column for colorbar
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(2, 4, width_ratios=[1, 1, 1, 0.045], wspace=0.28, hspace=0.28,
                  left=0.05, right=0.96, top=0.92, bottom=0.06)

    axes_list = [
        fig.add_subplot(gs[0, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[0, 2]),
        fig.add_subplot(gs[1, 0]),
        fig.add_subplot(gs[1, 1])
    ]

    panel_titles = [
        ("A. Deep Ensembles (Gaussian NLL)", '#E66101'),
        ("B. Variational BNN (BBB)", '#5E3C99'),
        ("C. BootDQN + Randomized Priors", '#B27B00'),
        ("D. MC Dropout (p=0.10)", '#2B83BA'),
        ("E. DP-BNN (Ours: Measure Prior)", '#B2182B')
    ]

    im_shared = None
    for idx, (m, ax) in enumerate(zip(model_order, axes_list)):
        im_shared = ax.imshow(
            bald_maps[m],
            extent=[-2.5, 2.5, -2.5, 2.5],
            origin='lower',
            cmap='plasma',
            vmin=0.0,
            vmax=1.8,
            interpolation='bicubic'
        )
        # Overlay Cavity (cyan dashed circle) and Annulus (white dotted)
        ax.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.2))
        ax.add_patch(plt.Circle((0, 0), 1.3, color='white', fill=False, linestyle=':', lw=1.5, alpha=0.8))
        ax.add_patch(plt.Circle((0, 0), 2.45, color='white', fill=False, linestyle=':', lw=1.5, alpha=0.8))

        title_text, title_col = panel_titles[idx]
        ax.set_title(title_text, fontsize=11, fontweight='bold', color=title_col, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=10)
        ax.set_ylabel(r"$x_2$", fontsize=10)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=9)

    # Dedicated Colorbar Axis spanning top & bottom or top row
    cbar_ax = fig.add_subplot(gs[0, 3])
    cbar = fig.colorbar(im_shared, cax=cbar_ax, orientation='vertical')
    cbar.set_label("BALD Epistemic Uncertainty (nats)", fontsize=9.5, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.5)

    # Panel F: Bar Chart in gs[1, 2]
    ax_bar = fig.add_subplot(gs[1, 2])"""

if old_block in text:
    text = text.replace(old_block, new_block)
    with open("generate_complete_2d_figure.py", "w") as f:
        f.write(text)
    print("Successfully replaced layout code!")
else:
    print("Old block not found!")
