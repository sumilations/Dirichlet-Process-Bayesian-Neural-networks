with open("test_1x5_figure.py", "r") as f:
    text = f.read()

old_plot = """    # ------------------------------------------------------------------
    # Render 1x5 Horizontal Publication Figure
    # A: Data & Geometry
    # B: Deep Ensembles Heatmap
    # C: BootDQN + Priors Heatmap
    # D: DP-BNN (Ours) Heatmap
    # Colorbar
    # E: Quantitative Bar Chart (with MC Dropout)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18.5, 3.8), dpi=300)
    fig.patch.set_facecolor('white')

    gs = GridSpec(1, 6, width_ratios=[1, 1, 1, 1, 0.045, 1.25], wspace=0.25,
                  left=0.04, right=0.97, top=0.88, bottom=0.14)

    # Panel A: Data & Geometry
    ax_a = fig.add_subplot(gs[0, 0])
    Z_true = ground_truth_2d(XX, YY)
    ax_a.contourf(XX, YY, Z_true, levels=25, cmap='coolwarm', alpha=0.55)
    ax_a.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=16, edgecolors='black', lw=0.5,
                 label=r'Data ($\sigma_\epsilon=0.30$)')
    ax_a.add_patch(plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', lw=2.0, label=r'Cavity ($r \leq 1.0$)'))
    ax_a.add_patch(plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.4, label=r'Annulus ($1.3 \leq r \leq 2.45$)'))
    ax_a.add_patch(plt.Circle((0, 0), 2.45, color='black', fill=False, linestyle=':', lw=1.4))
    ax_a.set_title("A. Annulus & Cavity Geometry", fontsize=10.5, fontweight='bold', pad=8)
    ax_a.set_xlim(-2.5, 2.5); ax_a.set_ylim(-2.5, 2.5)
    ax_a.set_xlabel(r"$x_1$", fontsize=9.5); ax_a.set_ylabel(r"$x_2$", fontsize=9.5)
    ax_a.legend(loc='lower left', fontsize=7.0, framealpha=0.92)
    ax_a.set_aspect('equal')
    ax_a.tick_params(labelsize=8.5)

    # Panels B, C, D: Heatmaps
    heatmap_configs = [
        ("B. Deep Ensembles (Gaussian NLL)", 'Deep Ensembles (Gaussian NLL)', gs[0, 1], '#E66101'),
        ("C. BootDQN + Rand Priors", 'BootDQN + Rand Priors', gs[0, 2], '#B27B00'),
        ("D. DP-BNN (Ours)", 'DP-BNN (Ours)', gs[0, 3], '#B2182B')
    ]

    im_shared = None
    for title, mkey, loc, col in heatmap_configs:
        ax = fig.add_subplot(loc)
        im_shared = ax.imshow(
            bald_maps[mkey],
            extent=[-2.5, 2.5, -2.5, 2.5],
            origin='lower',
            cmap='plasma',
            vmin=0.0,
            vmax=1.8,
            interpolation='bicubic'
        )
        ax.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0))
        ax.add_patch(plt.Circle((0, 0), 1.3, color='white', fill=False, linestyle=':', lw=1.3, alpha=0.8))
        ax.add_patch(plt.Circle((0, 0), 2.45, color='white', fill=False, linestyle=':', lw=1.3, alpha=0.8))
        ax.set_title(title, fontsize=10.5, fontweight='bold', color=col, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=9.5); ax.set_ylabel(r"$x_2$", fontsize=9.5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8.5)

    # Colorbar in gs[0, 4]
    cbar_ax = fig.add_subplot(gs[0, 4])
    cbar = fig.colorbar(im_shared, cax=cbar_ax, orientation='vertical')
    cbar.set_label("BALD (nats)", fontsize=9.0, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.0)

    # Panel E: Quantitative Bar Chart in gs[0, 5]
    ax_bar = fig.add_subplot(gs[0, 5])"""

new_plot = """    # ------------------------------------------------------------------
    # Render 1x5 Horizontal Publication Figure with Colorbar under heatmaps
    # A: Data & Geometry
    # B: Deep Ensembles Heatmap
    # C: BootDQN + Priors Heatmap
    # D: DP-BNN (Ours) Heatmap
    # E: Quantitative Bar Chart (with MC Dropout)
    # ------------------------------------------------------------------
    fig = plt.figure(figsize=(18.8, 4.3), dpi=300)
    fig.patch.set_facecolor('white')

    # 2 rows: row 0 for main plots, row 1 for slim horizontal colorbar under B,C,D
    gs = GridSpec(2, 5, width_ratios=[1, 1, 1, 1, 1.3], height_ratios=[1, 0.05],
                  wspace=0.32, hspace=0.28, left=0.04, right=0.97, top=0.90, bottom=0.12)

    # Panel A: Data & Geometry
    ax_a = fig.add_subplot(gs[0, 0])
    Z_true = ground_truth_2d(XX, YY)
    ax_a.contourf(XX, YY, Z_true, levels=25, cmap='coolwarm', alpha=0.55)
    ax_a.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm', s=16, edgecolors='black', lw=0.5,
                 label=r'Data ($\sigma_\epsilon=0.30$)')
    ax_a.add_patch(plt.Circle((0, 0), 1.0, color='yellow', fill=False, linestyle='--', lw=2.0, label=r'Cavity ($r \leq 1.0$)'))
    ax_a.add_patch(plt.Circle((0, 0), 1.3, color='black', fill=False, linestyle=':', lw=1.4, label=r'Annulus ($1.3 \leq r \leq 2.45$)'))
    ax_a.add_patch(plt.Circle((0, 0), 2.45, color='black', fill=False, linestyle=':', lw=1.4))
    ax_a.set_title("A. Annulus & Cavity Geometry", fontsize=10.5, fontweight='bold', pad=8)
    ax_a.set_xlim(-2.5, 2.5); ax_a.set_ylim(-2.5, 2.5)
    ax_a.set_xlabel(r"$x_1$", fontsize=9.5); ax_a.set_ylabel(r"$x_2$", fontsize=9.5)
    ax_a.legend(loc='lower left', fontsize=7.2, framealpha=0.92)
    ax_a.set_aspect('equal')
    ax_a.tick_params(labelsize=8.5)

    # Panels B, C, D: Heatmaps
    heatmap_configs = [
        ("B. Deep Ensembles (Gaussian NLL)", 'Deep Ensembles (Gaussian NLL)', gs[0, 1], '#E66101'),
        ("C. BootDQN + Rand Priors", 'BootDQN + Rand Priors', gs[0, 2], '#B27B00'),
        ("D. DP-BNN (Ours)", 'DP-BNN (Ours)', gs[0, 3], '#B2182B')
    ]

    im_shared = None
    for title, mkey, loc, col in heatmap_configs:
        ax = fig.add_subplot(loc)
        im_shared = ax.imshow(
            bald_maps[mkey],
            extent=[-2.5, 2.5, -2.5, 2.5],
            origin='lower',
            cmap='plasma',
            vmin=0.0,
            vmax=1.8,
            interpolation='bicubic'
        )
        ax.add_patch(plt.Circle((0, 0), 1.0, color='cyan', fill=False, linestyle='--', lw=2.0))
        ax.add_patch(plt.Circle((0, 0), 1.3, color='white', fill=False, linestyle=':', lw=1.3, alpha=0.8))
        ax.add_patch(plt.Circle((0, 0), 2.45, color='white', fill=False, linestyle=':', lw=1.3, alpha=0.8))
        ax.set_title(title, fontsize=10.5, fontweight='bold', color=col, pad=8)
        ax.set_xlabel(r"$x_1$", fontsize=9.5); ax.set_ylabel(r"$x_2$", fontsize=9.5)
        ax.set_aspect('equal')
        ax.tick_params(labelsize=8.5)

    # Horizontal Colorbar spanning under B, C, D: gs[1, 1:4]
    cbar_ax = fig.add_subplot(gs[1, 1:4])
    cbar = fig.colorbar(im_shared, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("BALD Epistemic Uncertainty (nats)", fontsize=9.0, fontweight='bold')
    cbar.ax.tick_params(labelsize=8.0)

    # Panel E: Quantitative Bar Chart in gs[0, 4]
    ax_bar = fig.add_subplot(gs[0, 4])"""

if old_plot in text:
    text = text.replace(old_plot, new_plot)
    with open("test_1x5_figure.py", "w") as f:
        f.write(text)
    print("Updated test_1x5_figure.py layout successfully!")
else:
    print("Old plot block not found!")
