with open("generate_figure1_with_dataplot.py", "r") as f:
    text = f.read()

# Remove suptitle and adjust top margin
old_part = """    plt.suptitle("2D Annular Manifold Regression: Manifold Geometry, BALD Epistemic Uncertainty Heatmaps & Quantitative Contrast",
                 fontsize=12.5, fontweight='bold', y=0.98)"""

new_part = """    # Top suptitle removed as per publication standards"""

text = text.replace(old_part, new_part)
text = text.replace("top=0.92", "top=0.95")

with open("generate_figure1_with_dataplot.py", "w") as f:
    f.write(text)

