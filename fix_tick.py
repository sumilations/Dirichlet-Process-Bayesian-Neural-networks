with open("generate_complete_2d_figure.py", "r") as f:
    text = f.read()

text = text.replace("methods_bar = ['Deep Ensembles', 'BBB', 'BootDQN+Priors', 'MC Dropout', 'DP-BNN\\n(Ours)']",
                    "methods_bar = ['Deep Ensembles', 'BBB', 'BootDQN\\n+ Priors', 'MC Dropout', 'DP-BNN\\n(Ours)']")

with open("generate_complete_2d_figure.py", "w") as f:
    f.write(text)

