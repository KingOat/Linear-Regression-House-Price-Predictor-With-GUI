import tkinter as tk
import numpy as np
from keras.models import load_model

# Constantes
MIN_VALUE: float = 75000.0 
MAX_VALUE: float = 7700000.0

def recover_value(v: float) -> float:
    return v * (MAX_VALUE - MIN_VALUE) + MIN_VALUE


# Model
model = load_model("model.h5")

# Interface
# Create the main window
root = tk.Tk()
root.geometry("500x200")
root.title("House Pricing Prediction Software")

selected_features_labels = ["bedrooms", "bathrooms", "floors", "year built", "year renovated"]

# Create labels and entry widgets for the form
label_entry = []
for feature in selected_features_labels:
    label = tk.Label(root, text=f"{feature.title()}:")
    entry = tk.Entry(root)
    label_entry.append((label, entry))

# Add the labels and entry widgets to the main window
for i, couple in enumerate(label_entry):
    label, entry = couple
    label.grid(row=i, column=0, sticky="w")
    entry.grid(row=i, column=1, sticky="ew")

# Create a submit button
def submit():
    entries = [couple[1].get() for couple in label_entry]
    if all(entries):
        evaluation_input = np.array(list(map(int, entries)))
        evaluation = model.predict(evaluation_input.reshape((-1, 5)))
        evaluation = recover_value(evaluation)
        
        res_label = tk.Label(root, text=f"The estimated pricing of the house is: {evaluation[0][0]}$")
        res_label.grid(row=len(selected_features_labels)+1, column=0, columnspan=2, sticky="ew")
        

submit_button = tk.Button(root, text="Submit", command=submit)
submit_button.grid(row=len(selected_features_labels), column=0, columnspan=2, sticky="ew")

# Create result label
res_label = tk.Label(root, text="The estimated pricing of the house is: ...")
res_label.grid(row=len(selected_features_labels)+1, column=0, columnspan=2, sticky="ew")

# Configure the grid to scale the widgets
root.columnconfigure(0, weight=1)
root.columnconfigure(1, weight=1)

# Run the main loop
root.mainloop()