# glitchlab/app/controls.py

from tkinter import ttk


def labeled(parent, text, widget):
    f = ttk.Frame(parent)
    ttk.Label(f, text=text).pack(side="left")
    widget.pack(side="left", fill="x", expand=True, padx=6)
    return f


def spin(parent, from_, to, var, width=8, inc=1):
    return ttk.Spinbox(parent, from_=from_, to=to, textvariable=var, width=width, increment=inc)


def enum_combo(parent, values, var, width=12):
    return ttk.Combobox(parent, values=values, textvariable=var, state="readonly", width=width)


def checkbox(parent, text, var):
    return ttk.Checkbutton(parent, text=text, variable=var)


def slider(parent, from_, to, var, orient="horizontal"):
    return ttk.Scale(parent, from_=from_, to=to, variable=var, orient=orient)
