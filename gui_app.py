import cv2
import torch
import numpy as np
import torchvision.transforms as T
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
from helper_functions import (
    classical_shadow_mask, 
    classical_reflection_mask,
    classical_reflection_edges,
    tensor_to_image
)

from model import Pix2PixGenerator   # <-- use your existing generator class

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load both generators
G_shadow = Pix2PixGenerator(in_ch=5, out_ch=3).to(device)
G_shadow.load_state_dict(torch.load("G_shadow_5ch.pt", map_location=device))
G_shadow.eval()

G_refl = Pix2PixGenerator(in_ch=5, out_ch=3).to(device)
G_refl.load_state_dict(torch.load("G_refl_5ch.pt", map_location=device))
G_refl.eval()

transform = T.Compose([
    T.ToPILImage(),
    T.Resize((256,256)),
    T.ToTensor()
])

current_img = None
processed_img = None

# -----------------------------------
# GUI Actions
# -----------------------------------
def load_image():
    global current_img

    path = filedialog.askopenfilename()
    if not path:
        return

    current_img = cv2.imread(path)

    show_preview(current_img, panel_input)

def run_model():
    global processed_img

    if current_img is None:
        return

    img = current_img.copy()

    mode = mode_var.get()

    # ---------- SHADOW REMOVAL ----------
    if mode == "shadow":
        mask = classical_shadow_mask(img)
        edges = mask.copy()  # simple edges (optional)

    # -------- REFLECTION REMOVAL --------
    else:
        mask = classical_reflection_mask(img)
        edges = classical_reflection_edges(img)

    mask = cv2.resize(mask, (256,256))
    edges = cv2.resize(edges, (256,256))

    # convert input RGB
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (256,256))

    rgb_t = transform(rgb) * 2 - 1       # [-1,1]
    mask_t = torch.from_numpy(mask/255.).float().unsqueeze(0)
    edges_t = torch.from_numpy(edges/255.).float().unsqueeze(0)

    inp = torch.cat([rgb_t, mask_t, edges_t], dim=0).unsqueeze(0).to(device)

    with torch.no_grad():
        if mode == "shadow":
            out = G_shadow(inp)[0]
        else:
            out = G_refl(inp)[0]

    processed_img = tensor_to_image(out)
    show_preview(processed_img, panel_output)

def save_output():
    if processed_img is None:
        return
    path = filedialog.asksaveasfilename(defaultextension=".png")
    if path:
        cv2.imwrite(path, processed_img)

# -----------------------------------
# Tkinter GUI Layout
# -----------------------------------
root = Tk()
root.title("Shadow / Reflection Removal")
root.geometry("900x600")

mode_var = StringVar(value="shadow")

Label(root, text="Select Mode:").pack()

Radiobutton(root, text="Shadow Removal", variable=mode_var, value="shadow").pack()
Radiobutton(root, text="Reflection Removal", variable=mode_var, value="reflection").pack()

Button(root, text="Load Image", command=load_image).pack(pady=5)
Button(root, text="Run Model", command=run_model).pack(pady=5)
Button(root, text="Save Output", command=save_output).pack(pady=5)

panel_input = Label(root)
panel_input.pack(side="left", padx=10)

panel_output = Label(root)
panel_output.pack(side="right", padx=10)

def show_preview(img, panel):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil = img_pil.resize((350,350))
    img_tk = ImageTk.PhotoImage(img_pil)
    panel.config(image=img_tk)
    panel.image = img_tk

root.mainloop()