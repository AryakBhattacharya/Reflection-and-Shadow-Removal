README.md

Shadow & Reflection Removal and Enhancement

How to Run the Code:

1. The code contains various libreries which are required to make sure that the code runs smoothly. The libraries installed for the code are as follows:
    a) *PyTorch (model training + inference)*
    b) *Torchvision (transforms)*
    c) *OpenCV (classical image processing)*
    d) *NumPy*
    e) *Matplotlib*
    f) *Pillow (image loading for GUI)*
    g) *Tkinter (GUI interface)*  and
    h) *KaggleHub (dataset downloading)*

2. The *main.ipynb* file can be run both online or offline using various environments such as Google Colab or Jupyter respectively. I have also provided *main.py* version of the file so that it can be executed via VS Code or any other editor of choice.
Executing this code generates the *G_shadow_5ch.pt* and *G_refl_5ch.pt* model files, which will be used later for the GUI.

3. After the execution Of the above *main.py* file tt will generate the *generator* models which is later used for further processing.

4. The *model.py* file contains the Pix2Pix UNet Generator. It helps in executing the GUI for the program. It doesnt need to be executed as it simply defines classes and it is called by the *gui_app.py* file.

5. The *helper_functions.py* file contains
    a) classical shadow mask extraction
    b) classical reflection mask extraction
    c) image preprocessing
    d) tensor ↔ image conversion    and
    e) utility operations
functions, and it is also called by the *gui_app.py* file and doesnt need to be executed.

6. The *gui_app.py* file generates the GUI for the program. The *Tkinter user interface* allows a) Uploading an image
       b) Selecting Shadow Removal or Reflection Removal
       c) Running the corresponding GAN model   and
       d) Displaying and saving results


7. The *gui_app.py*, *helper_functions.py*, *model.py*, *G_shadow_5ch.pt*, and *G_refl_5ch.pt* model files need to be placed in the same directory for seemless execution.
