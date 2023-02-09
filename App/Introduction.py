import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image
from torchvision import transforms


#@st.experimental_memo

OPTIONS = ["1 Pose", "2 Partial view", "3 Object blocking", "4 Person blocking",
            "5 Multiple objects", "6 Smaller", "7 Larger", "8 Brighter", "9 Darker",
            "10 Background", "11 Color", "12 Shape", "13 Texture", "14 Pattern",
            "15 Style", "16 Subcategory"]

def main():
    st.markdown("# Introduction")
    st.sidebar.markdown("# Thank you for helping us!")
    st.write("We thank you a lot for your help. It is important that you read the following carefully before starting labeling, such that the analysis can be of high quality.")
    st.markdown("## The metalabels")
    st.write("The metalabels are extra labels for images of objects. These describe how the object and image differs from the a perfect image of the same object. The perfect images are called **Prototypical** images.")
    st.write("There are 16 different metalabels, and an image can contain multiple of these, or maybe even none. The 16 metalabels are listed here:")
    st.write(f"#### {OPTIONS[0]} \n The object has a different pose or is placed in different location within the image.")
    st.write(f"#### {OPTIONS[1]} \n The object is visible only partially due to the camera field of view that did not contain the full object – e.g. cropped out.")
    st.write(f"#### {OPTIONS[2]} \n The object is occluded by another object present in the image.")
    st.write(f"#### {OPTIONS[3]} \n The object is occluded by a person or human body part – this might include objects manipulated by human hands.")
    st.write(f"#### {OPTIONS[4]} \n There is, at least, one another prominent object present in the image.")
    st.write(f"#### {OPTIONS[5]} \n Object occupies only a small portion of the entire scene.")
    st.write(f"#### {OPTIONS[6]} \n Object dominates the image.")
    st.write(f"#### {OPTIONS[7]} \n The lighting in the image is brighter when compared to the prototypical images.")
    st.write(f"#### {OPTIONS[8]} \n The lightning in the image is darker when compared to the prototypical images.")
    st.write(f"#### {OPTIONS[9]} \n The background of the image is different from backgrounds of the prototypical images.")
    st.write(f"#### {OPTIONS[10]} \n The object has different color.")
    st.write(f"#### {OPTIONS[11]} \n The object has different shape.")
    st.write(f"#### {OPTIONS[12]} \n The object has different texture – e.g., a sheep that’s sheared.")
    st.write(f"#### {OPTIONS[13]} \n The object has different pattern – e.g., striped object.")
    st.write(f"#### {OPTIONS[14]} \n he overall image style is different– e.g., a sketch.")
    st.write(f"#### {OPTIONS[15]} \n The object is a distinct type or breed from the same class – e.g., a mini-van within the car class.")

    st.markdown("## How to label")


    #model = get_model()
    #model.eval()
    

    


if __name__ == "__main__":
    main()
