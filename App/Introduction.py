import os
import time
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
import pandas as pd


#@st.experimental_memo
st.set_page_config(layout="wide")
df_intro = pd.read_csv('Data_Analysis/imgs_for_intro.csv')

url = 'https://storage.cloud.google.com/bachebucket'

OPTIONS = ["1 Pose", "2 Partial view", "3 Object blocking", "4 Person blocking",
            "5 Multiple objects", "6 Smaller", "7 Larger", "8 Brighter", "9 Darker",
            "10 Background", "11 Color", "12 Shape", "13 Texture", "14 Pattern",
            "15 Style", "16 Subcategory"]

def imagesFormatter(deviation_type):
    """The first three images being prototypical. The last being the image you want to illustrate."""
    filenames_proto, str_example, label = getFiles(deviation_type)
    st.write(f'**Object:** {label}')
    ROOT1 = '/Images/intro_protos/'
    ROOT2 = '/Images/intro_examples/'
    col1, col2 = st.columns([3, 1], gap='large')
    col1.image([(url + ROOT1 + filenames_proto[0]),(url + ROOT1 + filenames_proto[1]),(url + ROOT1 + filenames_proto[2])], width=250, caption=['','Prototypical images',''])
    col2.image((url + ROOT2 + str_example), width=250, caption='Example image')

def getFiles(deviation_type: str):
    idx = df_intro[deviation_type] == 1
    str_example, list_protos, label = df_intro[idx]['file_name'].iloc[0], df_intro[idx]['proto_file_name'].to_list(), df_intro[idx]['str_label'].iloc[0]
    return list_protos, str_example, label

def main():
    st.markdown("# Introduction")
    st.sidebar.markdown("# Thank you for helping us!")

    st.write("We thank you a lot for your help! 🌟 It is important that you read the following carefully before starting labeling, such that the analysis can be of high quality.")
    st.write("Due to Cloud Run limitation, you have exactly one hour from now on")
    st.markdown("## The metalabels")
    st.write("The metalabels are extra labels for images of objects. "
                "These describe how the object and image differs from the a perfect image of the same object. "
                "The perfect images are called **Prototypical** images. "
                "There are 16 different metalabels, and an image can contain multiple of these, or maybe even none. "
                 "The 16 metalabels are listed here:")
    st.write(f"#### {OPTIONS[0]} \n **Description:** The object has a different pose or is placed in different location within the image.")
    imagesFormatter('pose')
    st.write(f"#### {OPTIONS[1]} \n **Description:** The object is visible only partially due to the camera field of view that did not contain the full object – e.g. cropped out.")
    imagesFormatter('partial_view')
    st.write(f"#### {OPTIONS[2]} \n **Description:** The object is occluded by another object present in the image.")
    imagesFormatter('object_blocking')
    st.write(f"#### {OPTIONS[3]} \n **Description:** The object is occluded by a person or human body part – this might include objects manipulated by human hands.")
    imagesFormatter('person_blocking')
    st.write(f"#### {OPTIONS[4]} \n **Description:** There is, at least, one another prominent object present in the image.")
    imagesFormatter('multiple_objects')
    st.write(f"#### {OPTIONS[5]} \n **Description:** Object occupies only a small portion of the entire scene.")
    imagesFormatter('smaller')
    st.write(f"#### {OPTIONS[6]} \n **Description:** Object dominates the image.")
    imagesFormatter('larger')
    st.write(f"#### {OPTIONS[7]} \n **Description:** The lighting in the image is brighter when compared to the prototypical images.")
    imagesFormatter('brighter')
    st.write(f"#### {OPTIONS[8]} \n **Description:** The lightning in the image is darker when compared to the prototypical images.")
    imagesFormatter('darker')
    st.write(f"#### {OPTIONS[9]} \n **Description:** The background of the image is different from backgrounds of the prototypical images.")
    imagesFormatter('background')
    st.write(f"#### {OPTIONS[10]} \n **Description:** The object has different color.")
    imagesFormatter('color')
    st.write(f"#### {OPTIONS[11]} \n **Description:** The object has different shape.")
    imagesFormatter('shape')
    st.write(f"#### {OPTIONS[12]} \n **Description:** The object has different texture – e.g., a sheep that’s sheared.")
    imagesFormatter('texture')
    st.write(f"#### {OPTIONS[13]} \n **Description:** The object has different pattern – e.g., striped object.")
    imagesFormatter('pattern')
    st.write(f"#### {OPTIONS[14]} \n **Description:** The overall image style is different– e.g., a sketch.")
    imagesFormatter('style')
    st.write(f"#### {OPTIONS[15]} \n **Description:** The object is a distinct type or breed from the same class – e.g., a mini-van within the car class.")
    imagesFormatter('subcategory')

    st.markdown("## How to label")
    st.write("Wow, that is a lot to remember! Don't worry, the description will be present on the sidebar of the next page, when you are doing the labeling.")

    

    


if __name__ == "__main__":
    main()
