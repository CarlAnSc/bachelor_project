import streamlit as st
from PIL import Image
import os
import random
from itertools import compress
import pandas as pd


st.set_page_config(layout="wide")
state = st.session_state
EXAMPLE_PATH = "App/Images/example_imgs"
BASE_PATH = "App/Images/anno_imgs"
OPTIONS = ["1 Pose", "2 Partial view", "3 Object blocking", "4 Person blocking",
            "5 Multiple objects", "6 Smaller", "7 Larger", "8 Brighter", "9 Darker",
            "10 Background", "11 Color", "12 Shape", "13 Texture", "14 Pattern",
            "15 Style", "16 Subcategory"]
df_anno2examples = pd.read_csv('Data Analysis/imgs_for_app.csv')

if "annotations" not in state:
    state.annotations = {}
    state.files = os.listdir(BASE_PATH)
    state.current_file = state.files[0]

    
state.examples = os.listdir(EXAMPLE_PATH)

def store_label(label):
    state.annotations[state.current_file] = label
    state.files.remove(state.current_file)
    if state.files:
        state.current_file = random.choice(state.files)

def get_protos(anno_filename):
    protofiles = df_anno2examples['proto_file_name'][df_anno2examples['file_name'] == anno_filename]
    label = df_anno2examples['str_label'][df_anno2examples['file_name'] == anno_filename]
    return protofiles.to_list(), label.to_list()[0]

#@st.experimental_memo
def main():

    st.markdown("# Labeling")
    st.markdown("1. You can savely move back to the Introduction page to check an example. \n"
                 "2. Remember to check your submition before moving to the next page. \n"
                 "3. Do NOT refresh the page. That will make you start over! \n")


    st.sidebar.markdown("# Progress")
    st.sidebar.markdown(f"Annotated: {len(state.annotations)} — Remaining: {len(state.files)}")
    st.sidebar.markdown("# Helper")
    st.sidebar.write(f"**{OPTIONS[0]}** The object has a different pose or is placed in different location within the image.")
    st.sidebar.write(f"**{OPTIONS[1]}** The object is visible only partially due to the camera field of view that did not contain the full object – e.g. cropped out.")
    st.sidebar.write(f"**{OPTIONS[2]}** The object is occluded by another object present in the image.")
    st.sidebar.write(f"**{OPTIONS[3]}** The object is occluded by a person or human body part – this might include objects manipulated by human hands.")
    st.sidebar.write(f"**{OPTIONS[4]}** There is, at least, one another prominent object present in the image.")
    st.sidebar.write(f"**{OPTIONS[5]}** Object occupies only a small portion of the entire scene.")
    st.sidebar.write(f"**{OPTIONS[6]}** Object dominates the image.")
    st.sidebar.write(f"**{OPTIONS[7]}** The lighting in the image is brighter when compared to the prototypical images.")
    st.sidebar.write(f"**{OPTIONS[8]}** The lightning in the image is darker when compared to the prototypical images.")
    st.sidebar.write(f"**{OPTIONS[9]}** The background of the image is different from backgrounds of the prototypical images.")
    st.sidebar.write(f"**{OPTIONS[10]}** The object has different color.")
    st.sidebar.write(f"**{OPTIONS[11]}** The object has different shape.")
    st.sidebar.write(f"**{OPTIONS[12]}** The object has different texture – e.g., a sheep that’s sheared.")
    st.sidebar.write(f"**{OPTIONS[13]}** The object has different pattern – e.g., striped object.")
    st.sidebar.write(f"**{OPTIONS[14]}** The overall image style is different– e.g., a sketch.")
    st.sidebar.write(f"**{OPTIONS[15]}** The object is a distinct type or breed from the same class – e.g., a mini-van within the car class.")



    if state.files:
        # Image to label
        selected_file = state.current_file
        filename = os.path.join(BASE_PATH, selected_file)

        # Get proto images images
        list_imgs = []
        proto_images, label = get_protos(selected_file)
        for filepath in proto_images:
            filename_ex = os.path.join(EXAMPLE_PATH, filepath)
            image_ex = Image.open(filename_ex)
            list_imgs.append(image_ex)
            
        st.write(f"#### Prototypical images for \"{label}\"")
        st.image(list_imgs, width=300)

        # Get image to be labeled
        st.write("#### Image to be labeled")
        image = Image.open(filename)
        st.image(image, caption=f"{selected_file}", width=300)


        with st.form("my_form", clear_on_submit=True):
            st.write("Choose the areas where the image differs:")
            checks = st.columns(4)
            with checks[0]:
                checkbox_1 = st.checkbox(OPTIONS[0])
                checkbox_2 = st.checkbox(OPTIONS[1])
                checkbox_3 = st.checkbox(OPTIONS[2])
                checkbox_4 = st.checkbox(OPTIONS[3])
            with checks[1]:
                checkbox_5 = st.checkbox(OPTIONS[4])
                checkbox_6 = st.checkbox(OPTIONS[5])
                checkbox_7 = st.checkbox(OPTIONS[6])
                checkbox_8 = st.checkbox(OPTIONS[7])
            with checks[2]:
                checkbox_9 = st.checkbox(OPTIONS[8])
                checkbox_10 = st.checkbox(OPTIONS[9])
                checkbox_11 = st.checkbox(OPTIONS[10])
                checkbox_12 = st.checkbox(OPTIONS[11])
            with checks[3]:
                checkbox_13 = st.checkbox(OPTIONS[12])
                checkbox_14 = st.checkbox(OPTIONS[13])
                checkbox_15 = st.checkbox(OPTIONS[14])
                checkbox_16 = st.checkbox(OPTIONS[15])

            choices = [checkbox_1, checkbox_2, checkbox_3, checkbox_4, checkbox_5, checkbox_6,
                       checkbox_7, checkbox_8, checkbox_9, checkbox_10, checkbox_11, checkbox_12,
                       checkbox_13, checkbox_14, checkbox_15, checkbox_16]
            
            # Every form must have a submit button.
            submitted = st.form_submit_button("Submit") #, on_click=store_label, args=(choices,))
            if submitted:
                st.write(f"Your choices: {list(compress(OPTIONS, choices))} ")

        if submitted:
            st.button("Next", on_click=store_label, args=(choices,), help="Make sure your submitted choices are correct")
        

    else:
        st.info("Well done! Everything is annotated.")
        st.write('If you made a mistake place let ud now on ```s204154@dtu.dk```')
        st.download_button(
        "Download annotations as CSV",
        "\n".join([f"{k}\t{v}" for k, v in state.annotations.items()]),
        file_name="export.csv",
    )
    
    

    

if __name__ == "__main__":
    main()
