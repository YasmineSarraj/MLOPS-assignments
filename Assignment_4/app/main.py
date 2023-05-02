import streamlit as st
import mlflow
import torch 
import numpy as np
from tqdm import tqdm_notebook
from PIL import Image

st.title('Do you actually know your penguins')

# Load the pre-trained model
model = torch.nn.Sequential(
                            # Input layer
                            torch.nn.Linear(4,8),
                            torch.nn.ReLU(),
                            # 1. Hidden layer
                            torch.nn.Linear(8,16),
                            torch.nn.ReLU(),
                            # 2. Hidden layer
                            torch.nn.Linear(16,32),
                            torch.nn.ReLU(),
                            # 3. Hidden layer
                            torch.nn.Linear(32,64),
                            torch.nn.ReLU(),
                            # Output layer
                            torch.nn.Linear(64,3),
                            torch.nn.Softmax()  # We have a multiclass single-label classification problem, since a penguin can only be from one species
                          )
model.load_state_dict(torch.load('model.pt'))


image = Image.open("meme.jpg")
image2=  Image.open("gentoo.jpg")
st.image(image, caption='Accuraete representation of us coding')

flipper = st.number_input('flipper lenght', min_value=None, max_value=None, label_visibility="visible")
mass = st.number_input('body mass', min_value=None, max_value=None, label_visibility="visible")
culmen_l = st.number_input('culmen lenght', min_value=None, max_value=None, label_visibility="visible")
culmen_d = st.number_input('culmen depth', min_value=None, max_value=None, label_visibility="visible")

penguins = ['Adelie','Chinstrap','Gentoo']

  
# Create a button to trigger the model
if st.button('which penguiiiin?'):

    # Make a prediction using the model
    with torch.no_grad():
        output = model.forward(torch.tensor([culmen_l, culmen_d,flipper, mass]))
        
        for idx, outputy in enumerate(output):
            prediction = np.argmax(outputy).item()


        
        
        # Display the prediction result
        if prediction == 0:
            st.audio("adelie_squabble.mp3")
            st.write('Your penguin is Adelie, this is how she sounds. It is lovely :)')
        elif prediction == 1:
            st.write('Your penguin is a Chinstrap Penguin, They are one of the most aggressive of all penguins. Do not mess with it')
            st.image(image2, caption='Look at it')   
        else:
            st.write('The pengin species is Gentoo it will give a rock to its spouse as a token of appreciation, you should do the same')