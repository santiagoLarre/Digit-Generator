import streamlit as st
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from cvae_model import CVAE, latent_dim, num_classes

model = CVAE()
model.load_state_dict(torch.load("trained_cvae.pth", map_location=torch.device("cpu")))
model.eval()

st.title("Handwritten Digit Image Generator")

digit = st.selectbox("Choose a digit to generate (0-9)", list(range(10)))

if st.button("Generate Images"):
    cols = st.columns(5)
    for i in range(5):
        z = torch.randn(1, latent_dim)
        y = F.one_hot(torch.tensor([digit]), num_classes).float()
        with torch.no_grad():
            img = model.decode(z, y).view(28, 28).numpy()

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray")
        ax.axis("off")
        cols[i].pyplot(fig)
