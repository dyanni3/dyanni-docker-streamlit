import streamlit as st
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def generate_colors(ncolors):
    colors_dict = {i:np.random.random(3) for i in range(ncolors)}
    return(colors_dict)

def generate_random(nsamples=1000, ndims=4, nclusters=4):
    X = np.random.random((nsamples, ndims))
    km = KMeans(n_clusters=nclusters)
    km.fit(X)
    pca = PCA(n_components=2)
    data, labels = X, km.labels_
    projected = pca.fit_transform(data)
    return(data, projected, labels)

def plot(projected_data, labels):
    with plt.style.context('dark_background'):
        colors_dict = generate_colors(len(np.unique(labels)))
        colors = list(map(lambda x: colors_dict[x], labels))
        fig, ax = plt.subplots(1, 1, figsize=(8,8))
        ax.scatter(projected_data[:,0], projected_data[:,1], c=colors, alpha=.6)
        ax.grid(lw=.3)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        return(fig)

def regenerate_plot(nsamples=1000, ndims=4, nclusters=4):
    _, projected, labels = generate_random(nsamples=nsamples, ndims=ndims, nclusters=nclusters)
    fig = plot(projected, labels)
    st.write(fig)

if __name__=='__main__':
    st.title('working app')
    data, projected, labels = generate_random()
    fig = plot(projected, labels)
    st.write(fig)

    add_selectbox = st.sidebar.selectbox(
    "Regenerate this random pointless data?",
    (("Yes")))

    regenerate_plot()


