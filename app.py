import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans
import numpy as np


def load_image(image_file) -> Image:
    """Loads the file as an Image object.

    Args:
        image_file:
            Input image file location.

    Returns:
        Loaded image object as `PIL.Image`.

    """
    img = Image.open(image_file)
    return img


def is_image_size_valid(image_file) -> bool:
    """Checks if the image is of size `256 x 256`.

    Args:
        image_file:
            Input image file location.

    Returns:
        `True` if the image is of size `256 x 256` else `False`.

    """
    if image_file is not None:
        if (
            Image.open(image_file).size[0] <= 256
            and Image.open(image_file).size[1] <= 256
        ):
            return True
        else:
            return False
    else:
        return False


def kmeans_image_compression(image: Image, cluster_size: int) -> Image:
    """Applies K-Means Compression to the input image.

    K-Means is computed for the `image` based on the `cluster_size` to find the k-centroids points.
    Post that it replaces the value of each surrounding color pixels with its centroid point.

    Args:
        image:
            Image as `PIL.Image` object.

        cluster_size:
            The number of clusters to form as well as the number of centroids to generate.

    Returns:
        Compressed image as `PIL.Image` object.

    """
    # Dimension of the original image
    rows = image.size[1]
    cols = image.size[0]

    # Flatten the image
    flattened_image = np.array(image).reshape(rows * cols, 3)

    # Implement k-means clustering to form k clusters
    kmeans = KMeans(n_clusters=cluster_size)
    kmeans.fit(flattened_image)

    # Replace each pixel value with its nearby centroid
    intermediate_form = kmeans.cluster_centers_[kmeans.labels_]
    compressed_image = np.clip(intermediate_form.astype("uint8"), 0, 255)

    # Reshape and return the image to original dimension
    return Image.fromarray(compressed_image.reshape(rows, cols, 3))


def main() -> None:
    """Main function with Streamlit UI."""
    st.title("Upload your image here...")

    st.sidebar.markdown(
        "<h1 style='text-align: center;'>Compression using K-Means</h1>",
        unsafe_allow_html=True,
    )
    with st.sidebar.expander("About the App"):
        st.write(
            """A Streamlit Application to compress Images using KMeans.\n\nThe app first finds k-centroid points 
            that represent its surrounding color combination using k-means and then replaces the value of each 
            surrounding color pixels with its centroid point."""
        )

    image_file = st.file_uploader("Upload Input Image", type=["jpg", "jpeg"])
    valid_size = is_image_size_valid(image_file)

    if image_file is not None and valid_size is not True:
        st.error("Expected image of size '256 x 256'")

    if image_file is not None and valid_size is True:
        col1, col2 = st.columns([0.5, 0.5])

        with col1:
            st.markdown(
                "<p style='text-align: center;'>Before</p>", unsafe_allow_html=True
            )
            st.image(load_image(image_file), caption="Input Preview", width=256)

        with col2:
            st.markdown(
                "<p style='text-align: center;'>After</p>", unsafe_allow_html=True
            )
            kmeans_value = st.sidebar.number_input(
                "Adjust the K-Means Value", min_value=0, max_value=512, value=5, step=1
            )
            with st.spinner("Processing..."):
                processed_image = kmeans_image_compression(
                    load_image(image_file), cluster_size=kmeans_value
                )
            st.image(processed_image, caption="Output Preview", width=256)

        st.success("Compression Successfully Done!")


if __name__ == "__main__":
    main()
