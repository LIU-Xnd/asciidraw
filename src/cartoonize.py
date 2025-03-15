from PIL import Image
import numpy as np
from sklearn.cluster import KMeans
from argparse import ArgumentParser

# IMG_PATH: str = "../data/lena.jpeg"
# OUT_PATH: str = "../data/lena.cartoon.jpeg"
# N_COLORS: int = 2


def cartoonize(
    IMG_PATH: str,
    N_COLORS: int = 2,
    verbose: bool = False,
):
    if verbose:
        print(f"Reading image {IMG_PATH}")

    image: Image = Image.open(IMG_PATH)

    # We convert it into an array
    img_arr: np.ndarray = np.array(image)
    # Rows by cols
    # They range from 0 to 255
    dim_channels = img_arr.shape[2:]
    n_channels: int = 1
    if len(dim_channels) == 0:
        n_channels = 1
        img_arr = img_arr[:, :, None]  # Expand its dims
    else:
        n_channels = dim_channels[0]
    # Number of pixels
    n_observations: int = img_arr.shape[0] * img_arr.shape[1]
    sample_matrix: np.ndarray = img_arr.reshape(
        n_observations,
        n_channels,
    )
    # Each color is essentially a 3-d vector of [R, G, B]

    # Perform cluster
    clusterer: KMeans = KMeans(
        n_clusters=N_COLORS,
    )
    if verbose:
        print("Performing color clustering ...")
    color_labels: np.ndarray = clusterer.fit_predict(
        sample_matrix,
    )
    color_centers: np.ndarray = np.round(clusterer.cluster_centers_).astype(int)
    color_centers = np.minimum(255, color_centers)

    for iloc_sample in range(sample_matrix.shape[0]):
        color_label: int = color_labels[iloc_sample]
        sample_matrix[iloc_sample, :] = color_centers[color_label]
    # Updated, cartoonized sample matrix
    # Need reshape it
    img_arr = sample_matrix.reshape(img_arr.shape)
    if n_channels == 1:
        img_arr = img_arr[:, :, 0]
    img_out: Image = Image.fromarray(
        img_arr,
        mode=image.mode,
    )
    return img_out


def main():
    arg_parser = ArgumentParser(
        description='Convert a "real-life" image into a "cartoon-ized" image'
        "based on color clustering."
    )

    arg_parser.add_argument("inputfile", help="Filepath of input image")
    arg_parser.add_argument(
        "-n",
        "--ncolors",
        help="Number of colors to use. Typically a small value like 2 or 3.",
        type=int,
        default=2,
    )
    arg_parser.add_argument(
        "-o", "--output", help="Filepath to put the processed image"
    )
    arg_parser.add_argument(
        "-v", "--verbose", help="Verbose while this program runs", action="store_true"
    )

    args = arg_parser.parse_args()
    IMG_PATH: str = args.inputfile
    OUT_PATH: str = args.output
    N_COLORS: int = args.ncolors
    verbose: bool = args.verbose
    img_out: Image = cartoonize(
        IMG_PATH=IMG_PATH,
        OUT_PATH=OUT_PATH,
        N_COLORS=N_COLORS,
        verbose=verbose,
    )
    img_out.save(OUT_PATH)
    if verbose:
        print(f"Done. Output to {OUT_PATH}")
    return


if __name__ == "__main__":
    main()
