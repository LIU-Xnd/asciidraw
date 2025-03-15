from cartoonize import cartoonize
import numpy as np
from PIL import Image
from argparse import ArgumentParser


def asciidraw(
    IMG_PATH: str,
    CHAR_POOL: list[str] | None = None,
    RESOLUTION: tuple[int, int] | None = None,
) -> str:
    """Read an image and draw it using given characters.

    Args:
        IMG_PATH (str): path to image.
        CHAR_POOL (list[str]): list of characters used to plot the image. Defaults to ['#', '.'].
        RESOLUTION (tuple[int, int]): resolution of resulting drawing. Defaults to (20, 60).

    Return:
        str: formatted string plotting the input image, seperated by newlines (including the last line).
    """
    if CHAR_POOL is None:
        CHAR_POOL: list[str] = ["#", "."]
    if RESOLUTION is None:
        RESOLUTION: tuple[int, int] = (20, 60)
    img_out: Image = cartoonize(
        IMG_PATH=IMG_PATH,
        N_COLORS=len(CHAR_POOL),
        verbose=False,
    )
    img_arr: np.ndarray = np.array(img_out)
    n_pixels: int = img_arr.shape[0] * img_arr.shape[1]
    n_channels: int = 1
    dim_channels: tuple[int] = img_arr.shape[2:]
    if len(dim_channels) > 0:
        n_channels = dim_channels[0]
    sample_matrix: np.ndarray = img_arr.reshape(n_pixels, n_channels)
    unique_colors: np.ndarray = np.unique(sample_matrix, axis=0)
    unique_colors = unique_colors[
        np.argsort(np.sum(unique_colors, axis=1))[::-1], :
    ]  # from bright to dark
    result_matrix: np.ndarray = np.empty(
        shape=(n_pixels,),
        dtype=str,
    )
    for i_color, vec_color in enumerate(unique_colors):
        result_matrix[np.all((sample_matrix == vec_color), axis=1)] = CHAR_POOL[i_color]
    result_matrix = result_matrix.reshape(img_arr.shape[0], img_arr.shape[1])
    ilocs_rows = np.round(
        np.linspace(0, result_matrix.shape[0] - 1, num=RESOLUTION[0], dtype=float)
    ).astype(int)
    ilocs_cols = np.round(
        np.linspace(0, result_matrix.shape[1] - 1, num=RESOLUTION[1], dtype=float)
    ).astype(int)
    ilocs_rows = np.minimum(result_matrix.shape[0] - 1, ilocs_rows)
    ilocs_cols = np.minimum(result_matrix.shape[1] - 1, ilocs_cols)
    result_matrix_subsampling = result_matrix[ilocs_rows, :][:, ilocs_cols]
    result_string = ""
    for i_row in range(result_matrix_subsampling.shape[0]):
        for i_col in range(result_matrix_subsampling.shape[1]):
            result_string += result_matrix_subsampling[i_row, i_col]
        result_string += "\n"
    return result_string


def main():
    arg_parser = ArgumentParser(
        description="""Read an image and draw it using given characters.

Args:
    IMG_PATH (str): path to image.
    CHAR_POOL (list[str]): list of characters used to plot the image. Defaults to ['#', '.'].
    RESOLUTION (tuple[int, int]): resolution of resulting drawing. Defaults to (60, 20).

Return:
    str: formatted string plotting the input image, seperated by newlines (including the last line).
    """,
    )

    arg_parser.add_argument("inputfile", help="path of input image")
    arg_parser.add_argument(
        "-o",
        "--output",
        help="path of output text file. If not given, print to stdout.",
        default=None,
    )
    arg_parser.add_argument(
        "-c",
        "--charpool",
        help="pool of characters used to draw. E.g., -c '#.*' (Use single-quote to escape certain characters) Defaults to '#.'",
        type=str,
        default="#.",
    )
    arg_parser.add_argument(
        "-r",
        "--resolution",
        help="rows,cols Resolution of resulting char plot. Defaults to 20,60",
        type=str,
        default="20,60",
    )

    args = arg_parser.parse_args()
    IMG_PATH: str = args.inputfile
    OUT_PATH: str | None = args.output
    CHAR_POOL: list[str] = list(args.charpool)
    RESOLUTION: tuple[int, int] = tuple(map(int, (args.resolution).split(",")))

    result_string: str = asciidraw(
        IMG_PATH=IMG_PATH,
        CHAR_POOL=CHAR_POOL,
        RESOLUTION=RESOLUTION,
    )

    if OUT_PATH is None:
        print(result_string)
        return

    with open(OUT_PATH, "w") as f:
        f.write(result_string)
    return


if __name__ == "__main__":
    main()
