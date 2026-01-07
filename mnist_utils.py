import matplotlib.pyplot as plt


def show_sample(axis, image, title):
    """
    Plot a single grayscale image on a given matplotlib axis.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Axis object on which to draw the image.
    image : torch.Tensor or numpy.ndarray
        Image data, optionally with a singleton channel dimension.
    title : str
        Title to display above the image.

    Returns
    -------
    None
    """
    axis.imshow(image.squeeze(), cmap="gray")
    axis.set_title(title)
    axis.axis("off")


def show_sample_grid(images, titles, nrows=5, ncols=5, figsize=(6, 6)):
    """
    Plot a grid of images with corresponding titles.

    Parameters
    ----------
    images : iterable
        Collection of image tensors or arrays to display.
    titles : iterable of str
        Titles corresponding to each image.
    n_rows : int, optional
        Number of rows in the image grid.
    n_cols : int, optional
        Number of columns in the image grid.
    figsize : tuple, optional
        Size of the matplotlib figure in inches.

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.flatten()

    for axis, image, title in zip(axes, images, titles):
        show_sample(axis, image, title)

    plt.tight_layout()
    plt.show()


def get_predictions(model, images):
    """
    Run a batch of images through a model and return the predicted class index
    for each image.

    Parameters
    ----------
    model : torch.nn.Module
        A trained model that outputs logits for each class.
    images : torch.Tensor
        A batch of input images with shape (N, C, H, W).

    Returns
    -------
    torch.Tensor
        A tensor of shape (N,) containing the predicted class indices.
    """
    logits = model(images)
    predictions = logits.argmax(dim=1)
    return predictions


def get_titles(labels, predictions):
    """
    Create display titles that pair true labels with predicted labels.

    Parameters
    ----------
    labels : iterable
        True labels for each sample.
    predictions : iterable
        Predicted labels for each sample.

    Returns
    -------
    list of str
        Strings formatted as "true_label (predicted_label)".
    """
    titles = [
        f"{label} ({prediction})"
        for label, prediction in zip(labels, predictions)
    ]
    return titles