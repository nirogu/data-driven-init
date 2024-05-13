"""
Data Driven Initialization for Neural Network Models.

Author
------
Nicolas Rojas
"""

import torch


def linear_classification_weights(X: torch.Tensor, method: str) -> torch.Tensor:
    """Initialize the weights of a linear layer, using the input data.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the linear classification.
    method : str
        Method to be used for initialization. Can be either "mean" or "median".

    Returns
    -------
    torch.Tensor
        Weights of the linear layer.
    """
    # sometimes, there are no elements for the given class. Return 0 if that is the case
    if len(X) == 0:
        return torch.zeros(1, X.shape[1])
    # weights can be mean or median of input data
    if method == "mean":
        weights = torch.mean(X, dim=0, keepdim=True)
    elif method == "median":
        weights = torch.median(X, dim=0, keepdim=True)[0]
    # normalize weights
    weights /= torch.norm(weights)

    return weights


def separe_class_projections(
    p0: torch.Tensor, p1: torch.Tensor, method: str
) -> torch.Tensor:
    """Find the value that best separates the projections of two different classes.

    Parameters
    ----------
    p0 : torch.Tensor
        Projections of the first class.
    p1 : torch.Tensor
        Projections of the second class.
    method : str
        Method to be used for separation. Can be either "mean" or "quadratic".

    Returns
    -------
    torch.Tensor
        Value that best separates the projections of the two classes.
    """
    # if classes dont overlap, return middle point of dividing space
    if torch.all(p1 >= p0.max()):
        separation = (p1.min() + p0.max()) / 2
    else:
        # case where classes overlap
        p0_overlap = p0 > p1.min()
        p1_overlap = p1 < p0.max()

        if method == "mean":
            # separation value is the weighted mean of the overlapping points
            n0 = p0_overlap.sum()
            n1 = p1_overlap.sum()
            sum0 = p0[p0_overlap].sum()
            sum1 = p1[p1_overlap].sum()
            separation = (n0 / n1 * sum1 + n1 / n0 * sum0) / (n1 + n0)

        elif method == "quadratic":
            # separation value is mid point of quadratic function
            # (points histogram represented as a parabola)
            q = torch.tensor([0, 0.05, 0.50, 0.95, 1])
            lq = len(q)
            pp0_overlap = p0[p0_overlap]
            pp1_overlap = p1[p1_overlap]
            pp1_vals = torch.quantile(pp1_overlap, q, dim=0, keepdim=True).reshape(
                -1, 1
            )
            pp0_vals = torch.quantile(pp0_overlap, 1 - q, dim=0, keepdim=True).reshape(
                -1, 1
            )
            A1, A0 = torch.ones(lq, 3), torch.ones(lq, 3)
            A1[0:, 0] = ((q / 100)) ** 2
            A1[0:, 1] = q / 100
            A0[0:, 0] = (1 - (q / 100)) ** 2
            A0[0:, 1] = 1 - (q / 100)
            coeff0 = torch.linalg.pinv(A0.T @ A0) @ (A0.T @ pp0_vals).squeeze()
            coeff1 = torch.linalg.pinv(A1.T @ A1) @ (A1.T @ pp1_vals).squeeze()
            a0, b0, c0 = coeff0[0], coeff0[1], coeff0[2]
            a1, b1, c1 = coeff1[0], coeff1[1], coeff1[2]
            a = a1 - a0
            b = b1 + 2 * a0 + b0
            c = c1 - a0 - c0
            i1 = (-b + (b**2 - 4 * a * c) ** 0.5) / (2 * a)
            i2 = (-b - (b**2 - 4 * a * c) ** 0.5) / (2 * a)
            separation = max(i1, i2)

    return separation


def linear_classification_bias(
    X0: torch.Tensor, X1: torch.Tensor, weights: torch.Tensor, method: str
) -> torch.Tensor:
    """Find the bias of a feed-forward classification layer, given its weights.

    Parameters
    ----------
    X0 : torch.Tensor
        Input data of the first class.
    X1 : torch.Tensor
        Input data of the second class.
    weights : torch.Tensor
        Weights of the linear layer.
    method : str
        Method to be used for bias calculation. Can be either "mean" or "quadratic".

    Returns
    -------
    torch.Tensor
        Bias of the linear layer.
    """
    # sometimes, there are no elements for the given class. Return 0 if that is the case
    if len(X0) == 0 or len(X1) == 0:
        return torch.tensor(0)

    # project observations over weights to get 1D vectors
    p0 = (X0 @ weights.T).squeeze()
    p1 = (X1 @ weights.T).squeeze()

    # find bias according to class projections
    bias = separe_class_projections(p0, p1, method)
    return bias


def linear_classification_output_layer(
    X: torch.Tensor, y: torch.Tensor, weights_method: str, bias_method: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize the output (linear) layer of a classification neural network.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the classification.
    y : torch.Tensor
        Labels of the input data.
    weights_method : str
        Method to be used for weights initialization. Can be either "mean" or "median".
    bias_method : str
        Method to be used for bias initialization. Can be either "mean" or "quadratic".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weights and bias of the output layer.
    """
    all_classes = torch.unique(y)
    all_weights = []
    all_biases = []
    binary = len(all_classes) == 2

    # when there are only 2 classes, it is only necessary to consider class 1
    if binary:
        all_classes = all_classes[1:]

    for yi in all_classes:
        # for each class, initialize weights and biases
        X0 = X[y != yi]
        X1 = X[y == yi]

        weights = linear_classification_weights(X1, weights_method)
        bias = linear_classification_bias(X0, X1, weights, bias_method)

        all_weights.append(weights)
        all_biases.append(bias)

    # transform lists of tensors to tensors
    if binary:
        all_weights = all_weights[0]
    else:
        all_weights = torch.cat(all_weights)
    all_biases = torch.tensor(all_biases)

    return all_weights, all_biases


def select_support_vectors(
    X: torch.Tensor, y: torch.Tensor, distances, num_neurons: int, num_classes: int
) -> tuple[torch.Tensor, list[int]]:
    """
    Find the support vectors for a set of vectors.

    Support vectors are defined as the elements of a certain class
    that are closer to elements from a different class.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the classification.
    y : torch.Tensor
        Labels of the input data.
    distances : torch.Tensor
        Pairwise distances between input data.
    num_neurons : int
        Number of neurons to be used in the layer.
    num_classes : int
        Number of classes in the input data.

    Returns
    -------
    tuple[torch.Tensor, list[int]]
        Support vectors and their corresponding classes.
    """
    # get how many neurons should belong to each class
    quotient, remainder = divmod(num_neurons, num_classes)
    neurons_classes = [quotient] * num_classes
    for idx in range(remainder):
        neurons_classes[idx] += 1

    vectors = []
    classes = []
    # iterate over each class with its number of corresponding neurons
    for label, num_vectors in enumerate(neurons_classes):
        # get elements belonging to desired class
        label_indices = y == label
        X1 = X[label_indices]
        # obtain distances between elements belonging to class and elements
        label_outside_distances = distances[label_indices, :][:, ~label_indices]
        # get mean distance for each element in class and order from lesser to greater
        label_outside_distances = label_outside_distances.mean(dim=1)
        min_elements = label_outside_distances.argsort()[:num_vectors]
        # vectors closer to elements from other classes are support vectors
        vectors.append(X1[min_elements])
        classes.extend([label] * num_vectors)
    vectors = torch.cat(vectors)
    # return vectors with their corresponding classes
    return vectors, classes


def linear_classification_hidden_layer(
    X: torch.Tensor,
    y: torch.Tensor,
    num_neurons: int,
    num_classes: int,
    weights_method: str,
    bias_method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize a hidden linear layer of a classification neural network.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the classification.
    y : torch.Tensor
        Labels of the input data.
    num_neurons : int
        Number of neurons to be used in the layer.
    num_classes : int
        Number of classes in the input data.
    weights_method : str
        Method to be used for weights initialization. Can be either "mean" or "median".
    bias_method : str
        Method to be used for bias initialization. Can be either "mean" or "quadratic".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weights and bias of the hidden layer.
    """
    # get pairwise distances of input observations
    distances = torch.cdist(X, X)
    # get support vectors for the layer
    characteristic_vectors, cv_classes = select_support_vectors(
        X, y, distances, num_neurons, num_classes
    )

    # get distances between support vectors and every element
    distances = torch.cdist(characteristic_vectors, X)
    # neighborhoods are Voronoi regions of support vectors
    neighborhoods = distances.argmin(dim=0)

    layer_weights = []
    layer_bias = []
    for neighborhood in range(num_neurons):
        # get points belonging to neighborhood
        close_points = neighborhoods == neighborhood
        close_X = X[close_points]
        close_y = y[close_points]

        # get elements belonging to same class as support vector
        k = cv_classes[neighborhood]
        X0 = close_X[close_y != k]
        X1 = close_X[close_y == k]

        # get weights and biases of layer using elements in same class as support vector
        weights = linear_classification_weights(
            X1 - X1.mean(dim=1, keepdim=True), weights_method
        )
        bias = linear_classification_bias(X0, X1, weights, bias_method)

        layer_weights.append(weights)
        layer_bias.append(bias)

    layer_weights = torch.cat(layer_weights)
    layer_bias = torch.tensor(layer_bias)
    # return weights and biases of each layer as single tensor
    return layer_weights, layer_bias


def rnn_bias(
    X0: torch.Tensor,
    X1: torch.Tensor,
    weights: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    h_weights: torch.Tensor,
    method: str,
) -> torch.Tensor:
    """Find the bias of a recurrent classification layer, given all of its weights.

    Parameters
    ----------
    X0 : torch.Tensor
        Input data of the first class.
    X1 : torch.Tensor
        Input data of the second class.
    weights : torch.Tensor
        Weights of the linear layer.
    h0 : torch.Tensor
        Hidden state of the first class.
    h1 : torch.Tensor
        Hidden state of the second class.
    h_weights : torch.Tensor
        Weights of the hidden state.
    method : str
        Method to be used for bias calculation. Can be either "mean" or "quadratic".

    Returns
    -------
    torch.Tensor
        Bias of the recurrent layer.
    """
    # sometimes, there are no elements for the given class. Return 0 if that is the case
    if len(X0) == 0 or len(X1) == 0:
        return torch.tensor(0)

    # project observations over weights to get 1D vectors
    p0 = (X0 @ weights.T + h0 @ h_weights.T).squeeze()
    p1 = (X1 @ weights.T + h1 @ h_weights.T).squeeze()

    # find bias according to class projections
    bias = separe_class_projections(p0, p1, method)
    return bias


def rnn_hidden_layer(
    X: torch.Tensor,
    h: torch.Tensor,
    y: torch.Tensor,
    num_neurons: int,
    num_classes: int,
    weights_method: str,
    bias_method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Initialize a recurrent layer of a classification neural network.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the classification.
    h : torch.Tensor
        Hidden state of the input data.
    y : torch.Tensor
        Labels of the input data.
    num_neurons : int
        Number of neurons to be used in the layer.
    num_classes : int
        Number of classes in the input data.
    weights_method : str
        Method to be used for weights initialization. Can be either "mean" or "median".
    bias_method : str
        Method to be used for bias initialization. Can be either "mean" or "quadratic".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Weights, hidden state weights, and bias of the recurrent layer.
    """
    # get pairwise distances of input observations
    distances = torch.cdist(X, X)
    # get support vectors for the layer
    characteristic_vectors, cv_classes = select_support_vectors(
        X, y, distances, num_neurons, num_classes
    )

    # get distances between support vectors and every element
    distances = torch.cdist(characteristic_vectors, X)
    # neighborhoods are Voronoi regions of support vectors
    neighborhoods = distances.argmin(dim=0)

    layer_weights = []
    layer_h_weights = []
    layer_bias = []
    for neighborhood in range(num_neurons):
        # get points from X, y, and h belonging to neighborhood
        close_points = neighborhoods == neighborhood
        close_X = X[close_points]
        close_h = h[close_points]
        close_y = y[close_points]

        # get elements belonging to same class as support vector
        k = cv_classes[neighborhood]
        X0 = close_X[close_y != k]
        X1 = close_X[close_y == k]
        h0 = close_h[close_y != k]
        h1 = close_h[close_y == k]

        # get weights and biases of layer using elements in same class as support vector
        weights = linear_classification_weights(
            X1 - X1.mean(dim=1, keepdim=True), weights_method
        )
        h_weights = linear_classification_weights(
            h1 - h1.mean(dim=1, keepdim=True), weights_method
        )
        bias = rnn_bias(X0, X1, weights, h0, h1, h_weights, bias_method)

        layer_weights.append(weights)
        layer_h_weights.append(h_weights)
        layer_bias.append(bias)

    layer_weights = torch.cat(layer_weights)
    layer_h_weights = torch.cat(layer_h_weights)
    layer_bias = torch.tensor(layer_bias)
    # return weights and biases of each layer as single tensor
    return layer_weights, layer_h_weights, layer_bias


def conv_bias(
    X: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    kernel_classes: list[int],
    method: str,
) -> torch.Tensor:
    """Find the bias of a convolutional classification layer, given its kernel.

    Parameters
    ----------
    X : torch.Tensor
        Input data of the classification.
    y : torch.Tensor
        Labels of the input data.
    kernel : torch.Tensor
        Kernel of the convolutional layer.
    kernel_classes : list[int]
        Classes of the kernel.
    method : str
        Method to be used for bias calculation. Can be either "mean" or "quadratic".

    Returns
    -------
    torch.Tensor
        Bias of the convolutional layer.
    """
    # relevant values are kernel dimensions
    out_dims, in_dims, rows, columns = kernel.shape
    layer_bias = []
    # find the bias of each individual kernel layer
    for iteration in range(out_dims):
        # find the points corresponding to the same class as the kernel layer
        iteration_class = kernel_classes[iteration]
        X0 = X[y == iteration_class]
        X1 = X[y != iteration_class]
        iteration_kernel = kernel[iteration : iteration + 1]

        # get convolution of observations and kernel layer to get 1D vectors
        p0 = torch.nn.functional.conv2d(X0, iteration_kernel).flatten()
        p1 = torch.nn.functional.conv2d(X1, iteration_kernel).flatten()

        # find bias according to class projections
        bias = separe_class_projections(p0, p1, method)
        layer_bias.append(bias)

    # return inverse of kernel to center class division
    layer_bias = -torch.tensor(layer_bias)
    return layer_bias


def init_weights_conv(
    X: torch.Tensor,
    y: torch.Tensor,
    out_channels: int,
    kernel_row: int,
    kernel_col: int,
    num_classes: int,
    bias_method: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Initialize a convolutional layer of a classification neural network.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the classification.
    y : torch.Tensor
        Labels of the input data.
    out_channels : int
        Number of output channels of the convolutional layer.
    kernel_row : int
        Number of rows of the kernel.
    kernel_col : int
        Number of columns of the kernel.
    num_classes : int
        Number of classes in the input data.
    bias_method : str
        Method to be used for bias initialization. Can be either "mean" or "quadratic".

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weights and bias of the convolutional layer.
    """
    # get dimensions of input data
    num_data, num_channels, rows, columns = X.shape
    fragments = []
    fragments_classes = []
    # get number of row and column partitions over data
    row_iters = rows // kernel_row
    col_iters = columns // kernel_col

    for unique_class in range(num_classes):
        # get median element of each class to represent it
        X_class = X[y == unique_class]
        X_class = X.median(dim=0).values
        # iterate over groups of rows and columns in median object
        for row_idx in range(row_iters):
            for col_idx in range(col_iters):
                # flatten fragment to use as kernel candidate
                fragment = X_class[
                    :,
                    kernel_row * row_idx : kernel_row * (row_idx + 1),
                    kernel_col * col_idx : kernel_col * (col_idx + 1),
                ]
                fragment = fragment.flatten()
                if fragment.norm() <= 0.01:
                    continue
                fragments.append(fragment)
                fragments_classes.append(unique_class)

    fragments = torch.stack(fragments)
    # normalize fragments
    fragments -= fragments.mean(1, keepdim=True)
    # if there are not enough fragments, use existing fragments plus a
    # random noise as extra fragments until fragment number is enough
    # (at least equal to the number of output channels)
    while fragments.shape[0] < out_channels:
        difference = out_channels - fragments.shape[0]
        difference = fragments[: min(difference, fragments.shape[0])]
        difference += torch.normal(0, 0.1, size=difference.shape)
        fragments = torch.cat((fragments, difference), 0)

    # get pairwise correlations of kernel candidates
    correlations = torch.zeros(len(fragments), len(fragments))
    for idx1 in range(len(fragments)):
        for idx2 in range(idx1, len(fragments)):
            correlations[idx1, idx2] = abs(
                torch.nn.functional.cosine_similarity(
                    fragments[idx1], fragments[idx2], dim=0
                )
            )
            correlations[idx2, idx1] = correlations[idx1, idx2]

    fragments_classes = torch.tensor(fragments_classes)
    # find optimal kernels from kernel candidates, using support vectors method
    characteristic_vectors, kernel_classes = select_support_vectors(
        fragments, fragments_classes, correlations, out_channels, num_classes
    )
    current_num_weights = characteristic_vectors.shape[0]
    # un-flatten selected kernels
    characteristic_vectors = characteristic_vectors.reshape(
        (current_num_weights, num_channels, kernel_row, kernel_col)
    )
    # normalize selected kernels
    for weight in range(current_num_weights):
        for channel in range(num_channels):
            characteristic_vectors[weight, channel, :, :] /= torch.linalg.matrix_norm(
                characteristic_vectors[weight, channel, :, :]
            )

    # if there are not enough kernels, use existing kernels plus a
    # random noise as extra kernels until kernel number is enough
    # (at least equal to the number of output channels)
    while current_num_weights < out_channels:
        difference = out_channels - current_num_weights
        difference = characteristic_vectors[: min(difference, current_num_weights)]
        difference += torch.normal(0, 0.1, size=difference.shape)
        characteristic_vectors = torch.cat((characteristic_vectors, difference), 0)
        current_num_weights = characteristic_vectors.shape[0]

    # find layer biases using selected kernels
    layer_bias = conv_bias(X, y, characteristic_vectors, kernel_classes, bias_method)
    return characteristic_vectors, layer_bias


def init_weights_classification(
    model: torch.nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    weights_method: str = "mean",
    bias_method: str = "mean",
) -> torch.nn.Module:
    """Initialize a classification neural network.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to be initialized.
    X : torch.Tensor
        Input data to the classification.
    y : torch.Tensor
        Labels of the input data.
    weights_method : str, optional
        Method to be used for weights initialization. Can be either "mean" or "median".
        Default is "mean".
    bias_method : str, optional
        Method to be used for bias initialization. Can be either "mean" or "quadratic".
        Default is "mean".

    Returns
    -------
    torch.nn.Module
        Initialized neural network model.

    Raises
    ------
    ValueError
        If X and y have different number of samples.
        If input data has less than 2 classes.

    Example
    -------
    >>> import torch
    >>> from torch import nn
    >>> num_classes = 3
    >>> num_data = 20
    >>> num_features = 10
    >>> hidden_size = 5
    >>> X = torch.rand(num_data, num_features)
    >>> y = torch.randint(0, num_classes, (num_data,))
    >>> model = nn.Sequential(
    >>>     nn.Linear(num_features, hidden_size),
    >>>     nn.ReLU(),
    >>>     nn.Linear(hidden_size, num_classes)
    >>> )
    >>> model = init_weights_classification(model, X, y)
    """
    # Throw error when X and y have different number of samples
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    num_classes = len(torch.unique(y))
    # There must be at least two classes. Else, throw error
    if num_classes < 2:
        raise ValueError("Input data must present at least 2 classes")

    # Model is a single linear layer
    if isinstance(model, torch.nn.Linear):
        # find weights and biases of single layer
        weights, bias = linear_classification_output_layer(
            X, y, weights_method, bias_method
        )
        model.weight = torch.nn.Parameter(weights)
        model.bias = torch.nn.Parameter(bias)
        return model

    # Model is a sequential model with at least one linear layer (output)
    num_layers = len(model) - 1

    for layer in range(num_layers):
        if isinstance(model[layer], torch.nn.Linear):
            # layer is linear layer
            num_neurons = model[layer].out_features
            layer_weights, layer_bias = linear_classification_hidden_layer(
                X, y, num_neurons, num_classes, weights_method, bias_method
            )
            model[layer].weight = torch.nn.Parameter(layer_weights)
            model[layer].bias = torch.nn.Parameter(layer_bias)
        elif isinstance(model[layer], torch.nn.Conv2d):
            # layer is 2D convolutional layer
            kernel_row, kernel_col = model[layer].kernel_size
            out_channels = model[layer].out_channels
            layer_weights, layer_bias = init_weights_conv(
                X, y, out_channels, kernel_row, kernel_col, num_classes, bias_method
            )
            model[layer].weight = torch.nn.Parameter(layer_weights)
            model[layer].bias = torch.nn.Parameter(layer_bias)
        elif isinstance(model[layer], torch.nn.RNN):
            # layer is (stack of) recurrent layer. This kind of
            # layer requires an special treatment due to the way
            # it is represented in pytorch: not as a group of
            # layers but as a single object with multiple weights
            # and biases
            num_rnn_layers = model[layer].num_layers
            num_neurons = model[layer].hidden_size
            activation = (
                torch.nn.functional.tanh
                if model[layer].nonlinearity == "tanh"
                else torch.nn.functional.relu
            )
            # get last element in sequence
            layer_X = X[:, -1, :].detach().clone()
            for layer_idx in range(num_rnn_layers):
                # initialize the x-weights of each recurrent layer in stack
                layer_weights, layer_bias = linear_classification_hidden_layer(
                    layer_X, y, num_neurons, num_classes, weights_method, bias_method
                )
                setattr(
                    model[layer],
                    f"weight_ih_l{layer_idx}",
                    torch.nn.Parameter(layer_weights),
                )
                setattr(
                    model[layer],
                    f"bias_ih_l{layer_idx}",
                    torch.nn.Parameter(layer_bias),
                )
                # propagate layer_x through recurrent stack
                layer_X = activation(layer_X @ layer_weights.T + layer_bias)
            # obtain final state of h for each recurrent layer in stack to use as h0
            _, h0 = model[layer](X)
            # get last element in sequence
            layer_X = X[:, -1, :].detach().clone()
            for layer_idx in range(num_rnn_layers):
                # initialize the x-weights and h-weights of each recurrent layer in stack
                layer_weights, layer_h_weights, layer_bias = rnn_hidden_layer(
                    layer_X,
                    h0[layer_idx],
                    y,
                    num_neurons,
                    num_classes,
                    weights_method,
                    bias_method,
                )
                setattr(
                    model[layer],
                    f"weight_ih_l{layer_idx}",
                    torch.nn.Parameter(layer_weights),
                )
                setattr(
                    model[layer],
                    f"bias_ih_l{layer_idx}",
                    torch.nn.Parameter(layer_bias),
                )
                setattr(
                    model[layer],
                    f"weight_hh_l{layer_idx}",
                    torch.nn.Parameter(layer_h_weights),
                )
                setattr(
                    model[layer],
                    f"bias_hh_l{layer_idx}",
                    torch.nn.Parameter(torch.zeros_like(layer_bias)),
                )
                # propagate layer_x through recurrent stack
                layer_X = activation(
                    layer_X @ layer_weights.T
                    + h0[layer_idx] @ layer_h_weights.T
                    + layer_bias
                )
        # propagate X no matter the layers type
        X = model[layer](X)

    # Last layer (linear output layer)
    layer_weights, layer_bias = linear_classification_output_layer(
        X, y, weights_method, bias_method
    )
    model[num_layers].weight = torch.nn.Parameter(layer_weights)
    model[num_layers].bias = torch.nn.Parameter(layer_bias)

    return model


def linear_regression(
    X: torch.Tensor, y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit a single linear regression over input data.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the regression.
    y : torch.Tensor
        Target values of the input data.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weights and bias of the linear regression.
    """
    # expand X with a column of ones to find bias along with weights
    ones = torch.ones(X.shape[0], 1)
    X = torch.cat((ones, X), dim=1)
    # find parameters by multiplying pseudoinverse of X with y
    weights = torch.linalg.pinv(X) @ y
    # bias is first column of parameters and weights are the remaining columns
    bias = weights[0]
    weights = torch.unsqueeze(weights[1:], dim=0)
    # return weights and biases
    return weights, bias


def piecewise_linear_regression(
    X: torch.Tensor, y: torch.Tensor, num_pieces: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit multiple linear regressions over different sections of input data.

    Parameters
    ----------
    X : torch.Tensor
        Input data to the regression.
    y : torch.Tensor
        Target values of the input data.
    num_pieces : int
        Number of segments to divide the input data.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Weights and bias of the piecewise linear regression.
    """
    # order data according to X dimensions values
    ordered_idx = torch.argsort(X, dim=0)[:, 0]
    X = X[ordered_idx]
    y = y[ordered_idx]
    # find the size of a segment
    piece_length = len(y) // num_pieces
    all_weights = []
    all_biases = []

    # iterate over every segment
    for piece in range(num_pieces):
        # get data belonging to segment
        piece_idx = range(piece_length * piece, piece_length * (piece + 1))
        partial_X = X[piece_idx]
        partial_y = y[piece_idx]
        # fit linear regression over segment to obtain partial weights and biases
        weights, bias = linear_regression(partial_X, partial_y)
        all_weights.append(weights)
        all_biases.append(bias)
    # merge all weights and biases into individual tensors
    all_weights = torch.cat(all_weights, dim=0)
    all_biases = torch.tensor(all_biases)
    # return results
    return all_weights, all_biases


def init_weights_regression(
    model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor
) -> torch.nn.Module:
    """Initialize a regression neural network.

    Parameters
    ----------
    model : torch.nn.Module
        Neural network model to be initialized.
    X : torch.Tensor
        Input data to the regression.
    y : torch.Tensor
        Target values of the input data.

    Returns
    -------
    torch.nn.Module
        Initialized neural network model.

    Raises
    ------
    ValueError
        If X and y have different number of samples.

    Example
    -------
    >>> import torch
    >>> from torch import nn
    >>> num_data = 20
    >>> num_features = 10
    >>> hidden_size = 5
    >>> X = torch.rand(num_data, num_features)
    >>> y = torch.rand(num_data)
    >>> model = nn.Sequential(
    >>>     nn.Linear(num_features, hidden_size),
    >>>     nn.ReLU(),
    >>>     nn.Linear(hidden_size, 1)
    >>> )
    >>> model = init_weights_regression(model, X, y)
    """
    # Throw error when X and y have different number of samples
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of samples")

    # Model is a single linear layer
    if isinstance(model, torch.nn.Linear):
        # fit singular linear regression and set models parameters
        weights, bias = linear_regression(X, y)
        model.weight = torch.nn.Parameter(weights)
        model.bias = torch.nn.Parameter(bias)
        return model

    # Model is a sequential model with at least one linear layer (output)
    num_layers = len(model) - 1
    for layer in range(num_layers):
        if isinstance(model[layer], torch.nn.Linear):
            # layer is linear layer
            layer_weights, layer_bias = piecewise_linear_regression(
                X, y, num_pieces=model[layer].out_features
            )
            model[layer].weight = torch.nn.Parameter(layer_weights)
            model[layer].bias = torch.nn.Parameter(layer_bias)
        # propagate X no matter the layers type
        X = model[layer](X)

    # Last layer (linear output layer)
    layer_weights, layer_bias = linear_regression(X, y)
    model[num_layers].weight = torch.nn.Parameter(layer_weights)
    model[num_layers].bias = torch.nn.Parameter(layer_bias)

    return model
