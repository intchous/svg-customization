import torch


def procrustes_distance(src_points, tgt_points):
    src_centered = src_points - src_points.mean(dim=0)
    tgt_centered = tgt_points - tgt_points.mean(dim=0)
    epsilon = 1e-8

    regularization = torch.eye(src_centered.size(1)).to(
        src_centered.device) * epsilon
    src_centered_t_tgt_centered = torch.matmul(
        src_centered.t(), tgt_centered) + regularization

    u, _, v = torch.svd(src_centered_t_tgt_centered)

    rotation = torch.matmul(u, v.t())

    src_rotated = torch.matmul(src_centered, rotation)

    scale = torch.sum(tgt_centered * src_rotated) / \
        (torch.sum(src_rotated * src_rotated) + epsilon)
    src_transformed = scale * src_rotated

    # Calculate the Procrustes distance with a small positive value for numerical stability
    distance = torch.sqrt(
        torch.sum((tgt_centered - src_transformed) ** 2) + epsilon)
    return distance


def local_procrustes_loss(src_points, tgt_points, window_size=5, return_avg=False):
    assert src_points.shape == tgt_points.shape, "src_points and tgt_points should have the same shape"
    assert window_size > 0, "window_size should be greater than 0"

    n_points = src_points.size(0)
    distance_sum = 0.0
    window_size = min(window_size, n_points)

    for i in range(n_points):
        src_window = src_points[torch.arange(i, i + window_size) % n_points]
        tgt_window = tgt_points[torch.arange(i, i + window_size) % n_points]
        distance_sum += procrustes_distance(src_window, tgt_window)

    if return_avg:
        return (distance_sum / n_points)
    else:
        return distance_sum


def local_procrustes_loss_centered(src_points, tgt_points, window_size=5, return_avg=False):
    assert src_points.shape == tgt_points.shape, "src_points and tgt_points should have the same shape"
    assert window_size > 0, "window_size should be greater than 0"

    n_points = src_points.size(0)
    distance_sum = 0.0
    window_size = min(window_size, n_points)
    half_window = window_size // 2

    for i in range(n_points):
        src_window = src_points[torch.arange(
            i - half_window, i + half_window + 1) % n_points]
        tgt_window = tgt_points[torch.arange(
            i - half_window, i + half_window + 1) % n_points]
        distance_sum += procrustes_distance(src_window, tgt_window)

    if return_avg:
        return (distance_sum / n_points)
    else:
        return distance_sum


def local_procrustes_loss_centeredv2(src_points, tgt_points, window_size=5, return_avg=False):
    assert src_points.shape == tgt_points.shape, "src_points and tgt_points should have the same shape"
    assert window_size > 0, "window_size should be greater than 0"

    n_points = src_points.size(0)
    window_size = min(window_size, n_points)
    half_window = window_size // 2

    indices = torch.arange(n_points + window_size - 1) % n_points
    windows = indices.unfold(0, window_size, 1)

    src_windows = src_points[windows]
    tgt_windows = tgt_points[windows]

    # Calculate Procrustes distance for each window
    distances = torch.stack([procrustes_distance(
        src_windows[i], tgt_windows[i]) for i in range(n_points)])

    if return_avg:
        return distances.mean()
    else:
        return distances.sum()


# ----------------------------------------------------------------


def laplacian_smoothing_loss(points, num_neighbors=1, weight=1.0):
    n_points = points.size(0)

    avg_neighbors = torch.zeros_like(points)

    for i in range(-num_neighbors, num_neighbors + 1):
        if i == 0:
            continue
        index_shift = (torch.arange(n_points) - i) % n_points
        avg_neighbors += points[index_shift]
    avg_neighbors /= (2 * num_neighbors)

    diff = points - avg_neighbors

    epsilon = 1e-8
    smoothness = torch.sqrt(torch.sum(diff ** 2) / n_points + epsilon)

    return weight * smoothness


# ----------------------------------------
