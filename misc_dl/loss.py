import numpy as np


# focal loss - modified version of binary cross entropy, downweights easy examples and focuses on hard examples
# add modulating factor to the standard cross entropy loss
# FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2):
    epsilon = 1e-8
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    # calculate binary cross entropy
    cross_entropy = -y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)

    # calculate focal term 
    pt = np.where(y_true == 1, y_pred, 1 - y_pred)
    focal_term = alpha * (1 - pt) ** gamma

    loss = focal_term * cross_entropy

    return np.mean(loss)

# huber loss - less sensitive to outliers in data than mean squared error
# L(y, f(x)) = 0.5 * (y - f(x))^2 if |y - f(x)| <= delta
# L(y, f(x)) = delta * |y - f(x)| - 0.5 * delta^2 otherwise
def huber_loss(y_true, y_pred, delta=1):
    error = y_true - y_pred
    is_small_error = np.abs(error) <= delta
    small_error_loss = 0.5 * error ** 2
    large_error_loss = delta * np.abs(error) - 0.5 * delta ** 2

    loss = np.where(is_small_error, small_error_loss, large_error_loss)

    return np.mean(loss)