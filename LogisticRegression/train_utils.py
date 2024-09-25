import numpy as np
from typing import Tuple

from LogisticRegression.model import LinearModel
from utils import get_data_batch


def calculate_loss(
        model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
    
    Returns:
        loss: float, loss of the model
    '''
    y_preds = model(X).squeeze()
    epsilon = 1e-5
    y_preds = np.clip(y_preds, epsilon, 1 - epsilon) # clip predictions to avoid log(0)
    if is_binary:
        loss = -np.mean(y * np.log(y_preds) + (1 - y) * np.log(1 - y_preds)) # binary cross-entropy loss
    else:
        # raise NotImplementedError('Calculate cross-entropy loss here')
        loss = -np.mean(np.log(y_preds))
    return loss


def calculate_accuracy(
        model: LinearModel, X: np.ndarray, y: np.ndarray, is_binary: bool = False
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    y_preds = model(X).squeeze()
    if is_binary:
        acc = np.mean((y_preds > 0.5) == y) # binary classification accuracy
    else:
        # raise NotImplementedError('Calculate accuracy for multi-class classification here')
        acc = np.mean(np.argmax(y_preds, axis=1) == y) # multi-class classification accuracy
    return acc


def evaluate_model(
        model: LinearModel, X: np.ndarray, y: np.ndarray,
        batch_size: int, is_binary: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data and return the loss and accuracy.

    Args:
        model: LinearModel, the model to be evaluated
        X: np.ndarray, features
        y: np.ndarray, labels
        batch_size: int, batch size for evaluation
    
    Returns:
        loss: float, loss of the model
        acc: float, accuracy of the model
    '''
    # get predicitions
    # raise NotImplementedError(
    #     'Get predictions in batches here (otherwise memory error for large datasets)')

    # calculate loss and accuracy
    loss = calculate_loss(model, X, y, is_binary)
    acc = calculate_accuracy(model, X, y, is_binary)
    
    return loss, acc


def fit_model(
        model: LinearModel, X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray, num_iters: int,
        lr: float, batch_size: int, l2_lambda: float,
        grad_norm_clip: float, is_binary: bool = False
) -> Tuple[list, list, list, list]:
    '''
    Fit the model on the given training data and return the training and validation
    losses and accuracies.

    Args:
        model: LinearModel, the model to be trained
        X_train: np.ndarray, features for training
        y_train: np.ndarray, labels for training
        X_val: np.ndarray, features for validation
        y_val: np.ndarray, labels for validation
        num_iters: int, number of iterations for training
        lr: float, learning rate for training
        batch_size: int, batch size for training
        l2_lambda: float, L2 regularization for training
        grad_norm_clip: float, clip value for gradient norm
        is_binary: bool, if True, use binary classification
    
    Returns:
        train_losses: list, training losses
        train_accs: list, training accuracies
        val_losses: list, validation losses
        val_accs: list, validation accuracies
    '''
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    for i in range(int(num_iters) + 1):
        # get batch
        X_batch, y_batch = get_data_batch(X_train, y_train, batch_size)
        
        # get predicitions
        y_preds = model(X_batch).squeeze()
        
        # calculate loss
        loss = calculate_loss(model, X_batch, y_batch, is_binary)
        
        # calculate accuracy
        acc = calculate_accuracy(model, X_batch, y_batch, is_binary)
        
        # calculate gradient
        if is_binary:
            grad_W = ((y_preds - y_batch) @ X_batch).reshape(-1, 1)
            grad_b = np.mean(y_preds - y_batch)
        else:
            # grad_W = TODO
            # grad_b = TODO  
            y_batch = np.eye(10)[y_batch]
            grad_W = (X_batch.T @ (y_preds - y_batch)) / len(X_batch)
            grad_b = np.mean(y_preds - y_batch, axis=0)
        

        # # regularization
        grad_W += l2_lambda * model.W
        grad_b += l2_lambda * model.b
        
        # # clip gradient norm
        # raise NotImplementedError('Clip gradient norm here')
        # clip gradient norm
        grad_norm = np.linalg.norm(grad_W) + np.linalg.norm(grad_b)
        if grad_norm > grad_norm_clip:
            grad_W = grad_W * (grad_norm_clip / grad_norm)
            grad_b = grad_b * (grad_norm_clip / grad_norm)
        
        # # update model
        # raise NotImplementedError('Update model here (perform SGD)')
        # update model
        model.W -= lr * grad_W
        model.b -= lr * grad_b
        

        if i % 100 == 0:
            # append loss
            train_losses.append(loss)
            # append accuracy
            train_accs.append(acc)

            # evaluate model
            val_loss, val_acc = evaluate_model(
                model, X_val, y_val, batch_size, is_binary)
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            print(
                f'Iter {i}/{num_iters} - Train Loss: {loss:.4f} - Train Acc: {acc:.4f}'
                f' - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}'
            )
            
            # TODO: early stopping here if required
            # early stopping
            if len(val_accs) >= 10 and val_accs[-1] < max(val_accs[-10:]):
                print("Early stopping triggered.")
                break
    
    return train_losses, train_accs, val_losses, val_accs
