import gc
import numpy as np
import torch
from argparse import Namespace
from typing import Union, Tuple, List

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import get_data_batch, get_contrastive_data_batch
from LogisticRegression.model import LinearModel
from LogisticRegression.train_utils import fit_model as fit_linear_model,\
    calculate_loss as calculate_linear_loss,\
    calculate_accuracy as calculate_linear_accuracy


def calculate_loss(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the loss of the model on the given data.

    Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        loss: float, loss of the model
    '''
    # raise NotImplementedError('Calculate negative-log-likelihood loss here')
    loss = torch.nn.NLLLoss()(y_logits, y.long())
    return loss


def calculate_accuracy(
        y_logits: torch.Tensor, y: torch.Tensor
) -> float:
    '''
    Calculate the accuracy of the model on the given data.

    Args:
        Args:
        y_logits: torch.Tensor, softmax logits
        y: torch.Tensor, labels
    
    Returns:
        acc: float, accuracy of the model
    '''
    # raise NotImplementedError('Calculate accuracy here')
    acc = (y_logits.argmax(dim=1) == y).float().mean()
    return acc



def fit_contrastive_model(
        encoder: torch.nn.Module,
        X: torch.Tensor,
        y: torch.Tensor,
        num_iters: int = 1000,
        batch_size: int = 256,
        learning_rate: float = 1e-3
) -> None:
    '''
    Fit the contrastive model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - X: torch.Tensor, features
    - y: torch.Tensor, labels
    - num_iters: int, number of iterations for training
    - batch_size: int, batch size for training

    Returns:
    - losses: List[float], list of losses at each iteration
    '''
    # TODO: define the optimizer for the encoder only
    optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)

    # TODO: define the loss function
    criterion = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.PairwiseDistance(), margin=50.0)

    losses = []

    encoder.train()
    for i in range(1000):
        # raise NotImplementedError('Write the contrastive training loop here')
        optimizer.zero_grad()
        anchor, positive, negative = get_contrastive_data_batch(X.cpu().numpy(), y.cpu().numpy(), batch_size)
        anchor, positive, negative = torch.from_numpy(anchor).to(ptu.device) ,torch.from_numpy(positive).to(ptu.device) ,torch.from_numpy(negative).to(ptu.device)
        anchor_embeddings = encoder(anchor)
        positive_embeddings = encoder(positive)
        negative_embeddings = encoder(negative)
        loss = criterion(anchor_embeddings, positive_embeddings, negative_embeddings)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if i % 50 == 0:
            print(f'Iteration: {i}/{num_iters}, loss: {loss.item()}')
    
    return losses


def evaluate_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X: torch.Tensor,
        y: torch.Tensor,
        batch_size: int = 256,
        is_linear: bool = False
) -> Tuple[float, float]:
    '''
    Evaluate the model on the given data.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X: torch.Tensor, images
    - y: torch.Tensor, labels
    - batch_size: int, batch size for evaluation
    - is_linear: bool, whether the classifier is linear

    Returns:
    - loss: float, loss of the model
    - acc: float, accuracy of the model
    '''
    # raise NotImplementedError('Get the embeddings from the encoder and pass it to the classifier for evaluation')
    

    # HINT: use calculate_loss and calculate_accuracy functions for NN classifier and calculate_linear_loss and calculate_linear_accuracy functions for linear (softmax) classifier
    
    

    # return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)
    encoder.eval()
    classifier.eval()
    if is_linear:
       X = ptu.to_numpy(X)  
        
    y_preds = classifier(X)
    if is_linear:
        return calculate_linear_loss(y_preds, y), calculate_linear_accuracy(y_preds, y)
    else:
        return calculate_loss(y_preds, y), calculate_accuracy(y_preds, y)


def fit_model(
        encoder: torch.nn.Module,
        classifier: Union[LinearModel, torch.nn.Module],
        X_train: torch.Tensor,
        y_train: torch.Tensor,
        X_val: torch.Tensor,
        y_val: torch.Tensor,
        args: Namespace
) -> Tuple[List[float], List[float], List[float], List[float]]:
    '''
    Fit the model.

    Args:
    - encoder: torch.nn.Module, the encoder model
    - classifier: Union[LinearModel, torch.nn.Module], the classifier model
    - X_train: torch.Tensor, training images
    - y_train: torch.Tensor, training labels
    - X_val: torch.Tensor, validation images
    - y_val: torch.Tensor, validation labels
    - args: Namespace, arguments for training

    Returns:
    - train_losses: List[float], list of training losses
    - train_accs: List[float], list of training accuracies
    - val_losses: List[float], list of validation losses
    - val_accs: List[float], list of validation accuracies
    '''
    if args.mode == 'fine_tune_linear':
        # raise NotImplementedError('Get the embeddings from the encoder and use already implemeted training method in softmax regression')
        encoder.eval()
        encoded_representations = []
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            X_batch = ptu.to_numpy(encoder(X_batch))
            encoded_representations.append(X_batch)
           
        X_train = np.concatenate(encoded_representations, axis=0)  
        
        y_train = ptu.to_numpy(y_train).astype(int)
        y_val = ptu.to_numpy(y_val).astype(int)

        print('Encoding Completed')
        train_losses, train_accs, val_losses, val_accs = fit_linear_model(classifier, X_train, y_train, X_val, y_val, args.lr, args.batch_size, args.num_iters, args.l2_lambda, args.grad_norm_clip)
        return train_losses, train_accs, val_losses, val_accs
    else:
        # TODO: define the optimizer
        optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=args.lr)
        
        # raise NotImplementedError('Write the supervised training loop here')
        # return the losses and accuracies both on training and validation data
        train_losses = []
        train_accs = []
        val_losses = []
        val_accs = []

        encoder.train()
        classifier.train()

        for i in range(args.num_iters):
            optimizer.zero_grad()

            # Get a batch of training data
            X_batch, y_batch = get_data_batch(X_train, y_train, args.batch_size)

            # Forward pass
            embeddings = encoder(X_batch)
            y_preds = classifier(embeddings)

            # Calculate loss and accuracy
            train_loss, train_acc = calculate_loss(y_preds, y_batch), calculate_accuracy(y_preds, y_batch)

            # Backward pass
            train_loss.backward()
            optimizer.step()

            

            # Evaluate on validation data every `eval_interval` iterations
            if i % 50 == 0:
                # Append losses and accuracies
                train_losses.append(train_loss.item())
                train_accs.append(train_acc)
                val_loss, val_acc = evaluate_model(encoder, classifier, X_val, y_val, args.batch_size)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print(
                    f'Iteration {i}/{args.num_iters} - Train Loss: {train_loss:.4f} - Train Acc: {train_acc:.4f}'
                    f' - Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}'
                )

        return train_losses, train_accs, val_losses, val_accs
