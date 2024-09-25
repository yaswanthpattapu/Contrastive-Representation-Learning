import gc
import torch
from argparse import Namespace

import ContrastiveRepresentation.pytorch_utils as ptu
from utils import *
from LogisticRegression.model import SoftmaxRegression as LinearClassifier
from ContrastiveRepresentation.model import Encoder, Classifier
from ContrastiveRepresentation.train_utils import fit_contrastive_model, fit_model


def main(args: Namespace):
    '''
    Main function to train and generate predictions in csv format

    Args:
    - args : Namespace : command line arguments
    '''
    # Set the seed
    torch.manual_seed(args.sr_no)

    # Get the training data
    X, y = get_data(args.train_data_path)
    X_train, y_train, X_val, y_val = train_test_split(X, y)
    
    # TODO: Convert the images and labels to torch tensors using pytorch utils (ptu)
    X_train = ptu.from_numpy(X_train)
    y_train = ptu.from_numpy(y_train)
    X_val = ptu.from_numpy(X_val)
    y_val = ptu.from_numpy(y_val)
    
    # Create the model
    encoder = Encoder(args.z_dim).to(ptu.device)
    num_classses = 10
    if args.mode == 'fine_tune_linear':
        # classifier = # TODO: Create the linear classifier model
        # classifier = Classifier()
        classifier = LinearClassifier(args.z_dim,num_classses) # TODO: Create the linear classifier model --10 is hard coded for CIFAR10
        print('classifer loaded')
    elif args.mode == 'fine_tune_nn':
        # classifier = # TODO: Create the neural network classifier model
        # classifier = Classifier()
        classifier = Classifier(args.z_dim,num_classses) # TODO: Create the neural network classifier model
        print('classifer loaded')
        classifier = classifier.to(ptu.device)
    
    if args.mode == 'cont_rep':
        # raise NotImplementedError('Implement the contrastive representation learning')
        #Fit the model
        fit_contrastive_model(encoder, X_train.cpu(), y_train.cpu(), args.num_iters, args.batch_size, args.lr)

        print('Training of encoder finished')
        # save the encoder
        torch.save(encoder.state_dict(), 'models/encoder.pth') #saving encoder incase if memory out of error comes
        print('encoder saved in models folder')        

        # Load the encoder
        encoder.load_state_dict(torch.load('models/encoder.pth'))
        print('Encoder loaded')
    

        gc.collect() #garbage collection
        torch.cuda.empty_cache() #empty cache

        # Get the embeddings for the training data
        temp=[]
        # process X_train in batches
        encoder.eval()
        encoded_representations = []
        for i in range(0, len(X_train), args.batch_size):
            X_batch = X_train[i:i+args.batch_size]
            if torch.numel(X_batch) > 0:
                X_batch = ptu.to_numpy(encoder(X_batch))
                encoded_representations.append(X_batch)
            
        z = torch.cat(encoded_representations,0)
        print('Encoding completed')

        # Plot the t-SNE after fitting the encoder
        # plot_tsne(z, y_val)
        plot_tsne(z.cpu(), y_val)
        print('t-SNE plot saved in plots folder')
        
        
    else: # train the classifier (fine-tune the encoder also when using NN classifier)
        # load the encoder
        encoder.load_state_dict(torch.load('models/encoder.pth'))
        print('Encoder loaded')

        #encoding X_val in batches
        temp = []
       
        for i in range(0, len(X_val), args.batch_size):
            X_batch = X_val[i:i+args.batch_size]
            if torch.numel(X_batch) > 0:
                X_batch = ptu.to_numpy(encoder(X_batch))
                encoded_representations.append(X_batch)
           
            
        X_val= torch.cat(temp,0)
        print('X_val encoded')
        
        # Fit the model
        train_losses, train_accs, test_losses, test_accs = fit_model(
            encoder, classifier, X_train, y_train, X_val, y_val, args)
        
        # Plot the losses
        plot_losses(train_losses, test_losses, f'{args.mode} - Losses')
        
        # Plot the accuracies
        plot_accuracies(train_accs, test_accs, f'{args.mode} - Accuracies')
        
        # Get the test data
        X_test, _ = get_data(args.test_data_path)
        X_test = ptu.from_numpy(X_test).float()

        # Save the predictions for the test data in a CSV file
        y_preds = []
        for i in range(0, len(X_test), args.batch_size):
            X_batch = X_test[i:i+args.batch_size].to(ptu.device)
            repr_batch = encoder(X_batch)
            if 'linear' in args.mode:
                repr_batch = ptu.to_numpy(repr_batch)
            y_pred_batch = classifier(repr_batch)
            if 'nn' in args.mode:
                y_pred_batch = ptu.to_numpy(y_pred_batch)
            y_preds.append(y_pred_batch)
        y_preds = np.concatenate(y_preds).argmax(axis=1)
        np.savetxt(f'data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv',\
                y_preds, delimiter=',', fmt='%d')
        print(f'Predictions saved to data/{args.sr_no}_{"repr_lin" if "linear" in args.mode else "repr_nn"}.csv')
