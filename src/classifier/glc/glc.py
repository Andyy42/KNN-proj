#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description:
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Gaussian Linear Classifier."""
# =============================================================================
# Imports
# =============================================================================
import os,sys
import numpy as np
import argparse
import h5py
import matplotlib.pyplot as plt
from utils.evaluate import *
from utils.split_embeddings import load_h5_file, create_split, average_data, check_split


def load_embd(embd_file):
    '''
    Reads embeddings stored in .h5 file.

    Args:
        embd_file (str): The path to the .h5 file containing the embeddings.

    Returns:
        tuple: A tuple containing the following elements:
            - data (ndarray): The embeddings stored as a matrix of size N x D, where N is the number of datapoints and D is the embedding dimension.
            - physical (ndarray): The physical names for each embedding, stored as a vector of size N x 1. If not provided, it returns a copy of the label vector.
            - labs (ndarray): The language labels, stored as a vector of size N x 1.

    Raises:
        Exception: If there is an error reading the embeddings file.

    '''
    try:
        with h5py.File(embd_file,'r') as f:
            data=np.array(f['Data'])
            try:
                labs=np.array(f['Name'])
            except KeyError:
                labs=[]
            try:
                physical=np.array(f['Physical'])
            except KeyError:
                print("The file does not contain physical names for the embeddings")
                physical=labs.copy()
    except Exception as e:
        print("Could not read embeddings file, exiting")
        raise e
        sys.exit()
    return data,physical,labs

def save_scores(scores,physical,classes,filename):
    '''
    Saves the scores to a file.

    Args:
        scores (ndarray): The scores to be saved.
        physical (ndarray): The physical names for each embedding.
        classes (ndarray): The language labels.
        filename (str): The path to the file where the scores will be saved.

    Returns:
        None

    '''
    with open(filename,'w') as f:
        f.write("segmentid\t"+("\t").join(classes.astype(str))+"\n")
        for i,fn in enumerate(physical):
            f.write(fn.decode('utf-8')+"\t"+("\t").join(scores[i,:].astype(str))+'\n')


def compute_wc_cov(X,y,balance_wc_estimate=False):
    '''
    Computes the within-class covariance matrix.

    Args:
        X (ndarray): The input samples.
        y (ndarray): The target values.
        balance_wc_estimate (bool, optional): Whether to balance the within-class covariance estimate. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - WC (ndarray): The within-class covariance matrix.
            - mus (ndarray): The means of each class.

    '''
    Dim,N = X.shape
    K = len(np.unique(y))
    Cwc = np.zeros((Dim,Dim))
    mus = np.zeros((Dim,K))

    for cls in range(K):
        d = X[:,np.nonzero(y == cls)[0]]
        mu = np.mean(d,1)
        mus[:,cls] = mu
        d -= mu[:,np.newaxis]
        if balance_wc_estimate:
            Cwc += np.dot(d,d.T) / d.shape[1]
        else:
            Cwc += np.dot(d,d.T) # just accumulate scatter
    if balance_wc_estimate:
        WC = Cwc / K
    else:
        WC = Cwc / X.shape[1]
    return WC,mus


class GLC:
    def fit(self, X, y, balance_wc_estimate=False):
        """
        Fits the GLC classifier to the training data.

        Args:
            X: array-like, shape (n_samples, n_features)
              The input samples.
            y: array-like, shape (n_samples,)
              The target values.
            balance_wc_estimate: bool, optional (default=False)
              Whether to balance the within-class covariance estimate.

        Returns:
            self: object. Returns the instance itself.
        """
        self.classes, y = np.unique(y, return_inverse=True)
        self.WC, self.means = compute_wc_cov(X, y, balance_wc_estimate=balance_wc_estimate)
        BiM = np.linalg.solve(self.WC, self.means)
        self.MtBiM = np.sum(self.means * BiM, axis=0)
        return self

    def loglh(self, X, mask=None):
        """
        Calculate the log-likelihood of the given input data.

        Args:
            X: numpy array
              Input data array.
            mask: numpy array, optional
              Mask array to apply on the input data.

        Returns:
            Y: numpy array
              Log-likelihood values.

        """
        BiX = np.linalg.solve(self.WC, X)
        D = self.MtBiM[:, np.newaxis] - np.dot(2 * self.means.T, BiX)
        XtBiX = np.sum(X * BiX, axis=0)
        Y = -0.5 * (D + XtBiX)
        return Y

    def save(self,filename):
        '''
        Saves the GLC classifier to a file.

        Args:
            filename (str): The path to the file where the classifier will be saved.

        Returns:
            None

        '''
        with h5py.File(filename,'w') as f:
            f.create_dataset('WC',data=self.WC)
            f.create_dataset('means',data=self.means)
            f.create_dataset('classes',data=self.classes)

    def load(self,filename):
        '''
        Loads the GLC classifier from a file.

        Args:
            filename (str): The path to the file where the classifier is saved.

        Returns:
            self: object. Returns the instance itself.

        '''
        with h5py.File(filename,'r') as f:
            self.WC=np.array(f['WC'])
            self.means=np.array(f['means'])
            self.classes=np.array(f['classes'])
        BiM=np.linalg.solve(self.WC, self.means)
        self.MtBiM=np.sum(self.means*BiM, axis=0)
        return self

def test_different_lengths(dev_data, test_data, args):
    """
    Test the GLC classifier with different lengths of data.

    Args:
        dev_data (Data): The development data.
        test_data (Data): The test data.
        args: Additional arguments.

    Returns:
        None
    """

    N = 30

    # Initial values
    X_dev_start = np.copy(dev_data.X)
    y_dev_start = np.copy(dev_data.y)
    ids_dev_start = np.copy(dev_data.ids)
    # Initial values
    X_test_start = np.copy(test_data.X)
    y_test_start = np.copy(test_data.y)
    ids_test_start = np.copy(test_data.ids)

    for i in range(N):

        print(40*"=")
        print(f"[{i+1}/{N}]")

        check_split(dev_data.y, test_data.y)

        print("Train GLC")
        glc = GLC().fit(dev_data.X.T, dev_data.y, balance_wc_estimate=args.balance_wc_estimate)
        print("Compute scores for the test embeddings")
        # test_vecs,test_filenames,test_labs=load_embd(args.test_file)
        languages=glc.classes
        scores=glc.loglh(test_data.X.T)

        # The inverse with ideces to reconstruct from unique
        # Simply converts labels to array of indeces
        _, test_indices_y = np.unique(test_data.y, return_inverse=True)

        if args.compute_results:
            print("Compute performance metrics as in NIST LRE22 + accuracy")
            res01=report_avg_cdet_mincdet_eer(scores.T,test_indices_y,class_names=languages,p_tar=0.1)
            res05=report_avg_cdet_mincdet_eer(scores.T,test_indices_y,class_names=languages)

            correct = (np.argmax(scores, axis=0) == test_indices_y).sum()
            accuracy = correct / len(test_indices_y)

            str_res01="Cavg0.1: "+str(res01['cavg'])+"\tmin_Cavg01: "+str(res01['min_cavg'])+"\n"
            str_res05="Cavg0.5: "+str(res05['cavg'])+"\tmin_Cavg05: "+str(res05['min_cavg'])+"\n"
            str_prim="Cprimary: " + str(5.0*res01['cavg']+res05['cavg'])+"\tmin_Cprimary: "+str(5.0*res01['min_cavg']+res05['min_cavg'])+"\n"
            str_acc=f"Accuracy: {accuracy}\n"
            print(str_res01)
            print(str_res05)
            print(str_prim)
            print(str_acc)
            with open(args.out_dir+'/results.txt', 'a') as f:
                f.write(str_res01)
                f.write(str_res05)
                f.write(str_prim)
            with open(args.out_dir+'/cprim.txt', 'a') as f:
                f.write(str(5.0*res01['cavg']+res05['cavg'])+"\n")
            with open(args.out_dir+'/accuracy.txt', 'a') as f:
                f.write(str(accuracy)+"\n")

        # Merges consecutive data samples by 2,3,4,5, ... in each iteration
        dev_data  = average_data(data=X_dev_start, group_id=y_dev_start , names=ids_dev_start, n=i+2)
        test_data = average_data(data=X_test_start, group_id=y_test_start , names=ids_test_start, n=i+2)

def main():
    parser = argparse.ArgumentParser(description='GLC')
    # parser.add_argument('-f', '--train_file', type=str, default='', help='.h5 file with the training embeddings')
    # parser.add_argument('-t', '--test_file', type=str, default='', help='.h5 file with the test (evaluation) embeddings')
    parser.add_argument('-f', '--data_file', type=str, default='', help='.h5 file with embeddings')
    parser.add_argument('-o', '--out_dir',type=str, required=True, help='the directory where to save the model and the scores')
    parser.add_argument('--balance_wc_estimate', type=bool, default=False, help='balance_wc_estimate')
    # parser.add_argument('-m','--model', type=str, default='', help='pretrained model')
    parser.add_argument('-p','--plotting', type=bool, default=False, help='plot confusion matrix for the test data')
    parser.add_argument('-r','--compute_results', type=bool, default=False, help='plot confusion matrix for the test data')
    args = parser.parse_args()
    if os.path.isdir(args.out_dir):
        pass
    else:
        os.makedirs(args.out_dir)    
    # if os.path.isfile(args.model):
    #     print("Loading pretrained model")
    #     glc=GLC().load(args.model)

    assert(os.path.isfile(args.data_file))
    print('Reading training data')

    # train_vecs,train_filenames,train_labs=load_embd(args.train_file)
    name, group_id, data = load_h5_file(file=args.data_file, group_index="SubgroupID")
    dev_data, test_data  = create_split(name=name, group_id=group_id, data=data, ignore_groups=[b"5", b"7"], average_n=1)

    test_different_lengths(dev_data=dev_data, test_data=test_data, args=args)

if __name__ == '__main__':
    main()

