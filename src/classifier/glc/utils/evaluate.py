#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description:
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Evaluation module for classification models.

This module provides functions for evaluating the performance of classification models.
It includes functions for creating confusion matrices, converting log-likelihoods to
log-likelihood ratios, and evaluating Cdet, minCdet, and EER for detecting a target class.

Functions:
- confusion_matrix: Create a confusion matrix based on input likelihoods and true class labels.
- logsumexp: Calculate the log-sum-exp of an array along a specified axis.
- loglh2detection_llr: Convert log-likelihoods to log-likelihood ratios.
- evaluate_cdet_mincdet_eer_for_class_detector: Evaluate Cdet, minCdet, and EER for detecting a target class.
- report_avg_cdet_mincdet_eer: Report average Cdet, minCdet, and EER for multiple target classes.

"""
# =============================================================================
# Imports
# =============================================================================
import os, sys, re, gzip
import h5py
import numpy as np

# Rest of the code...
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description:
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Evaluation."""
# =============================================================================
# Imports
# =============================================================================
import os, sys, re, gzip
import h5py
import numpy as np

def confusion_matrix(scores, labs, priors=None):
    """Create confusion matrix, where rows corresponds to true classes and
    columns corresponds to classes recognized from input likelihoods:
    Input:
    scores: N-by-T matrix of log-likelihoods
    labs: T dimensional vectors of true class labels (indices) from range 0:C,
            where C is the number of classes.
    prior: C dimensionl vector of class priors (default: equal priors)
    Output:
    C-by-N confusion matrix, where the field at coordinates [c,n] is the
    number of examples from class c recognized as class n
    """

    n_classes = max(labs) + 1
    priors = np.ones(n_classes) if priors is None else np.array(priors,
        np.double).flatten()
    labs_mx = np.zeros((len(labs), n_classes), dtype=int)
    labs_mx[range(len(labs)), labs] = 1
    rec_mx = np.zeros_like(scores, dtype=int)
    rec_mx[range(len(labs)), np.argmax(scores + np.log(priors[:scores.shape[1]]), axis=1)] = 1

    return labs_mx.T.dot(rec_mx)

def logsumexp(x, axis=0):
    xmax = x.max(axis)
    xmax_e = np.expand_dims(xmax, axis)
    x = xmax + np.log(np.sum(np.exp(x - xmax_e), axis))
    try:
        not_finite = np.where(~np.isfinite(xmax))[0]
        x[not_finite] = xmax[not_finite]
    except Exception:
        pass
    return x


def loglh2detection_llr(loglh, prior=None):
    """
    Converts log-likelihoods to log-likelihood ratios returned by detectors
    of the individual classes as defined in NIST LRE evaluations (i.e. for each
    detector calculates ratio between 1) likelihood of the corresponding class
    and 2) sum of likelihoods of all other classes). When summing likelihoods
    of the compeeting clases, these can be weighted by vector of priors.

    Input:
    loglh - matrix log-likelihoods (trial, class)
    prior - vector of class priors (default: equal priors)

    Output:
    logllr - matrix of log-likelihood ratios (trial, class)
    """
    logllr = np.empty_like(loglh)
    nc = loglh.shape[1]
    prior = np.ones(nc) if prior is None else np.array(prior, np.double).ravel()
    
    for d in range(nc):
        competitors = np.r_[:d, d+1:nc]
        logweights = np.log(prior[competitors] / sum(prior[competitors]))
        logllr[:,d] = loglh[:,d] - logsumexp(loglh[:,competitors] + logweights, axis=1)
    return logllr

def evaluate_cdet_mincdet_eer_for_class_detector(scores, labs, target_class, threshold=0.0, priors=None, p_tar=0.5,Cfa=1.0,Cmiss=1.0):
    """Evaluate Cdet minCdet and EER for task of detecting a targret class,
    which is one of several classes. The other classes serves an nontarget.
    Howewer, when calculating P(FA), non-target examles are NOT simply pooled
    from non-target classes. Instead, P(FA) is calculated for each non-target
    and averaged.

    Input:
    scores: T dimensional vector of detection scores (e.g. log-likelihood ratios
            from loglh2detection_llr())
    labs: T dimensional vectors of true class labels (indices) from range 0:C,
            where C is the number of classes. Every class has to be represented
            (i.e each value from 0:C range has to appear in labs).
    target_class: index (label) of target class
    threshold: scores above this threshold are the detections for evaluation Cdet
    priors: C dimensional vector of class weights. The coefficient corresponding
            to the target class is ignored. The remaining coefficients are
            normalized to sum up to one and used as weights when calculating
            P(FA) as a weighted awerage over classes (default: equal weights).
    p_tar:  Probability of target class (default: 0.5)
    Cfa:    False alarm cost
    Cmiss   Miss cost

    Output:
    cdet, minCdet, eer, minCdet_th, eer_th, cdet_pmiss, cdet_pfa, minCdet_pmiss, minCdet_pfa
    """
    n_classes = max(labs) + 1
    if priors is None:
        priors = np.ones(n_classes)
    if len(priors) != n_classes:
        raise Exception("The vector of priors is expected to have %d elements (one for each class in labs)" % n_classes)

    # represent labs using 1-of-N coding
    labs_mx = np.zeros((len(labs), n_classes), dtype=int)
    labs_mx[range(len(labs)), labs] = 1
    class_counts = np.sum(labs_mx, axis=0).astype(np.double)
    if np.any(class_counts == 0):
        raise Exception("Out of the %d classes, there are classes with no examples" % n_classes)

    # renormalizes priors for nontarget languages
    p_nontar = priors / (priors.sum()-priors[target_class])
    p_nontar[target_class] = 0.0
    # Given a threshold, obtain vector of detection probabilities for each class
    p_hit = np.sum(np.logical_and((scores[:,np.newaxis] > threshold), labs_mx), axis=0) / class_counts
    # Calculate P(FA) as an average (weighted by priors) over non-target classes
    cdet_pfa = p_nontar.dot(p_hit)
    cdet_pmiss = 1.0 - p_hit[target_class]
    cdet = cdet_pfa * Cfa * (1.0 - p_tar) + cdet_pmiss * Cmiss * p_tar
    # obtain matrix where rows are vectors of detection probabilities as above,
    # but now corresponding to every possible threshold
    sorted_score_ids = np.argsort(scores)[::-1]
    p_hit_mx = np.r_[np.zeros((1, n_classes)), labs_mx[sorted_score_ids,:].cumsum(axis=0)] / class_counts
    p_fa = p_hit_mx.dot(p_nontar)
    p_miss = 1.0 - p_hit_mx[:,target_class]
    
    # evaluate cdet for every threshold and select minCdet
    cdets = p_fa * Cfa * (1.0 - p_tar) + p_miss * Cmiss * p_tar
    minCdet_ind = np.argmin(cdets)
    minCdet = cdets[minCdet_ind]

    # Similarly, obtain EER, where P(FA) is averaged over non-target classes
    eer_ind = np.argmin(np.abs(p_fa-p_miss))
    eer = 0.5 * (p_fa[eer_ind] + p_miss[eer_ind])


    return cdet, minCdet, eer, cdet_pmiss, cdet_pfa, p_miss[minCdet_ind], p_fa[minCdet_ind]

def report_avg_cdet_mincdet_eer(scores, labs, class_names=None,  priors=None, p_tar=0.5, Cfa=1.0, Cmiss=1.0):
    """Report LID results given:
    Input:
    scores: matrics of loglikelihood scores
    labs: true labels
    priors: assumed priors for the classes, by default uniform prior
    p_tar: default 0.5
    Cfa:   default 1.0
    Cmiss: default 1.0
    """
    res={}
    n_classes = max(labs) + 1 
    assert(scores.shape[1]==n_classes)
    if priors is None:
        priors = np.ones(n_classes)
    if class_names is None:
        class_names=["class" + str(i) for i in range(n_classes)]

    # calculate threshold, which is optimal for well calibrated log-likelihood ratio scores
    threshold = np.log(Cfa/Cmiss)-np.log(p_tar/(1.0-p_tar))

    # convert input (hopefully well calibrated) log-likelihoods into log-likelihood ratios
    llrs = loglh2detection_llr(scores, priors)
    np.set_printoptions(precision=2,threshold=500,linewidth=500,suppress=False)
    n_all_targets = Cavg = EERavg = minCavg = 0

    # Report Cdet, minCdet, EER for each target class
    # and average the quantities over all target classes
    per_lang={}
    for (target_class_id, target_class_name) in enumerate(class_names):
        (cdet, minCdet, eer, cdet_pmiss, cdet_pfa, minCdet_pmiss, minCdet_pfa) = evaluate_cdet_mincdet_eer_for_class_detector(llrs[:,target_class_id], labs, target_class_id,
                threshold, priors, p_tar, Cfa, Cmiss)

        n_targets = np.sum(labs==target_class_id)
        n_all_targets += n_targets
        Cavg    += cdet    /n_classes
        minCavg += minCdet /n_classes
        EERavg   += eer    /n_classes

        per_lang[target_class_name]={'cdet':cdet, 'minCdet':minCdet, 'eer':eer, 'n_targets':n_targets, 'cdet_pmiss':cdet_pmiss,
                'cdet_pfa':cdet_pfa, 'minCdet_pmiss':minCdet_pmiss, 'minCdet_pfa':minCdet_pfa}

    res['per_lang']=per_lang
    res['avg_eer']=EERavg
    res['cavg']=Cavg
    res['min_cavg']=minCavg
    res['n_all_targets']=n_all_targets
    res['n_tgt_langs']=n_classes

    return res


if(__name__=="__main__"):
    pass

