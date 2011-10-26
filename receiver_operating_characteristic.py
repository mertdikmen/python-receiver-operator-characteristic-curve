import numpy as np

num_roc_points = 250

def compute_roc(pos_res, neg_res):
#    pos_res.sort()
    neg_res.sort()

#    thresholds = np.linspace(min_score, max_score, num_roc_points, endpoint=True)[::-1]
    inds = neg_res.size - np.floor( np.logspace(0, np.log10(neg_res.size), num_roc_points) ) 

    thresholds = neg_res[inds.astype('int')]

    true_pos = np.zeros(num_roc_points)
    false_pos = np.zeros(num_roc_points)

    for i,t in enumerate(thresholds):
        true_pos[i] = np.sum(pos_res > t).astype('float32') / pos_res.size
        false_pos[i] = np.sum(neg_res > t).astype('float32') / neg_res.size

    return true_pos, false_pos, thresholds
