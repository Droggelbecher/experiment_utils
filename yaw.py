
import numpy as np

def yawdiff(a, b):
    return np.mod(a-b+np.pi, 2.0 * np.pi) - np.pi

def smooth_yaw_inplace(a):
    while True:
        diffs = a[1:] - a[:-1]
        idxs_gt = np.where(diffs > np.pi)[0]
        idxs_lt = np.where(diffs <= -np.pi)[0]
        if not len(idxs_gt) and not len(idxs_lt):
            return a
        if len(idxs_gt):
            a[idxs_gt[0] + 1:] -= 2.0 * np.pi
            continue
        if len(idxs_lt):
            a[idxs_lt[0] + 1:] += 2.0 * np.pi
            continue
    return a

def compute_yaw(xs, ys):
    dx = np.hstack(([0], xs[1:] - xs[:-1]))
    dy = np.hstack(([0], ys[1:] - ys[:-1]))
    return np.arctan2(dy, dx)

def normyaw(ys):
    return np.mod(ys+np.pi, 2.0 * np.pi) - np.pi
