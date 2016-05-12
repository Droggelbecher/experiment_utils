
def jaccard(r1, r2):
    union = np.count_nonzero(r1 | r2)
    if union == 0:
        return 0
    return 1.0 - np.count_nonzero(r1 & r2) / float(union)

