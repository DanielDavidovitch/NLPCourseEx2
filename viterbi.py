import numpy as np
import operator

def viterbi (word_list, A, B, Pi):

    # initialization
    T = len(word_list)
    N = A.shape[0] # number of tags

    delta_table = np.zeros((N, T)) # initialise delta table
    psi = np.zeros((N, T))  # initialise the best path table

    delta_table[:,0] = B[:, word_list[0]] * Pi

    for t in range(1, T):
        for s in range (0, N):
            trans_p = delta_table[:, t-1] * A[:, s]
            psi[s, t], delta_table[s, t] = max(enumerate(trans_p), key=operator.itemgetter(1))
            delta_table[s, t] = delta_table[s, t] * B[s, word_list[t]]

    # Back tracking
    seq = np.zeros(T);
    seq[T-1] =  delta_table[:, T-1].argmax()
    for t in range(T-1, 0, -1):
        seq[t-1] = psi[int(seq[t]),t]

    return seq


if __name__ == '__main__':
    A = np.array([[0.3, 0.7], [0.2, 0.8]])
    B = np.array([[0.1, 0.1, 0.3, 0.5], [0.3, 0.3, 0.2, 0.2]])
    Pi = np.array([0.4, 0.6])
    print viterbi([3, 3, 3, 3], A, B, Pi)
