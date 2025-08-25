import numpy as np

# Graded cell
def center_data(Y):
    """
    Center your original data
    Args:
         Y (ndarray): input data. Shape (n_observations x n_pixels)
    Outputs:
        X (ndarray): centered data
    """
    ### START CODE HERE ###
    mean_vector = np.mean(Y, axis=0)
    mean_matrix = np.reshape(np.repeat(mean_vector, Y.shape[0]), Y.shape, order='F')
    # use np.reshape to reshape into a matrix with the same size as Y. Remember to use order='F'
    
    X = Y - mean_matrix
    ### END CODE HERE ###
    return X

def get_cov_matrix(X):
    """ Calculate covariance matrix from centered data X
    Args:
        X (np.ndarray): centered data matrix
    Outputs:
        cov_matrix (np.ndarray): covariance matrix
    """

    ### START CODE HERE ###
    m = X.shape[0]
    cov_matrix = np.dot(X.T, X)
    cov_matrix = (1/(m-1)) * (cov_matrix)
    ### END CODE HERE ###
    
    return cov_matrix

# GRADED cell
def perform_PCA(X, eigenvecs, k):
    """
    Perform dimensionality reduction with PCA
    Inputs:
        X (ndarray): original data matrix. Has dimensions (n_observations)x(n_variables)
        eigenvecs (ndarray): matrix of eigenvectors. Each column is one eigenvector. The k-th eigenvector 
                            is associated to the k-th eigenvalue
        k (int): number of principal components to use
    Returns:
        Xred
    """
    
    ### START CODE HERE ###
    V = eigenvecs[:, :k]
    Xred = np.dot(X, V)
    ### END CODE HERE ###
    return Xred