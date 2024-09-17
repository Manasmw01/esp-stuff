import numpy as np

def gaussian_inverse(matrix):
    """
    Calculate the inverse of a matrix using NumPy's linear algebra module.
    
    Args:
    - matrix (np.ndarray): A square matrix to invert (e.g., 46x46).
    
    Returns:
    - np.ndarray: The inverse of the input matrix if it is invertible.
    """
    # Check if the matrix is square
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square to compute its inverse.")
    
    # Try to calculate the inverse
    try:
        matrix_inv = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        raise ValueError("Matrix is singular and cannot be inverted.")
    
    return matrix_inv

# Example usage:
size = 46  # Set the size of the matrix
matrix = np.random.rand(size, size)  # Create a random matrix of the specified size

# Calculate the inverse
matrix_inv = gaussian_inverse(matrix)

# Print both the input and output
print("Input Matrix ({}x{}):\n".format(size, size), matrix)
print("\nInverse of the Matrix ({}x{}):\n".format(size, size), matrix_inv)
