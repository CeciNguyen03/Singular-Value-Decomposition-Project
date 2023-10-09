import numpy as np
from numpy import array
import numpy.linalg as na
import scipy.linalg as la 
from scipy.linalg import svd

#Singular-value decomposition
def my_svd(matrix):
    try:
        #Calculate SVD, for test
        #U, S, V = np.linalg.svd(matrix, full_matrices=False)

        #Calculate the eigenvalues and eigenvectors of A.T A
        AtA = matrix.T @ matrix
        eigvals, eigvecs = np.linalg.eig(AtA)

        #Sort the eigenvalues and eigenvectors in descending order
        sortedIndices = np.argsort(eigvals)[::-1]
        eigvals = eigvals[sortedIndices]
        eigvecs = eigvecs[:, sortedIndices]

        #Calculate singular values and the matrix U
        singularVals = np.sqrt(eigvals)
        U = matrix @ eigvecs / singularVals

        #Calculate the matrix V
        V = eigvecs

        #Calculate matrix condition number
        condNum = singularVals[0] / singularVals[-1]

        #Calculate the matix inverse using eigenvalue/eigenvector method
        ##eigvals, eigvecs = np.linalg.eig(matrix.T @ matrix) (used for a test)
        if any(np.isclose(singularVals, 0.0)):
            return ValueError("Matrix is singular! It does not have an inverse.")
        
        #Calculate the matrix inverse using the SVD decomp
        matrix_inverse = V @ np.diag(1.0/singularVals) @ U.T

        return {
            "U": U,
            "S": np.diag(singularVals),
            "V": V.T,
            "Condition Number": condNum,
            "Matrix Inverse": matrix_inverse
        }
    except np.linalg.linalg.LinAlgError:
        return ValueError("Matrix is singular! It does not have an inverse.")
    
#Call the function!
if __name__ == "__main__":
    #Create a sample matrix 
    SampMatrix = np.array([[1,2,3],
                           [4,5,6],
                           [7,8,9]])
    #Print the result
    result = my_svd(SampMatrix)
    for key, value in result.items():
        print(f"{key}:\n{value}\n")
    
    #Python svd built-in function
    U, S, VT = svd(SampMatrix)
    print("Matrix U: \n")
    print(U)
    print("\nMatrix S:\n")
    print(S)
    print("\nMatrix VT:\n")
    print(VT)

