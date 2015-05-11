import math


class ShapeException(Exception):
    pass


def shape(vector):
    """
    with no regard for poorly made lists
    returns the shape of the list
    """
    if isinstance(vector[0], list):
        return (len(vector), len(vector[0]))
    return (len(vector), )


def compare(vect1, vect2):
    """
    Checks if the shapes of the two entries
    are the same
    """
    return shape(vect1) == shape(vect2)


def vector_add(vect1, vect2):
    """
    Returns the sum of two vectors
    if they are the same shape
    """
    if compare(vect1, vect2):
        return [coeff1 + coeff2 for (coeff1, coeff2)
                in list(zip(vect1, vect2))]

    raise ShapeException


def vector_sub(vect1, vect2):
    """
    Returns the difference of two vectors
    if their shapes are the same
    """
    return vector_add(vect1, [-num for num in vect2])


def vector_sum(*vectors):
    """
    Returns the sum of any number of vectors
    if their shapes are the same
    """
    vectors = list(vectors)
    if len(vectors) > 1:
        return vector_add(vectors.pop(), vector_sum(*vectors))
    else:
        return vectors[0]


def dot(vect1, vect2):
    """
    Returns the dot product of two vectors
    if their shape is the same
    """
    if compare(vect1, vect2):
        return sum([coeff1 * coeff2 for (coeff1, coeff2)
                    in list(zip(vect1, vect2))])

    raise ShapeException


def vector_multiply(vector, scalar):
    """
    Returns the result of multiplying a vector
    by a scalar
    """
    return [entry * scalar for entry in vector]


def vector_mean(*vectors):
    """
    Returns the mean of any set of vectors
    if their shape is the same
    """
    return vector_multiply(vector_sum(*vectors), 1/len(vectors))


def magnitude(vector):
    """
    Returns the magnitude of a vector
    in the Euclidean metric
    """
    return math.sqrt(sum([entry**2 for entry in vector]))


def matrix_row(matrix, row):
    """
    returns a specified row of a matrix
    """
    return matrix[row]


def transpose(matrix):
    """
    returns the transpose of a matrix
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def matrix_col(matrix, col):
    """
    returns a specified column of a matrix
    """
    return matrix_row(transpose(matrix), col)


def matrix_scalar_multiply(matrix, scalar):
    """
    Returns the matrix multiplied by a scalar
    """
    return [vector_multiply(vect, scalar) for vect in matrix]


def matrix_vector_multiply(matrix, vector):
    """
    Returns the result of multiplying
    a given matrix by a given vector
    if their shapes are appropriate
    """
    if compare(matrix[0], vector):
        return [dot(row, vector) for row in matrix]
    raise ShapeException


def matrix_matrix_multiply(mat1, mat2):
        """
        Returns the result of multiplying
        a given two matrices
        if their shapes are appropriate
        """
    if shape(mat1)[1] == shape(mat2)[0]:
        return [[dot(row, col) for col in transpose(mat2)] for row in mat1]

    raise ShapeException
