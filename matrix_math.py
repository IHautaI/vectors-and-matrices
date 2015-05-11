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
    return shape(vect1) == shape(vect2)


def vector_add(vect1, vect2):

    if compare(vect1, vect2):
        return [coeff1 + coeff2 for (coeff1, coeff2) in list(zip(vect1,vect2))]

    raise ShapeException


def vector_sub(vect1, vect2):
    return vector_add(vect1, [-num for num in vect2])


def vector_sum(*vectors):
    vectors = list(vectors)
    if len(vectors) > 1:
        return vector_add(vectors.pop(), vector_sum(*vectors))
    else:
        return vectors[0]


def dot(vect1, vect2):
    if compare(vect1, vect2):
        return sum([coeff1 * coeff2 for (coeff1, coeff2) in list(zip(vect1,vect2))])

    raise ShapeException

def vector_multiply(vector, scalar):
    return [entry * scalar for entry in vector]


def vector_mean(*vectors):
    return vector_multiply(vector_sum(*vectors), 1/len(vectors))


def magnitude(vector):
    return math.sqrt(sum([entry**2 for entry in vector]))


def matrix_row(matrix, row):
    return matrix[row]


def transpose(matrix):
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def matrix_col(matrix, col):
    return matrix_row(transpose(matrix),col)


def matrix_scalar_multiply(matrix, scalar):
    return [vector_multiply(vect, scalar) for vect in matrix]


def matrix_vector_multiply(matrix, vector):
    if compare(matrix[0], vector):
        return [dot(row, vector) for row in matrix]
    raise ShapeException


def matrix_matrix_multiply(mat1, mat2):
    if shape(mat1)[1] == shape(mat2)[0]:
        return [[dot(row,col) for col in transpose(mat2)] for row in mat1]

    raise ShapeException
