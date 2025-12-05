import numpy as np
import scipy
from scipy.special import expit


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            raise ValueError('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax     = matvec_Ax
        self.matvec_ATx    = matvec_ATx
        self.matmat_ATsA   = matmat_ATsA
        self.b             = b
        self.regcoef       = regcoef

    def func(self, x):

        # TODO: Implement

        # m = self.b.size
        # if m != self.matvec_ATx(x).size or m != self.matvec_Ax(x).size:
        #     # raise AssertionError()
        #     return None

        # if scipy.sparse.issparse(x):
        #     sprase_x                  = scipy.sparse.csr_matrix(x)
        #     sparse_Ax                 = scipy.sparse.csr_matrix(self.matvec_Ax(sprase_x))
        #     sparse_b                  = scipy.sparse.csr_matrix(self.b)
        #     sparse_result             = -sparse_b @ sparse_Ax

        #     return (-1.0 / m) * np.sum(scipy.stats.log(expit(sparse_result.toarray()))) + \
        #            (self.regcoef / 2.0) * (scipy.linalg.norm(sprase_x) ** 2)
        # else:
        #     return (1.0 / m) * np.sum(np.logaddexp(0, -self.b * self.matvec_Ax(x))) + \
        #            (self.regcoef / 2.0) * (np.linalg.norm(x) ** 2)

        z = self.matvec_Ax(x) * self.b
        return (1.0 / self.b.size) * np.mean(np.log(1.0 + np.exp(-z))) + 0.5 * self.regcoef * (x.T @ x)

    def grad(self, x):

        # TODO: Implement

        # m = self.b.size
        # if m != self.matvec_ATx(x).size or m != self.matvec_Ax(x).size:
        #     # raise AssertionError()
        #     return None

        # if scipy.sparse.issparse(x):
        #     sparse_x                  = scipy.sparse.csr_matrix(x)
        #     sparse_Ax                 = scipy.sparse.csr_matrix(self.matvec_Ax(sparse_x))
        #     sparse_b                  = scipy.sparse.csr_matrix(self.b)
        #     sparse_b_Ax               = scipy.sparse.csr_matrix(-sparse_b * sparse_Ax)
        #     sparse_ATx                = scipy.sparse.csr_matrix(self.matvec_ATx(sparse_b * scipy.special.expit(sparse_b_Ax.toarray())))

        #     return (-1.0 / m) * np.sum(sparse_ATx) + self.regcoef * sparse_x
        # else:
        #     return (-1.0 / m) * np.sum(self.matvec_ATx(self.b * expit(-self.b * self.matvec_Ax(x)))) + self.regcoef * x

        z = self.matvec_Ax(x) * self.b
        return (-1.0 / self.b.size) * (self.matvec_ATx(expit(z) - 1.0) * self.b) + self.regcoef * x

    def hess(self, x):

        # TODO: Implement

        # m = self.b.size
        # if m != self.matvec_ATx(x).size or m != self.matvec_Ax(x).size:
        #     # raise AssertionError()
        #     return None

        # if scipy.sparse.issparse(x):
        #     sparse_x                  = scipy.sparse.csr_matrix(x)
        #     sparse_matvec_Ax          = scipy.sparse.csr_matrix(self.matvec_Ax(sparse_x))
        #     sparse_b                  = scipy.sparse.csr_matrix(self.b)
        #     sigmoid                   = expit((sparse_b * sparse_matvec_Ax).toarray())
        #     sparse_sigmoid            = scipy.sparse.csr_matrix(sigmoid)
        #     sparse_matmat_ATsA        = scipy.sparse.csr_matrix(sparse_sigmoid * (1.0 - sparse_sigmoid))

        #     return (1.0 / m) * sparse_matmat_ATsA + self.regcoef * sparse_x
        # else:
        #     sigmoid = expit(self.b * self.matvec_Ax(x))
        #     return (1.0 / m) * self.matmat_ATsA(sigmoid * (1.0 - sigmoid)) + self.regcoef * np.eye(x.size)

        z = self.matvec_Ax(x) * self.b
        s = expit(z) * (1.0 - expit(z))
        if scipy.sparse.issparse(x):
            return (-1.0 / self.b.size) * (self.matmat_ATsA(s)) + self.regcoef * scipy.sparse.identity(x.size)
        else:
            return (-1.0 / self.b.size) * (self.matmat_ATsA(s)) + self.regcoef * np.eye(x.size)

class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

    def func_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None

    def grad_directional(self, x, d, alpha):
        # TODO: Implement optimized version with pre-computation of Ax and Ad
        return None


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    matvec_Ax = lambda x :       A @ x
    matvec_ATx = lambda x :      A.T @ x
    matmat_ATsA = lambda s :     (A.T * s) @ A

    if oracle_type == "usual":
        oracle = LogRegL2Oracle
    elif oracle_type == "optimized":
        oracle = LogRegL2OptimizedOracle
    else:
        raise "Unknown oracle_type=%s" % oracle_type

    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)

def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the gradient

    result = []
    length = x.size
    for i in range(0, length, 1):
        e_i     = np.zeros(length)
        e_i[i]  = 1.0
        result.append((func(x + eps * e_i) - func(x)) / eps)

    return np.array(result)

def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    # TODO: Implement numerical estimation of the Hessian

    length = x.size
    result = np.zeros((length, length))
    for i in range(0, length, 1):
        for j in range(0, length, 1):
            e_i      = np.zeros(length)
            e_i[i]   = 1.0

            e_j      = np.zeros(length)
            e_j[j]   = 1.0

            result[i, j] = (func(x + eps * e_i + eps * e_j) - func(x + eps * e_i) - \
                            func(x + eps * e_j) + func(x)) / (eps ** 2)

    return result
