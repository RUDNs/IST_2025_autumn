import numpy as np
from numpy.linalg import LinAlgError
import scipy
from datetime import datetime
from collections import defaultdict

class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    def __init__(self, method = "Wolfe", **kwargs):
        self._method = method
        if self._method == "Wolfe":
            self.c1 = kwargs.get("c1", 1e-4)
            self.c2 = kwargs.get("c2", 0.9)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Armijo":
            self.c1 = kwargs.get("c1", 1e-4)
            self.alpha_0 = kwargs.get("alpha_0", 1.0)
        elif self._method == "Constant":
            self.c = kwargs.get("c", 1.0)
        else:
            raise ValueError("Unknown method {}".format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError("LineSearchTool initializer must be of type dict")
        return cls(** options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha = None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        # TODO: Implement line search procedures for Armijo, Wolfe and Constant steps.

        def phi(alpha):
            return oracle.func_directional(x_k, d_k, alpha)

        def grad_phi(alpha):
            return oracle.grad_directional(x_k, d_k, alpha)

        phi0         = phi(0.0)

        # Условия Армихо
        if self._method == "Armijo":
            if previous_alpha is None:
                alpha = self.alpha_0
            else:
                alpha = previous_alpha

            while phi(alpha) > phi0 + self.c1 * alpha * grad_phi(alpha):
                alpha /= 2.0

        # Сильные условия Вульфа
        if self._method == "Wolfe":
            alpha = scipy.optimize.line_search(
                f          = oracle.func,
                myfprime   = oracle.grad,
                xk         = x_k,
                pk         = d_k,
                c1         = self.c1,
                c2         = self.c2,
                amax       = self.alpha_0,
            )[0]

            if alpha is None:
                if previous_alpha is None:
                    alpha = self.alpha_0
                else:
                    alpha = previous_alpha

                # Условия Армихо
                while phi(alpha) > phi0 + self.c1 * alpha * grad_phi(alpha):
                    alpha /= 2.0

        # Постоянный шаг
        if self._method == "Constant":
            alpha = self.c

        return alpha

def get_line_search_tool(line_search_options = None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()

def gradient_descent(oracle, x_0, tolerance = 1e-5, max_iter = 10000,
                     line_search_options = None, trace = False, display = False):
    """
    Gradien descent optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
        Printing format and is up to a student and is not checked in any way.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = gradient_descent(oracle, np.zeros(5), line_search_options={'method': 'Armijo', 'c1': 1e-4})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """
    # TODO: Implement gradient descent
    # Use line_search_tool.line_search() for adaptive step size.

    now                = datetime.now()

    start_time         = datetime(
                                now.year,
                                now.month,
                                now.day,
                                now.hour,
                                now.minute,
                                now.second,
                         )

    history                = defaultdict(list) if trace else None
    if trace == True:
        history["time"]           = []
        history["func"]           = []
        history["grad_norm"]      = []
        history["x"]              = []

    message                 = ""
    x_star                  = None

    if np.isnan(x_0).any() or \
       np.isinf(x_0).any():
        message = "computational_error"
        x_star = x_0

        return x_star, message, history

    initial_grad        = oracle.grad(x_0)
    initial_grad_norm   = initial_grad @ initial_grad
    if initial_grad_norm < tolerance:
        if trace == True:
            history["time"].append((datetime.now() - start_time).total_seconds())
            history["func"].append(oracle.func(x_0).item())
            history["grad_norm"].append((np.sqrt(initial_grad_norm)).item())
            if x_0.size <= 2:
                history["x"].append(x_0)


        message = "success"
        x_star = x_0

        return x_star, message, history
    else:
        x_k                 = np.copy(x_0)
        ratio_tolerance     = 1e-5
        line_search_tool    = get_line_search_tool(line_search_options)
        current_grad_val    = oracle.grad(x_k)
        alpha_k             = None

        for iter_num in range(0, max_iter, 1):
            d_k = -oracle.grad(x_k)

            alpha_k = line_search_tool.line_search(
                oracle          = oracle,
                x_k             = x_k,
                d_k             = d_k,
                previous_alpha  = alpha_k,
            )

            x_k = x_k + alpha_k * d_k
            if np.isnan(x_k).any() or \
               np.isinf(x_k).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            current_func_val = oracle.func(x_k)
            if np.isnan(current_func_val).any() or \
               np.isinf(current_func_val).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            current_grad_val = oracle.grad(x_k)
            if np.isnan(current_grad_val).any() or \
               np.isinf(current_grad_val).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            current_grad_norm = current_grad_val @ current_grad_val
            if np.isnan(current_grad_norm).any() or \
               np.isinf(current_grad_norm).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            if display == True:
                print(f"Текущая точка -                                 {x_k:.5f}")
                print(f"Текущее значение функции -                      {current_func_val:.5f}")
                print(f"Текущее значение градиента -                    {current_grad_val:.5f}")
                print(f"Текущее значение евклидовой нормы градиента -   {current_grad_norm:.5f}")
                print(f"Номер итерации -                                {iter_num}")
                print()

            if trace == True:
                history["time"].append((datetime.now() - start_time).total_seconds())
                history["func"].append(current_func_val.item())
                history["grad_norm"].append((np.sqrt(current_grad_norm)).item())
                if x_k.size <= 2:
                    history["x"].append(x_k)

            if current_grad_norm < ratio_tolerance * initial_grad_norm:
                message = "success"
                x_star = x_k

                return x_star, message, history

        if message == "":
            message = "iterations_exceeded"

        x_star = x_k

        if trace == True:
            history["time"].append((datetime.now() - start_time).total_seconds())

        return x_star, message, history

def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2

    Example:
    --------
    >> oracle = QuadraticOracle(np.eye(5), np.arange(5))
    >> x_opt, message, history = newton(oracle, np.zeros(5), line_search_options={'method': 'Constant', 'c': 1.0})
    >> print('Found optimal point: {}'.format(x_opt))
       Found optimal point: [ 0.  1.  2.  3.  4.]
    """

    # TODO: Implement Newton's method.
    # Use line_search_tool.line_search() for adaptive step size.

    now                    = datetime.now()

    start_time             = datetime(
                                now.year,
                                now.month,
                                now.day,
                                now.hour,
                                now.minute,
                                now.second,
                             )

    history                = defaultdict(list) if trace else None
    if trace == True:
        history["time"]           = []
        history["func"]           = []
        history["grad_norm"]      = []
        history["x"]              = []

    message                = ""
    x_star                 = None

    if np.isnan(x_0).any() or \
       np.isinf(x_0).any():
        message = "computational_error"
        x_star = x_0

        return x_star, message, history

    initial_grad        = oracle.grad(x_0)
    initial_grad_norm   = initial_grad @ initial_grad
    if initial_grad_norm < tolerance:
        if trace == True:
            history["time"].append((datetime.now() - start_time).total_seconds())
            history["func"].append((oracle.func(x_0)))
            history["grad_norm"].append(np.sqrt(initial_grad_norm).item())

            if x_0.size <= 2:
                history["x"].append(x_0)

        message = "success"
        x_star = x_0

        return x_star, message, history
    else:
        x_k                 = np.copy(x_0)
        ratio_tolerance     = 1e-5
        line_search_tool    = get_line_search_tool(line_search_options)
        current_grad_val    = oracle.grad(x_k)
        alpha_k             = None
        d_k                 = None

        for iter_num in range(0, max_iter, 1):
            current_hess_matrix     = oracle.hess(x_k)
            current_grad_val        = oracle.grad(x_k)
            try:
                # d_k = scipy.linalg.cho_solve(scipy.linalg.cho_factor(current_hess_matrix), -current_grad_val)
                d_k = np.linalg.solve(current_hess_matrix, -current_grad_val)
            except LinAlgError:
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            if np.isnan(d_k).any() or \
               np.isinf(d_k).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            alpha_k = line_search_tool.line_search(
                oracle          = oracle,
                x_k             = x_k,
                d_k             = d_k,
                previous_alpha  = alpha_k,
            )

            x_k = x_k - alpha_k * d_k
            if np.isnan(x_k).any() or \
               np.isinf(x_k).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            current_func_val = oracle.func(x_k)
            if np.isnan(current_func_val).any() or \
               np.isinf(current_func_val).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            current_grad_val = oracle.grad(x_k)
            if np.isnan(current_grad_val).any() or \
               np.isinf(current_grad_val).any():
                message = "computational_error"
                x_star = x_k

                return x_star, message, history

            current_grad_norm = current_grad_val @ current_grad_val
            if np.isnan(current_grad_norm).any() or \
               np.isinf(current_grad_norm).any():
                message = "computational_error"
                x_star = x_k

            if display == True:
                print(f"Текущая точка -                                 {x_k:.5f}")
                print(f"Текущее значение функции -                      {current_func_val:.5f}")
                print(f"Текущее значение градиента -                    {current_grad_val:.5f}")
                print(f"Текущее значение евклидовой нормы градиента -   {current_grad_norm:.5f}")
                print(f"Текущее значение гессиана функции -             {current_hess_matrix:.5f}")
                print(f"Номер итерации -                                {iter_num}")
                print()

            if trace == True:
                history["time"].append((datetime.now() - start_time).total_seconds())
                history["func"].append(current_func_val)
                history["grad_norm"].append((np.sqrt(current_grad_norm)).item())

                if x_k.size <= 2:
                    history["x"].append(x_k)

            if current_grad_norm < ratio_tolerance * initial_grad_norm:
                message = "success"
                x_star = x_k

                return x_star, message, history

        if message == "":
            message = "iterations_exceeded"

        x_star = x_k

        if trace == True:
            history["time"].append((datetime.now() - start_time).total_seconds())

        return x_star, message, history
