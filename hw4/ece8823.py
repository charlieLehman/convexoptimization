import numpy as np
from tqdm import tqdm

class GradientDescent(object):
    SOLVER_PARAMS_DEFAULT = {
        'method':{
            'name':'backtracking',
            'f':None,
            'grad_f':None,
            'alpha':0.001,
            'beta':0.8
        },
    }

    def __init__(self, solver_params=SOLVER_PARAMS_DEFAULT):
        self.solver_params = solver_params
        self.x = None
        self.k = 0

    def solve(self, x0, max_iter, tol):
        return_dict = {}
        for k, d in self.solver_params.items():
            print('Running {}'.format(k))
            self.f = d['f']
            self.grad_f = d['grad_f']
            if d['name'] == 'backtracking':
                self.alpha = d['alpha']
                self.beta = d['beta']
                _x,_k = self._solve_backtracking(x0, max_iter, tol)
            elif d['name'] == 'newton':
                self.hess_f = d['hess_f']
                _x,_k = self._solve_newton(x0, max_iter, tol)
            else:
                raise NotImplementedError('{} is not an implemented method.'.format(d['name']))

            return_dict.update({k:{'x':_x, 'k':_k}})
        return return_dict

    def _solve_newton(self, x0, max_iter, tol):
        self.x = x0.copy()
        for k in range(max_iter):

            # Update Direction
            inv_hess = np.linalg.inv(self.hess_f(self.x))
            print(inv_hess)
            grd = -self.grad_f(self.x)
            d = inv_hess@grd.T

            # Termination Condition
            _dd = grd@d/2
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            self.x += d
        return self.x, k

    def _solve_backtracking(self, x0, max_iter, tol):
        self.x = x0.copy()
        for k in range(max_iter):

            # Update Direction
            d = -self.grad_f(self.x).T

            # Termination Condition
            _dd = d.T@d
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            t = self._backtracking(d, -_dd)
            self.x += t*d
        return self.x, k

    def _backtracking(self, d, _dd):
        t = 1
        c1 = lambda t: self.f(self.x + t*d)
        c2 = lambda t: self.f(self.x) + self.alpha*t*_dd
        cond = c1(t) < c2(t)
        while not cond:
            t *= self.beta
            cond = c1(t) < c2(t)
        return t
