import numpy as np
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange as trange
from time import time

class GradientDescent(object):
    def __init__(self, solver_params):
        self.solver_params = solver_params
        self.x = None
        self.k = 0

    def _error(self):
        return self.f(self.x)

    def solve(self, max_iter, tol):
        return_dict = {}
        for k, d in self.solver_params.items():
            self.f = d['f']
            self.alpha = d.get('alpha', None)
            self.beta = d.get('beta', None)
            self.should_backtrack = (self.alpha is not None) and (self.beta is not None)
            self.grad_f = d['grad_f']
            x0 = d['x0']
            start = time()
            if d['name'] == 'backtracking':
                _x,_k = self._solve_backtracking(x0, max_iter, tol)
            elif d['name'] == 'newton':
                self.hess_f = d['hess_f']
                _x,_k = self._solve_newton(x0, max_iter, tol)
            elif d['name'] == 'bfgs':
                self.hess_f = d['hess_f']
                _x,_k = self._solve_bfgs(x0, max_iter, tol)
            else:
                raise NotImplementedError('{} is not an implemented method.'.format(d['name']))

            _t = time()-start
            return_dict.update({k:{'x':_x, 'k':_k, 't':_t}})
        return return_dict

    def _solve_bfgs(self, x0, max_iter, tol):
        self.x = x0.copy()
        P = np.linalg.inv(self.hess_f(self.x))
        for k in trange(max_iter):

            # Update Direction
            grd = -self.grad_f(self.x)
            d = P.dot(grd)

            # Termination Condition
            _dd = grd.T.dot(d)/2
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            t = 1
            if self.should_backtrack:
                t = self._backtracking(d, -_dd)
            x_old = self.x.copy()
            self.x += t*d

            if k > 0:
                s = self.x - x_old
                s_t = P@s
                y = grd - grd_old
                P += -s_t@s_t.T/(s.T@s_t) + y@y.T/(y.T@s)
            grd_old = grd.copy()

        return self.x, k

    def _solve_newton(self, x0, max_iter, tol):
        self.x = x0.copy()
        for k in trange(max_iter):

            # Update Direction
            inv_hess = np.linalg.inv(self.hess_f(self.x))
            grd = -self.grad_f(self.x)
            d = inv_hess.dot(grd)

            # Termination Condition
            _dd = grd.T.dot(d)/2
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            t = 1
            if self.should_backtrack:
                t = self._backtracking(d, -_dd)
            self.x += t*d
        return self.x, k

    def _solve_backtracking(self, x0, max_iter, tol):
        self.x = x0.copy()
        for k in trange(max_iter):

            # Update Direction
            d = -self.grad_f(self.x)

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
            if np.isinf(c1(t)) or np.isinf(c2(t)):
                raise Exception('INFINITY')
                break

            t *= self.beta
            cond = c1(t) < c2(t)
        return t
