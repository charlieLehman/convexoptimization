import numpy as np
from tqdm import tqdm_notebook as tqdm
from tqdm import tnrange as trange
from time import time

class GradientDescent(object):
    def __init__(self, solver_params):
        self.solver_params = solver_params
        self.x = None
        self.k = 0
        self.results = solver_params.get('results', [])
        self.methods = solver_params['methods']
        self.res_template = lambda method, k, _dd: {'k':k,
                             't':time()-self.start,
                             'fx':self.f(self.x).squeeze().astype(float),
                             'error':_dd.squeeze().astype(float),
                             'method':method,
                             'x':self.x,
                             'run':self.solver_params.get('run', 0)}

    def solve(self, max_iter, tol):
        for k, d in self.methods.items():
            self.f = d['f']
            self.alpha = d.get('alpha', None)
            self.beta = d.get('beta', None)
            self.should_backtrack = (self.alpha is not None) and (self.beta is not None)
            self.grad_f = d['grad_f']
            x0 = d['x0']
            self.start = time()
            if d['name'] == 'backtracking':
                _x,_k = self._solve_backtracking(x0, max_iter, tol)
            elif d['name'] == 'newton':
                self.hess_f = d['hess_f']
                _x,_k = self._solve_newton(x0, max_iter, tol)
            elif d['name'] == 'bfgs':
                self.hess_f = d['hess_f']
                _x,_k = self._solve_bfgs(x0, max_iter, tol)
            elif d['name'] == 'heavy_ball':
                self.alpha = d['alpha']
                self.beta = d['beta']
                _x,_k = self._solve_heavyball(x0, max_iter, tol)
            else:
                raise NotImplementedError('{} is not an implemented method.'.format(d['name']))
        return self.results

    def _solve_bfgs(self, x0, max_iter, tol):
        self.x = x0.copy()
        #P = np.eye(2)
        P = self.hess_f(x0)
        for k in range(max_iter):
            # Update Direction
            grd = self.grad_f(self.x)
            d = -np.linalg.inv(P)@grd

            # Termination Condition
            _dd = -grd.T@d/2
            self.results.append(self.res_template('BFGS', k, _dd))
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            t = 1
            if self.should_backtrack:
                t = self._backtracking(d, grd)
            x_old = self.x.copy()
            self.x += t*d
            if k > 0:
                s = self.x - x_old
                s_t = P@s
                y = grd - grd_old
                sden = s.T@s_t
                yden = y.T@s
                P += -s_t@s_t.T/sden + y@y.T/yden
            grd_old = grd.copy()

        return self.x, k

    def _solve_newton(self, x0, max_iter, tol):
        self.x = x0.copy()
        for k in range(max_iter):

            # Update Direction
            inv_hess = np.linalg.inv(self.hess_f(self.x))
            grd = self.grad_f(self.x)
            d = -inv_hess@grd

            # Termination Condition
            _dd = -grd.T@d/2
            self.results.append(self.res_template('Newton', k, _dd))
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            t = 1
            if self.should_backtrack:
                t = self._backtracking(d, grd)
            self.x += t*d
        return self.x, k

    def _solve_backtracking(self, x0, max_iter, tol):
        self.x = x0.copy()
        for k in range(max_iter):

            # Update Direction
            grd = self.grad_f(self.x)
            d = -grd

            # Termination Condition
            _dd = d.T@d
            self.results.append(self.res_template('SGD', k, _dd))
            if _dd <= tol:
                return self.x, k

            # Update Step Size
            t = self._backtracking(d, grd)
            self.x += t*d
        return self.x, k

    def _solve_heavyball(self, x0, max_iter, tol):
        self.x = x0.copy()
        x_old = self.x.copy()
        for k in range(max_iter):

            # Update Direction
            grd = self.grad_f(self.x)
            d = -grd

            # Termination Condition
            _dd = d.T@d
            self.results.append(self.res_template('SGD', k, _dd))
            if _dd <= tol:
                return self.x, k

            self.x += self.alpha*d, self.beta*(self.x-x_old)
            x_old = self.x.copy()
        return self.x, k

    def _backtracking(self, d, grd):
        t = 1
        c1 = lambda t: self.f(self.x + t*d)
        c2 = lambda t: self.f(self.x) + self.alpha*t*d.T@grd
        _c1 = c1(t)
        _c2 = c2(t)
        __c1 = 0
        __c2 = 0
        cond = _c1 < _c2
        for _ in range(10000):
            __c1 = _c1.copy()
            __c2 = _c2.copy()

            t *= self.beta
            _c1 = c1(t)
            _c2 = c2(t)
            cond = _c1 < _c2
            if cond:
                return t
        return t
