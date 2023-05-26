import numpy as np
from scipy.optimize import minimize
from EconModel import EconModelClass

class optimaltax(EconModelClass):
    def settings(self):
        """ Fundamental settings """
        pass
    
    def setup(self):
        """ Set baseline parameters """
        
        # Unpack
        par = self.par
        
        # parameters
        par.alpha = 0.5 # Substitutability between goods
        par.kappa = 1.0 # Free private consumption component
        par.nu = 1 / (2*16*16) # Disutility of labor scaling factor
        par.w = 1.0# Real wage
        par.tau = 0.3 # Labor income tax rate 
        par.G = 1.0 # Government spending
        par.sigma = 1.001 # Elasticity of substitution
        par.rho = 1.001 # CRRA coefficient
        par.epsilon = 1.0 # CRRA coefficient
        
    def allocate(self):
        """ Allocate model """
        
    def utility(self, C, G, L, type='simple'):
        """ Utility function"""
        # unpack
        par = self.par
        
        # utility
        if type == 'simple':
            c_util = np.log(C**par.alpha * G**(1-par.alpha))
            l_util = - par.nu * (L*L / 2)
            util = c_util + l_util
        elif type == 'complex':
            #CES utility
            c_util = ((((par.alpha * C**((par.sigma-1)/par.sigma)) + ((1-par.alpha) * G**((par.sigma-1)/par.sigma)))**(par.sigma/(par.sigma-1)))**(1-par.rho) - 1) / (1-par.rho)
            l_util = - par.nu * (L**(1+par.epsilon) / (1+par.epsilon))
            util = c_util + l_util
        else:
            raise ValueError(f'Unknown type: {type}')
        
        return util
        
    def BC_implied_C(self, L):
        """ Budget constraint """
        # unpack
        par = self.par
        
        # budget constraint
        C = par.kappa + (1-par.tau) * par.w * L
        
        return C
    
    def solve(self, x0=10, type = 'simple'):
        """ Solve model """
        
        # Unpack
        par = self.par
        
        # Define objective function
        obj = lambda L: -self.utility(self.BC_implied_C(L), par.G, L, type=type)
        
        #Make labor between 0 and 24
        bounds = [(0,24)]
        
        # Solve
        res = minimize(obj, x0, method='Nelder-Mead',bounds=bounds, tol=1.0e-10)
        
        return res.x[0]
    
    def analytic_solution(self, w_tilde):
        """ Analytical solution """
        # unpack
        par = self.par
        
        # analytical solution
        L = (-par.kappa + np.sqrt(par.kappa*par.kappa + 4 * par.alpha/par.nu * w_tilde*w_tilde)) / (2 * w_tilde)
        
        return L