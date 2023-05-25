import numpy as np
from scipy.optimize import minimize
from EconModel import EconModelClass

class hairsalon(EconModelClass):
    def settings(self):
        """ Fundamental settings """
        pass
    
    def setup(self):
        """ Set baseline parameters """
        # Unpack
        par = self.par
        
        # parameters
        par.nu = 0.5 # elasticity of demand
        par.w = 1.0 # real wage
        par.rho = 0.9 # AR(1) coefficient
        par.iota = 0.01 # Fixed adjustment cost
        par.sigma = 0.1 # Standard deviation of AR(1) shock
        par.R = (1.01)**(1/12) # Monthly interest rate
        par.Delta = 0.05 # Adjustment limit
        par. gamma = 1.0 # Adjustment speed
        
        # Other
        par.T = 120 # Number of periods
        par.K = 100 # Number of nodes for shock
        
    def allocate(self):
        """ Allocate model """
        # unpack
        par = self.par
        
        # Allocate
        np.random.seed(2023) # Set seed
        par.epsilon = np.random.normal(-0.5 * par.sigma**2, par.sigma, (par.T,par.K)) # Shock
        
        par.kappa = np.zeros((par.T,par.K)) + np.nan # Series of marginal cost
        par.kappa[0,:] = np.exp(par.rho * np.log(1) + par.epsilon[0,:]) # AR(1) marginal cost # Initial marginal cost
        for t in range(1,par.T):
            par.kappa[t,:] = np.exp(par.rho * np.log(par.kappa[t-1,:]) + par.epsilon[t,:]) # AR(1) marginal cost
    
        
    def profit(self, kappa, L, L_lag):
        """ Profit function """
        # unpack
        par = self.par
        
        # profit function
        pi = kappa * L**(1-par.nu) - par.w * L - par.iota * (L != L_lag)
        
        return pi
    
    def analytic_solution(self, kappa):
        """ Analytic solution """
        # unpack
        par = self.par
        
        # analytic solution
        L = ((1-par.nu)*kappa / par.w)**(1/par.nu)
        
        return L
    
    def L_rule(self, kappa, L_lag, rule = 'analytic'):
        """ Analytic solution """
        # unpack
        par = self.par
        
        # analytic solution
        L = self.analytic_solution(kappa)
        
        # Rule
        if rule == 'analytic':
            return L
        elif rule == 'Delta':
            return np.where(abs(L_lag-L) > par.Delta, L, L_lag)
        elif rule == 'alternative':
            return np.where(abs(L_lag-L) > par.Delta, L_lag + par.gamma * (L-L_lag), L_lag)
        else:
            raise ValueError(f'Unknown type: {type}')
    
    def solve_static(self, kappa, guess=10):
        """ Solve model numerically"""
        
        # Unpack
        par = self.par
        
        obj = lambda L: -self.profit(kappa, L,L)
        res = minimize(obj, x0=guess, method='Nelder-Mead')
        
        return res.x[0]
    
    
    def H_value(self, rule='analytic'):
        
        # Unpack
        par = self.par
        
        # Allocate
        pi = np.zeros((par.T,par.K)) + np.nan
        L = np.zeros((par.T,par.K)) + np.nan
        h_total = 0
        
        #Initial value
        L[0,:] = self.L_rule(kappa=1,L_lag=0)
        pi[0,:] = self.profit(kappa=1,L=L[0,:],L_lag=0)
        
        # Simulate
        for t in range(1,par.T-1):
            L[t,:] = self.L_rule(par.kappa[t,:],L[t-1,:],rule=rule)
            pi[t,:] = self.profit(par.kappa[t,:],L[t,:],L[t-1,:])
            h_total += par.R**(-t) * np.mean(pi[t,:])
            
        return h_total