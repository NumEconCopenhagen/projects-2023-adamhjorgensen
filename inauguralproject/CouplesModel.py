import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from scipy.optimize import minimize, NonlinearConstraint

class CouplesModel():
    def __init__(self,**kwargs):
        self.settings()
        self.setup(**kwargs)
    
    def settings(self):
        pass
    
    def setup(self,**kwargs):
        
        #a. Make name spaces
        par = self.par = SimpleNamespace()
        sol_disc = self.sol_disc = SimpleNamespace()
        sol_cont = self.sol_cont = SimpleNamespace()
        
        #b. Set parameters
        #i. Preferences
        par.rho     = 2.0
        par.nu      = 0.001
        par.epsilon = 1.
        par.omega   = 0.5
        par.xi      = 0.0
        par.eta     = 0.0
        
        #ii. Household production
        par.alpha   = 0.5
        par.sigma   = 1.
        
        #iii. Wages
        par.wM      = 1.
        par.wF      = 1.
        
        #iv. Other
        par.Tmin = 0.
        par.Tmax = 24.
        
        #c. Grid parameters
        #i. Time spend on work
        par.Lmin = 0.
        par.Lmax = 24.
        par.nL = 49
        
        #ii. Time spend on home production
        par.Hmin = 0.
        par.Hmax = 24.
        par.nH = 49
        
        #iii. Female wages
        par.wF_min = 0.8
        par.wF_max = 1.2
        par.nwF    = 5
        
        #d. Update parameters
        for key,val in kwargs.items():
            setattr(par,key,val) 
        
        #e. Make grids
        par.L_vec = np.linspace(par.Lmin,par.Lmax,par.nL)
        par.H_vec = np.linspace(par.Hmin,par.Hmax,par.nH)
        par.wF_vec= np.linspace(par.wF_min,par.wF_max,par.nwF)
        
    def market(self, LM, LF):
        ''' Consumption of market goods'''
        #a. Unpack
        par = self.par
        
        #b. Return
        return par.wF*LF + par.wM*LM
    
    def homeprod(self, HM, HF):
        ''' Consumption of home produced goods'''
        #a. Unpack
        par = self.par
        
        #b. Return
        if par.sigma == 0.: #minimum
            return np.min(HM,HF)
        elif par.sigma == 1.: #Cobb-Douglas
            return HM**(1-par.alpha)*HF**par.alpha
        else: #CES
            return ((1-par.alpha)*HM**((par.sigma-1)/(par.sigma)) + par.alpha*HF**((par.sigma-1)/(par.sigma)))**(par.sigma/(par.sigma-1))
    
    def cons(self, C, H):
        '''Total consumption'''
        #a. Unpack
        par = self.par
        
        #b. Return
        return C**par.omega*H**(1-par.omega) + 1e-10
    
    def CRRA(self, var, param):
        '''CRRA utility function'''
        if param == 1.:
            return np.log(var)
        elif param == 0.:
            return var
        else:
            return var**(1-param)/(1-param)
        
    def utility(self, Q, TM, TF,HF):
        '''Utility function'''
        #a. Unpack
        par = self.par
        
        #b. Return
        return self.CRRA(Q,par.rho) \
                - par.nu * self.CRRA(TM,-1/par.epsilon) \
                - par.nu * self.CRRA(TF,-1/par.epsilon) \
                + par.xi * self.CRRA(HF,par.eta)
    
    def value_of_choice(self, LM,LF,HM,HF):
        '''Value of choice'''
        #a. Unpack
        par = self.par
        
        #b. Calculate
        C = self.market(LM,LF)
        H = self.homeprod(HM,HF)
        Q = self.cons(C,H)
        TM = LM+HM
        TF = LF+HF
        
        #c. Return
        return self.utility(Q,TM,TF,HF)
        
    
    def solve_discrete(self):
        '''Solver for discrete problem'''
        #a. Unpack
        par = self.par
        sol = self.sol_disc
        
        #b. Make grids
        mLM, mLF, mHM, mHF = np.meshgrid(par.L_vec, par.L_vec, par.H_vec, par.H_vec)
        vLM = mLM.ravel() #vectors
        vLF = mLF.ravel()
        vHM = mHM.ravel()
        vHF = mHF.ravel()
        
        #c. Calculate value of choice
        vV = self.value_of_choice(vLM,vLF,vHM,vHF)
        
        #d. Apply conditions with penalty
        cond = (vLM+vHM <= par.Tmax) & (vLF+vHF <= par.Tmax)
        vV[~cond] = -np.inf
        
        #e. Find optimal choice
        sol.V = np.max(vV)
        idx = np.argmax(vV)
        sol.LM = vLM[idx]
        sol.LF = vLF[idx]
        sol.HM = vHM[idx]
        sol.HF = vHF[idx]
        
    def solve_continuous(self):
        '''Solver for continuous problem (requires solution for discrete problem)'''
        #a. Unpack
        par = self.par
        sol = self.sol_cont
        init = self.sol_disc
        
        #b. Define objective function
        obj = lambda x: -self.value_of_choice(x[0],x[1],x[2],x[3])
        
        #c. Define constraints
        constr1 = NonlinearConstraint(lambda x: x[0]+x[2], par.Tmin, par.Tmax)
        constr2 = NonlinearConstraint(lambda x: x[1]+x[3], par.Tmin, par.Tmax)
        constr = (constr1, constr2)
        
        #d. Define bounds
        bounds = ((par.Lmin,par.Lmax),(par.Lmin,par.Lmax),(par.Hmin,par.Hmax),(par.Hmin,par.Hmax))
        
        #e. Solve
        res = minimize(obj, (init.LM,init.LF,init.HM,init.HF,), method='SLSQP', constraints=constr, bounds=bounds, tol=1e-8)
        
        # f. Save results
        sol.LM = res.x[0]
        sol.LF = res.x[1]
        sol.HM = res.x[2]
        sol.HF = res.x[3]
        sol.V = -res.fun
        
    def solve(self):
        '''Solve model'''
        self.solve_discrete()
        self.solve_continuous()
        
    def solve_wF_vec(self,discrete=False):
        '''Solve model for vector of wF'''
        #a. Unpack
        par = self.par
        if discrete: sol = self.sol_disc
        else:        sol = self.sol_cont
        
        #b. Save initial value
        init_wF = par.wF
        
        #c. Allocate
        n = len(par.wF_vec)
        sol.LM_vec = np.empty(n)
        sol.LF_vec = np.empty(n)
        sol.HM_vec = np.empty(n)
        sol.HF_vec = np.empty(n)
        sol.V_vec = np.empty(n)
        
        #d. Solve
        for i in range(n):
            par.wF = par.wF_vec[i]
            if discrete:
                self.solve_discrete()
            else:
                self.solve()
            sol.LM_vec[i] = sol.LM
            sol.LF_vec[i] = sol.LF
            sol.HM_vec[i] = sol.HM
            sol.HF_vec[i] = sol.HF
            sol.V_vec[i] = sol.V
        
        #e. Restore initial value
        par.wF = init_wF
        
    def plot(self, discrete=False):
        '''Plot results'''
        #a. Unpack
        par = self.par
        if discrete: sol = self.sol_disc
        else:        sol = self.sol_cont
        
        #b. Calculate variables
        y = np.log(sol.HF_vec/sol.HM_vec)
        x = np.log(par.wF_vec/par.wM)
        
        #c. Plot
        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.set_xlabel('wF')
        ax.set_ylabel('log( HF / HM )')
        plt.show()
        
    def OLS(self,y,x,constant=True):
        '''Ordinary least squares'''
        #a. Handle constant
        if constant:
            const = np.ones(len(y))
            X = np.column_stack((const,x))
        else:
            X = x
        
        #b. Return
        return np.linalg.inv(X.T@X)@X.T@y
        
    def estimate_beta(self):
        '''Estimate beta'''
        #a. Unpack
        par = self.par
        sol = self.sol_cont
        
        #b. Calculate variables
        y = np.log(sol.HF_vec/sol.HM_vec)
        x = np.log(par.wF_vec/par.wM)
        
        #c. Estimate beta
        return self.OLS(y,x)
    
    def SMD(self, theta0, pnames, target, bounds, method='Nelder-Mead', tol=1e-08, weights=None):
        
        #a. Define objective function
        obj = lambda x: self.SMD_obj(x, pnames, target, weights)
        
        #b. Optimize
        res = minimize(obj, theta0, bounds=bounds, method=method, tol=tol)
        
        return res
    
    def SMD_obj(self, theta, pnames, target, weights):
        '''Simulated Method of Momemts'''
        
        #a. Get actual moments
        actual_moment = np.array(target)
        
        #b. Get simulated moments
        sim_moment = self.simulate_moments(theta, pnames)
        
        #c. Calculate deviation
        g = actual_moment - sim_moment
        
        #d. Handle weights
        if weights==None:
            weights = np.eye(len(g))
        
        #d. Define objective function
        print('obj: ',g.T @ weights @ g)
        return g.T @ weights @ g
        
    def simulate_moments(self, theta, pnames):
        '''Simulate moments'''
        #a. update parameters
        self.updatepar(pnames, theta)
        print('guess: ', theta)
        
        #b. Solve model
        self.solve_wF_vec()
        
        #c. Estimate beta (moment)
        beta0, beta1 = self.estimate_beta()
        
        #d. Return moments
        return np.array([beta0, beta1])
        
    def updatepar(self,parnames, parvals):
        """ Update parameters """
        for i,parname in enumerate(parnames):
            parval = parvals[i]
            setattr(self.par,parname,parval)
        
        
        
        
        
    
    
    