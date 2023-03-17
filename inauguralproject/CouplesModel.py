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
        sol_cont = self.sol_vec  = SimpleNamespace()
        
        #b. Set parameters
        #i. Preferences
        par.rho     = 2.0
        par.nu      = 0.001
        par.epsilon = 1.
        par.omega   = 0.5
        
        #ii. Household production
        par.alpha   = 0.5
        par.sigma   = 1.
        
        #iii. Wages
        par.wM      = 1.
        par.wF      = 1.
        
        #iv. Other
        par.Tmin = 0.
        par.Tmax = 24.
        
        #c. Grids
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
        
        #iii. Make grids
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
            return ((1-par.alpha)*HM**((par.sigma-1)(par.sigma)) + par.alpha*HF**((par.sigma-1)(par.sigma)))**(par.sigma/(par.sigma-1))
    
    def cons(self, C, H):
        '''Total consumption'''
        #a. Unpack
        par = self.par
        
        #b. Return
        return C**par.omega*H**(1-par.omega) + 1e-10
        
    def utility(self, Q, TM, TF):
        '''Utility function'''
        #a. Unpack
        par = self.par
        
        #b. Return
        return Q**(1-par.rho)/(1-par.rho) - par.nu*(TM**(1+1/par.epsilon)/(1+1/par.epsilon) + TF**(1+1/par.epsilon)/(1+1/par.epsilon))
    
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
        return self.utility(Q,TM,TF)
        
    
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
        res = minimize(obj, (init.LM,init.LF,init.HM,init.HF,), method='SLSQP', constraints=constr, bounds=bounds, tol=1e-10)
        
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
        if discrete:
            sol = self.sol_disc
        else:
            sol = self.sol_cont
        
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
        
    def plot(self):
        '''Plot results'''
        #a. Unpack
        par = self.par
        sol = self.sol_cont
        
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
            X = np.column_stack((const,X))
        else:
            X = x
        
        #b. Return
        return np.linalg.inv(X.T@X)@X.T@y
        
    def estimate_beta(self, constant=True):
        '''Estimate beta'''
        #a. Unpack
        par = self.par
        sol = self.sol_cont
        
        #b. Calculate variables
        y = np.log(sol.HF_vec/sol.HM_vec)
        x = np.log(par.wF_vec/par.wM)
        
        #c. Estimate beta
        return self.OLS(y,x,constant)
    
    def SMM(self, theta, moments, weights):
        '''Simulated Method of Momemts'''
        
        theta_hat = moments.T @ weights @ moments
        
    def match_moment(self):
        
        
        
        
        
    
    
    