
import numpy as np

def air_barometric_formula(h,Pb=101325,Tb=293.15,Lb=-0.0065,hb=0): # barometric formula
    '''#Lb temperature gradient, K/m
    #Tb and T should be in K
    #288.15	-0.0065 '''
    g0 = 9.8 # gravitational acceleration
    M = 0.0289644 # MW of air kg/mol
    R = 8.3145 # gas constant J/k/mol
    a = 1+Lb*(h-hb)/Tb
    b = -1*g0*M/R/Lb
    return Pb*np.power(a, b)

def k_rw_mualem(Swe,m,base): 
    '''mualem aqueous relative perm. function'''
    t1 = 1 - np.power(Swe, 1/m)
    t2 = 1 - np.power(t1, m)
    return np.sqrt(Swe)*np.power(t2, 2)-base

def VG_S_P(Se,a,n,m=None): #van Genuchten saturation function, given Se output pressure
    '''Se (effective saturation)
    a (alpha in 1/m)
    n (van Genuchten )
    m (default 1-1/n)
    output is pressure in Pa '''
    if m is None:
        m=1-1/n
    s1 = np.power(Se, -1/m) - 1
    s2 = np.power(s1, 1/n)
    return s2*9800/a #in Pa

def VG_P_S(Pc,a,n,m=None):
    '''Se (effective saturation)
    a (alpha in 1/m) or entry pressure
    n (van Genuchten )
    m (default 1-1/n)
    output is effective saturation '''
    if m is None:
       m=1-1/n
    s0 = -a*Pc
    s1 = np.power(s0, 1/(1-m))
    s2 = np.power(1+s1, -m)
    return s2

def compute_inverse_entry_head(Sw,Pc,Swr=0.05,Sgr=0.01,n=2,m=None):
    #Se is the effective saturation, Pc is capillary pressure in Pa,
    #Solve entry pressure when saturation is known, inverse head is 1/m
    Se = np.divide(Sw-Swr, 1-Sgr)
    #print(Se)
    if m is None:
        m = 1 - np.divide(1, n)
    
    s1 = np.power(Se, -1/m) - 1
    s2 = np.power(s1, 1/n)

    psi = np.divide(Pc, 9800)

    alpha = np.divide(s2,psi)
    return alpha

def rock_phi_eval_Raymer(Vp_bulk,Vp_matrix=6.7,Vp_fluid=1.5): # P-wave velocity in km/s
    # this calculation is strictly for sandstone rich rocks
    b = (Vp_fluid - 2*Vp_matrix) / Vp_matrix
    c = (Vp_matrix - Vp_bulk) / Vp_matrix
    phi = (-1*b - np.sqrt(b*b - 4*c))/2 
    return phi


def rock_Sw_eval_Archie(sigma_bulk,sigma_matrix,sigma_fluid,phi,Sw_ini,Sgr=0.01,m=1.8):
    # calculate the water satuartion from bulk electrical conductivity using a modiefied Archie's equation
    p = np.log(1 - pow(Sw_ini * phi, m))/np.log(1 - (Sw_ini * phi))
    matrix_part = sigma_matrix * pow( 1 - phi, p) 
    temp = (sigma_bulk - matrix_part) / sigma_fluid
    Sw = pow(temp, 1/m) / phi
    Sw = np.minimum(1.0-Sgr, Sw)

    return Sw

def fixed_point_interation(Vp_bulk=5.25,sigma_bulk=0.02,Vp_matrix=6.7,Sw_guess=0.5,sigma_matrix=0,sigma_fluid=10,Sgr=0.01,m=1.8,max_iter=2):
    t_water = 0.66667 # wave travel time in water in s/km
    t_air = 3.3333 # # wave travel time in air in s/km

    for i in range(0, max_iter):
        t_fluid = Sw_guess * t_water + (1 - Sw_guess) * t_air
        Vp_fluid_guess = 1 / t_fluid
        #Vp_fluid_guess = Sw_guess*t_water + (1 - Sw_guess)*t_air
        phi = rock_phi_eval_Raymer(Vp_bulk,Vp_matrix=Vp_matrix,Vp_fluid=Vp_fluid_guess)
        Sw = rock_Sw_eval_Archie(sigma_bulk,sigma_matrix,sigma_fluid,phi,Sw_guess,Sgr=Sgr,m = m)     
        Sw_guess = Sw
    return phi, Sw

def apply_fix(vars, zone, quant, min=0, max=1):
    #fix negative values, greather than 1 values and nans for the variable, quant are the matrix with typically quantile of 0.05, 0.75, 0.95 
    # quant is a matrix with 3 rows, each row is quantile, and each column is for the rock type
    l = np.max(vars.shape)
    for i in range(0, l):
        v = vars[i]
        rt = int(zone[i])
        #print(rt)
        if np.isnan(v):
            vars[i] = quant[1,rt]       
        else:
            if v >= max:
                vars[i] = quant[2,rt]
            else:
                if v <= min:
                    vars[i] = quant[0,rt]
    
    return vars

def perm_scaling_ERT(k_i, logERT_diff, m=1.8,c=2):
     k_f =np.multiply( k_i, np.power(np.power(10,logERT_diff),1/m/c))
     return k_f

def por_scaling(phi, logERT_diff, a=None, b=None,phi_scale=2,n=3):
    if a is None:
        a = np.min(logERT_diff)
    if b is None:
        b = np.max(logERT_diff)
    
    #c = -1*scaling_fun_s(0,a,b)
    S = scaling_fun_s(logERT_diff,a,b, n=n) - scaling_fun_s(0,a,b, n=n)
    
    phi_scaled = np.multiply(1+ phi_scale*S, phi)

    return phi_scaled

def scaling_fun_s(x,a,b, n=2):
    # this function provids a continuous scaling function between a and b, and S goes from 0 to 1
    # a is smaller than b
    lam = np.minimum(1, np.maximum(0, np.divide(x-a,b-a)))
    S = 3 * np.power(lam, n) - 2*np.power(lam,3) #n =2 completely symmetric, n=3 better for por scaling
    return S
    
def capillary_entry_from_perm(k,phi,J0=0.15,gamma=72):
    '''#k is permeaiblity and in Darcy, J0 is the typically entry LJ function
    #gamma is the IFT and should be mN/m
    # entry pressure should be in Pa'''
    k_conversion = 9.87e-13 # convert Darcy to m^2
    gamma_conversion = 1/1000 # convert mN/m to N/m
    denom = np.sqrt(np.divide(k*k_conversion,phi))

    return np.divide(J0*gamma*gamma_conversion, denom)

def radioactivity_to_mole(Arad,t_half):
    '''
    convert radioactivity to number of moles
    Arad is radioactivity in Bq, 
    and t_half is half life in days
    T is temperature in C
    P is pressure in Bar
    '''
    N_A = 6.022e23
    temp1 = t_half*24*3600 # convert days to sec
    temp2 = np.divide(N_A*np.log(2), temp1)
    n = np.divide(Arad, temp2)
    print(f'convert mole to activity {Arad/n:4e} Bq/mole with halflife of {t_half:.3f} days')
    #R = 0.0831446261815324 #L*bar*K−1*mol−1
    #Tk = T +273.15
    #P = 1.013
    return n

def mole_to_radioactivity(n,t_half):
    ''' convert moles to radioactivity in Bq
    n is the number of moles
    t_half is the half-life in days
    return radioacitivty in Bq
    '''
    #convert moles to radioactivity
    #n is moles and t_half is half life in days
    N_A = 6.022e23
    temp1 = t_half*24*3600 # convert days to sec
    A = n*N_A*np.log(2)/(temp1)
    print(f'convert activity to mole {n/A:4e} mole/Bq with halflife of {t_half:.3f} days')
    return A

class rel_perm:
    def __init__(self, phase, res_sat, endpoint_perm, corey_exponent):
        self.phase = phase 
        self.Sr = res_sat
        self.kr0 = endpoint_perm
        self.n = corey_exponent

    def calc(self, sat, interp=False):
        res_sat_array = np.atleast_1d(self.Sr)
        #denom = 1 - np.sum(res_sat_array)

        temp1 = np.divide(sat - res_sat_array[0], 1 - np.sum(res_sat_array))
        #temp1 = np.divide(sat - res_sat_array[0], 1 - res_sat_array[0])
        temp2 = np.maximum(np.minimum(temp1, 1.0), 0.0)


        kr = self.kr0 * np.power(temp2, self.n)
        #print(kr)
        if interp is True:
            index_right = sat > 1 - res_sat_array[-1]
            kr_right = np.interp(sat[index_right], [1 - res_sat_array[-1], 1], [self.kr0, 1] )
            kr[index_right] = kr_right
        #krf = np.maximum(np.minimum(kr, 1.0), 0.0)
        return kr

def permeability_to_hydraulic_conductivity(K,rho=1000,g=9.81,mu=1e-3):
    ''' K is permeability in m^2
        rho is density in kg/m^3
        g is gravity acceleration in m/s^2
        mu is viscosity in Pa*s
        return hydraulic conductivity in m/s'''
    hc = K*(rho*g)/mu
    hc_min = np.min(hc) * 31556952000.00043 # convert m/s to mm/year
    print(f'minimum hydraulic conductivity is {hc_min:.3f} mm/year')
    return hc

class perm_model():
    def __init__(self, c=78.8, m= 1.82):
        self.c = c
        self.m = m
    
    def calc(self,por):
        perm = self.c * np.power(por, self.m) / (1 - por)/ 1000
        print('-'*100)
        print('input porosity is {:4.2f}'.format(por))
        print('the calculated permeabiity is {:4.6f} Darcy'.format(perm))
        return perm

def residuals(params, x_data, y_data, fitting_function):
    ''' params: no need to specify if used in scipy least_sqaure
        xdata: measured x values
        ydata: measured y values
        fitting function, example,
            def fit_model(x, c, m): # this is the fitting function
            return c * x**m / (1-x) # should be defined outsize the residual function 
        how to use? lssol = least_squares(residuals, x0=(50, 1), args=(xdata, ydata, fit_model)) '''
    y_predict = fitting_function(x_data, *params) # predicted
    diff = np.subtract( y_data, y_predict) # measured value - predicted value
    weights_sum = np.sum(1 / y_data ** 2)  # the weight is proportional to the 1/measured^2
    weights = 1 / y_data** 2 / weights_sum # normalize the weight
    #weights = 1 / np.abs( y_predict )**1
    #print(np.sum(weights))
    ## not sure if this makes sense, but we weigth with function value
    ## here I put it inverse linear as it gets squared in the chi-square
    ## but other weighting may be required

    # lssol = least_squares(residuals, x0=(50, 1), args=(xdata, ydata, fit_model))
    # print(lssol.x )
    return diff * weights