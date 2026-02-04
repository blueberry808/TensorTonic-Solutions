'''
Implement one update step of the Adam optimizer. Given current parameter(s), gradient(s), and running first/second moments, return the updated parameter(s) and updated moments.



'''

import numpy as np

def adam_step(param, grad, m, v, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    One Adam optimizer update step.
    Return (param_new, m_new, v_new).
    """
    # Write code here
    mt = momentum_estimate(beta1, m, grad)
    m_hat = memomentum_bias(mt, beta1, t) 

    vt = velocity_estimate(beta2, v, grad)
    v_hat = velocity_bias(vt, beta2, t) 

    weights = weight_update(param, lr, m_hat, v_hat, eps)
    return weights, mt, vt 




def momentum_estimate(B1, m, grad): 
    return B1*m + (1-B1) * grad 

def velocity_estimate(B2,v, grad): 
    return B2*v +  (1-B2) *(np.square(grad)) 

def memomentum_bias(mt, B1,t): 
    return (mt)/(1-B1**t) #** is the exponential symbol in numpy 

def velocity_bias(vt, B2,t): 
    return (vt)/(1-B2**t)

def weight_update(param, lr, mt, vt, eps): 
    return param - lr * (mt/(np.sqrt(vt) +eps))






