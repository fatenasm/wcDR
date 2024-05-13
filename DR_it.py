# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:28:08 2024

@author: felip
"""

import numpy as np
#from scipy.linalg import convolution_matrix
import matplotlib.pyplot as plt
import pickle

def DR(x,param):
    """
    Performs one Douglas-Rachford iteration, starting from x
    """
    z = x + param['gam']*nabla_f(x,param)
    y = prox_g(x - param['gam']*nabla_f(x,param), param)
    z_out = z + param['lambda']*(y - x)
    x_out = prox_f(z_out,param)
    return x_out, y, z_out

def DRE_val(x,y,param):
    """
    Computes the Douglas-Rachford envelope at z, given x = P(z), y = R(z)
    """
    return f_val(x,param) + np.dot(nabla_f(x,param), y - x) + g_val(y, param) + np.linalg.norm(y-x)**2/(2.0*param['gam'])

""" Objective function """
def f_val(x,param):
    return 0.5*np.linalg.norm(np.dot(param['A'],x) - param['b'])**2

def nabla_f(x, param):
    return np.dot(np.transpose(param['A']),np.dot(param['A'],x).ravel()-param['b'])

def prox_f(z,param):
    return np.linalg.solve(np.dot(np.transpose(param['A']),param['A']) + 
                           np.identity(param['n'])/param['gam'], 
                           np.dot(np.transpose(param['A']),param['b']).ravel() + 
                           z/param['gam'])

""" Regularizers """
def g_val(x,param):
    x_out = np.zeros(len(x))
    if param['penalty'] == 'MCP': # Minimax concave penalty
        for i in range(len(x)):
            if np.absolute(x[i]) <= param['theta']*param['sigma']:
                x_out[i] = param['sigma']*np.absolute(x[i]) - x[i]**2/(2.0*param['theta'])
            else:
                x_out[i] = param['theta']*param['sigma']**2/2.0
    elif param['penalty'] == 'SCAD': # Smoothly clipped absolute deviation
        for i in range(len(x)):
            if np.absolute(x[i]) <= param['sigma']:
                x_out[i] = param['sigma']*np.absolute(x[i])
            elif np.absolute(x[i]) > param['theta']*param['sigma']:
                x_out[i] = 0.5*(param['theta']+1.0)*param['sigma']**2
            else:
                x_out[i] = 0.5*(-x[i]**2 - param['sigma']**2 + 
                                2.0*param['theta']*param['sigma']*np.absolute(x[i]))/(param['theta']-1.0)
    elif param['penalty'] == 'l1':
        return param['mu']*np.linalg.norm(x,1)   
    return np.sum(x_out)

def prox_g(x, param):
    """
    Defines firm thresholding, requieres theta > gamma 
    """
    x_out = np.zeros(len(x))
    if param['penalty'] == 'MCP': # Minimax concave penalty
        for i in range(len(x)):
            if np.absolute(x[i]) > param['theta']*param['sigma']:
                x_out[i] = x[i]
            elif np.absolute(x[i]) < param['gam']*param['sigma']:
                x_out[i] = 0.0
            else:
                x_out[i] = (x[i] - param['gam']*param['sigma']*np.sign(x[i]))/(1.0-(param['gam']/param['theta']))
    elif param['penalty'] == 'SCAD': # Smoothly clipped absolute deviation
        c = param['sigma']*(param['theta']-1.0-param['gam']+param['theta']*param['gam'])/(param['theta']-1.0)
        for i in range(len(x)):
            if np.absolute(x[i]) > param['sigma']*param['theta']:
                x_out[i] = x[i]
            elif np.absolute(x[i]) < c:
                x_out[i] = np.sign(x[i])*np.maximum(0, np.absolute(x[i]) - param['sigma'])
            else:
                x_out[i] = ((param['theta']-1.0)*x[i] - np.sign(x[i])*param['theta']*param['gam']*param['sigma'])/(param['theta']-1.0-param['gam'])
    elif param['penalty'] == 'l1':
        for i in range(len(x)):
            x_out[i] = np.sign(x[i])*np.maximum(0.0, np.absolute(x[i]) - param['gam']*param['mu'])
    return x_out

def phi_val(x,y,param):
    return f_val(x,param) + g_val(y,param)

########################################################
############     Initilization     #####################
########################################################
m = 30#10#50#
n = 90#30#150#
mat_fname = "DR_m"+str(m)+"_n"+str(n)+".pkl"
# Read dictionary pkl file
with open(mat_fname, 'rb') as fp:
    param = pickle.load(fp)
    print('param dictionary')
    #print(param)
    
########################################################
k_max = 5001
param['k_max'] = k_max
########################################################
err = 10**(-6)
param['err'] = err
########################################################

def DR_it_val(x,param):
    """
    Iterates DR splitting method starting from x
    """
    y_prev = prox_g(x - param['gam']*nabla_f(x,param), param)
    xhist = np.zeros(param['k_max'])
    yhist = np.copy(xhist)
    diff = np.copy(xhist)
    DRE = np.copy(xhist)
    val = np.copy(xhist)
    for k in range(param['k_max']):
        x_prev = x
        x, y, z = DR(x_prev, param)
        diff[k] = np.linalg.norm(y-x_prev)
        xhist[k] = np.linalg.norm(x-x_prev)
        yhist[k] = np.linalg.norm(y-y_prev)
        DRE[k] = DRE_val(x_prev,y_prev,param)
        val[k] = phi_val(x_prev,y_prev,param)
        y_prev = y
        #z_prev = z
        if np.absolute(diff[k]) < param['err']:
            return diff[:(k+1)], xhist[:(k+1)], yhist[1:(k+1)], DRE[:(k+1)], val[:(k+1)]
    return diff[:(k+1)], xhist[:(k+1)], yhist[1:(k+1)], DRE[:(k+1)], val[:(k+1)]

########################################################

""" Iteration parameters """
#x0 = param['x0'][3,:] #param['x_sol'].ravel() + 0.1*np.random.rand(param['x_sol'].ravel().shape[0])  #param['x0'][5,:] #param['x_sol'].ravel() + 0.1 #param['x0'][0,:]
param['lambda'] = 1.0
gamma_sup = (2.0-param['lambda'])/(2.0*param['L'])
param['gam'] = 0.9*gamma_sup
""" Function parameters """
param['theta'] = 1.5*param['gam']
param['sigma'] = 0.01

# =============================================================================
# """ MCP """
# param['penalty'] = 'MCP'
# param['theta'] = 1.5*param['gam']
# diff_MCP, x_MCP, y_MCP, DRE_MCP, val_MCP = DR_it_val(x0, param)
# =============================================================================

########################################################
############     Plotting     ##########################
########################################################


diff = list()
xx = list()
yy = list()
DREnv = list()
vall = list()
for s in range(10):
    x0 = param['x0'][s,:]
    print("# of initial point is", s)
    #param['sigma'] = 10**(s-1)
    """ MCP """
    param['penalty'] = 'MCP'
    param['theta'] = 1.5*param['gam']
    diff_MCP, x_MCP, y_MCP, DRE_MCP, val_MCP = DR_it_val(x0, param)
    diff.append(diff_MCP)
    xx.append(x_MCP)
    yy.append(y_MCP)
    DREnv.append(DRE_MCP)
    vall.append(val_MCP)
    """ SCAD """
    param['penalty'] = 'SCAD'
    param['theta'] =1.5*(param['gam'] + 1.0)
    diff_SCAD, x_SCAD, y_SCAD, DRE_SCAD, val_SCAD = DR_it_val(x0, param)
    diff.append(diff_SCAD)
    xx.append(x_SCAD)
    yy.append(y_SCAD)
    DREnv.append(DRE_SCAD)
    vall.append(val_SCAD)
    """ l1 """
    param['penalty'] = 'l1'
    param['mu'] = 1.0
    diff_l1, x_l1, y_l1, DRE_l1, val_l1 = DR_it_val(x0, param)
    diff.append(diff_l1)
    xx.append(x_l1)
    yy.append(y_l1)
    DREnv.append(DRE_l1)
    vall.append(val_l1)
print("end")


# """ MCP """
# DREa = np.absolute(DREnv[0][:-1] - DREnv[0][1:])
# DREb = np.absolute(DREnv[2][:-1] - DREnv[2][1:])
# DREc = np.absolute(DREnv[4][:-1] - DREnv[4][1:])
# DREl1 = np.absolute(DRE_l1[:-1] - DRE_l1[1:])

# plt.figure()
# x1 = np.linspace(1, len(DREa), num=len(DREa))
# x2 = np.linspace(1, len(DREb), num=len(DREb))
# x3 = np.linspace(1, len(DREc), num=len(DREc))
# x4 = np.linspace(1, len(DREl1), num=len(DREl1))
# plt.yscale("log")
# plt.plot(x1, DREa, 'b-', label=r'$\sigma = 0.1$', linewidth=3)
# plt.plot(x2, DREb, 'r-', label=r'$\sigma = 1$', linewidth=3)
# plt.plot(x3, DREc, 'k-', label=r'$\sigma = 10$', linewidth=3)
# plt.plot(x4, DREl1, 'g-', label=r'$\ell_1$', linewidth=3)
# #plt.plot(x2, x_SCAD, 'r--', label = 'SCAD', linewidth=3)
# #plt.plot(x3, x_l1, 'k-', label = r'$\ell_1$',linewidth=3)
# #plt.plot(X, np.absolute(to_plot - valhis), 'm-', linewidth=3)
# #plt.plot(X, yhis, 'r--', linewidth=2)
# plt.xlabel("iterations")
# #plt.ylabel("function values")
# plt.legend()
# plt.grid(True, which="both")
# plt.rcParams.update({'font.size': 14.5})
# plt.xticks(rotation=45)
# type_of = "02_x_"+param['penalty']
# #file_name = type_of+"_m"+str(m)+"_n"+str(n)+".png" 
# file_name = type_of+"_m"+str(m)+"_n"+str(n)+".png" 
# to_save = 0
# if to_save == 1:
#     plt.savefig(file_name, dpi=300, bbox_inches='tight')
# plt.show()

# =============================================================================
# 
# DREa = DREnv[0]
# DREb = DREnv[2]
# DREc = DREnv[4]
# DREl1 = DRE_l1
# =============================================================================

plt.figure()
for s in range(10):
    #to_plot = np.absolute(DREnv[s][1:]-DREnv[s][:-1])
    to_plot = np.absolute(vall[s][1:]-vall[s][:-1])
    #to_plot = np.absolute(DREnv[s] - vall[s])
    #to_plot = diff[s]#xx[s]#
    xlin = np.linspace(1, len(to_plot), num=len(to_plot))
    plt.yscale("log")
    plt.plot(xlin, to_plot, linewidth=3)
    plt.xlabel("iterations")
    #plt.ylabel("function values")
    #plt.legend()
    plt.ylim(10**(-6.5), 10**(1.5)) 
    plt.grid(True, which="both")
    plt.rcParams.update({'font.size': 14.5})
    plt.xticks(rotation=45)
    
to_save = 1
if to_save == 1:
    type_of = "sn_MCP_val"
    #file_name = type_of+"_m"+str(m)+"_n"+str(n)+".png" 
    file_name = type_of+"_m"+str(m)+"_n"+str(n)+".png"
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
plt.show()


# plt.figure()
# for s in range(10):
#     #to_plot = np.absolute(DREnv[s+1][1:]-DREnv[s+1][:-1])
#     to_plot = diff[s]
#     xlin = np.linspace(1, len(to_plot), num=len(to_plot))
#     plt.yscale("log")
#     plt.plot(xlin, to_plot, linewidth=3)
#     plt.xlabel("iterations")
#     #plt.ylabel("function values")
#     #plt.legend()
#     plt.grid(True, which="both")
#     plt.rcParams.update({'font.size': 14.5})
#     plt.xticks(rotation=45)
    
# type_of = "04_DRE_SCAD_init"
# #file_name = type_of+"_m"+str(m)+"_n"+str(n)+".png" 
# file_name = type_of+"_m"+str(m)+"_n"+str(n)+".png" 
# to_save = 0
# if to_save == 1:
#     plt.savefig(file_name, dpi=300, bbox_inches='tight')
# plt.show()




