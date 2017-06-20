# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 11:05:49 2016

@author: mtinti
"""
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
from string import strip
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
from scipy.optimize import leastsq
from sklearn import metrics
from scipy import signal
from numpy import trapz
#experiment temperature range
from scipy.misc import derivative

_TM = [37,41,44,47,50,53,56,59,62,65]

def sigmoid(p,x):
    x0,y0,c,k=p
    y = c / (1 + np.exp(-k*(x-x0))) + y0
    return y
    
def sigmoid_derivative(p, x):
    x0,y0,c,k=p
    f = c / (1 + np.exp(-k*(x-x0))) + y0
    return -k / f    
    

def residuals(p, x, y):
    return y - sigmoid(p,x)

def resize(arr, lower=0.1, upper=0.9):
    arr=arr.copy()
    if lower>upper: lower,upper=upper,lower
    arr -= arr.min()
    arr *= (upper-lower)/arr.max()
    arr += lower
    return arr
        
def norm_max(X):
    X = X/X.max()
    return X

def norm_sum(X):
    X = X/X.sum()
    return X

def norm_first(X):
    X = X/X.values[0]
    return X
    
def find_fit(x,y):
    #y=resize(y, lower=0.1)
    p_guess=(np.median(x), np.median(y), 1.0, 1.0)
    p, cov, infodict, mesg, ier = leastsq(residuals,
                                          p_guess,
                                          args=(x,y),
                                          full_output=1)
    ss_err=(infodict['fvec']**2).sum()
    ss_tot=((y-y.mean())**2).sum()
    rsquared =1-(ss_err/ss_tot)
    res_fit = sigmoid(p, x)
    if np.isnan(res_fit).any() == True:
        mae=np.nan
        mse=np.nan
        tm =np.nan
    else:
        mae = metrics.mean_absolute_error(y, res_fit)
        mse = metrics.mean_squared_error(y, res_fit)   
        tm = np.nan
        temp_x = np.arange(x[0], x[-1], 0.001)
        for value in temp_x:
            temp_y = sigmoid(p, value) 
            if  temp_y <= 0.5:
                tm=value
                break
    return p, cov, infodict, mesg, ier, rsquared, ss_err, mae, mse, tm

def mexican_hat(profile, index=0, widths=np.array([2])):
    wavelet = signal.ricker
    cwtmatr = signal.cwt(profile, widths = widths, wavelet=wavelet)
    return cwtmatr[index]

def find_max_peak(profile):
    peak_max_pos = signal.find_peaks_cwt(profile, widths = np.array([2]))
    res =  [(profile[n],n) for n in peak_max_pos]
    return res    
    
    
class Experiment():
    def __init__(self,
                 control_df=pd.DataFrame(),
                 drug_df=pd.DataFrame(),
                 tm=np.zeros(0) 
                 ):
        self.control_df = control_df
        self.drug_df = drug_df 
        self.tm=tm
        self.drug_fit = pd.DataFrame()
        self.control_fit = pd.DataFrame()
        self.peak_df = pd.DataFrame()
        self.selected_ids = set()
        
    def fit(self):
        drug_fit = pd.DataFrame()
        control_fit = pd.DataFrame() 
        for prot in self.control_df.index.values: 
            control_fit[prot]=find_fit(self.tm, self.control_df.loc[prot])
            drug_fit[prot]=find_fit(self.tm, self.drug_df.loc[prot])  

        columns = ['p', 'cov', 'infodict', 'mesg', 'ier', 'rsquared','ss_err', 'mae', 'mse', 'tm']
        drug_fit=drug_fit.T  
        drug_fit.columns = columns#[n+'_drug' for n in columns]
        control_fit=control_fit.T
        control_fit.columns = columns#[n+'_control' for n in columns]
        ###
        self.drug_fit = drug_fit
        self.control_fit = control_fit
        
    def peak_analysis(self):
        peak_df = pd.DataFrame()
        for prot in self.control_df.index.values:
            diff = self.drug_df.loc[prot] - self.control_df.loc[prot]
            diff = [n if n > 0.1 else 0 for n in diff]
            mexican_profile = mexican_hat(diff, index=0)
            mexican_profile = mexican_profile/mexican_profile.max()
            mexican_profile = [n if n > 0 else 0 for n in mexican_profile]           
            res_peak = find_max_peak(mexican_profile)
            if len(res_peak)>0:
                peak_position = res_peak[0][1]
                area = trapz(mexican_profile, dx=5)                
            else:
                peak_position, area = 0,0                
            peak_df[prot]=mexican_profile, peak_position, area
        peak_df = peak_df.T
        peak_df.columns = ['mexican_profile','peak_position','area']
        self.peak_df=peak_df
        
    def select(self):
        
        diff_tm = self.drug_fit['tm']-self.control_fit['tm']        
        diff_tm = diff_tm[diff_tm>0]
        diff_tm = diff_tm[diff_tm > diff_tm.median()]
        peak_area = self.peak_df['area']
        peak_area = peak_area[peak_area>0]        
        selected_ids = set(diff_tm.index.values) & set(peak_area.index.values)
    
        def apply_filters(df): #, 'mae'
            limit_mae = df['mae'].median()+df['mae'].std()
            limit_mse = df['mse'].median()+df['mse'].std()
            limit_ss_err = df['ss_err'].median()+df['ss_err'].std()
            limit_rsquared = 0.9
            df = df[ (df['mae']<limit_mae) & (df['mse']<limit_mse) &  (df['ss_err']<limit_ss_err) ]
            df = df[df['rsquared']>limit_rsquared]
            return df
              
        print 'limit_drug'
        select_drug = apply_filters(self.drug_fit.copy())
        select_control = apply_filters(self.control_fit.copy())
        
        selected_ids =  selected_ids & set(select_drug.index.values) &  set(select_control.index.values) 
        diff_tm = diff_tm.loc[selected_ids]
        diff_tm = diff_tm[diff_tm>diff_tm.median()]
        
        self.selected_ids = diff_tm.index.values
        return self.selected_ids

    def make_report(self, out_name='report.csv', selected=True):
        
        def format_df(df, tag):
            temp_df = df[['rsquared','ss_err', 'mae', 'mse', 'tm']]
            temp_df['x0'] = [n[0] for n in df['p']]
            temp_df['y0'] = [n[1] for n in df['p']]
            temp_df['c']  = [n[2] for n in df['p']]
            temp_df['k']  = [n[3] for n in df['p']]
            temp_df.columns = [n+'_'+tag for n in temp_df.columns]
            return temp_df
            
              
        df_list = [format_df(self.drug_fit,'drug'),
                   format_df(self.control_fit,'control'),
                   self.peak_df['peak_position'],
                   self.peak_df['area']]
        
        combine = pd.concat(df_list,1)
        combine['tm_diff']=combine['tm_drug']-combine['tm_control']
        if selected == True:
            combine=combine.loc[self.selected_ids]
        combine.to_csv(out_name)
        #return combine.head()

        #drug = self.drug_fit
        #drug.columns = [n+'_drug' for n in self.drug_fit]
        
        
            
    def plot(self, prot):
        x = self.tm
        x_fit = np.arange(x[0], x[-1], 0.1)
        y_drug = self.drug_df.loc[prot]
        drug_fit = self.drug_fit.loc[prot]

        y_control = self.control_df.loc[prot]
        control_fit = self.control_fit.loc[prot] 

        peak_res = self.peak_df.loc[prot]

        fig,ax = plt.subplots()
        ax.scatter(x, y_drug, c='r', label='drug')
        ax.plot(x_fit,  sigmoid(drug_fit['p'], x_fit), '-', c='r', label='drug fit')
        #ax.plot(x_fit,  sigmoid_derivative(drug_fit['p'], x_fit), '-', c='r', label='der drug fit')
        
        ax.scatter(x, y_control, c='g', label='control')
        ax.plot(x_fit,  sigmoid(control_fit['p'], x_fit), '-', c='g',label='control fit')
        
        #print x
        #print peak_res['mexican_profile']
        ax.plot(x, peak_res['mexican_profile'], ':', c='b', label='peak')

        ax.set_xticks(x)
        
        plt.legend(scatterpoints=1, loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title(prot)
        plt.show()
        print 'tm diff =', drug_fit['tm']- control_fit['tm']
        print 'peak TM =', x[peak_res['peak_position']]
        print 'peak area =', peak_res['area']
        scores = ['rsquared','ss_err', 'mae', 'mse']
        print 'score', 'drug', 'control'
        for score in scores:
            print score, drug_fit[score], control_fit[score]
    

        
    def quality_check(self):
        fig,ax=plt.subplots()
        self.drug_fit['rsquared'].plot(kind='kde',label='drug',ax=ax)
        self.control_fit['rsquared'].plot(kind='kde',label='control',ax=ax)
        plt.title('rsquared')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots()
        self.drug_fit['ss_err'].plot(kind='kde',label='drug',ax=ax)
        self.control_fit['ss_err'].plot(kind='kde',label='control',ax=ax)
        plt.title('ss_err')
        plt.legend()
        plt.show()
        
        fig,ax=plt.subplots()
        self.drug_fit['mae'].plot(kind='kde',label='drug',ax=ax)
        self.control_fit['mae'].plot(kind='kde',label='control',ax=ax)
        plt.title('mae')
        plt.legend()
        plt.show()        

        fig,ax=plt.subplots()
        self.drug_fit['mse'].plot(kind='kde',label='drug',ax=ax)
        self.control_fit['mse'].plot(kind='kde',label='control',ax=ax)
        plt.title('mse')
        plt.legend()
        plt.show()

        fig,ax=plt.subplots()
        diff = self.drug_fit['tm']-self.control_fit['tm']
        
        diff[diff>0].plot(kind='kde',label='>0',ax=ax)
        diff[diff<0].plot(kind='kde',label='<0',ax=ax)
        ax.set_xlim(-10,10)
        #self.control_fit['tm'].plot(kind='kde',label='control',ax=ax)
        plt.title('dif')
        plt.legend()
        plt.show() 

        fig,ax=plt.subplots()
        peak = self.peak_df['area']
        peak.plot(kind='kde',label='area',ax=ax)
        print diff.head()
        #self.control_fit['tm'].plot(kind='kde',label='control',ax=ax)
        plt.title('area')
        plt.legend()
        plt.show() 
         
                     
def main(drug_name='drug1.r1.csv', control_name='dmso.r1.csv'):           
    drug = pd.DataFrame.from_csv(drug_name)
    control = pd.DataFrame.from_csv(control_name)
    common = set(drug.index.values) & set(control.index.values)
    
    common = list(common)#[0:10]+['Q9Y3F4;Q9Y3F4-2;H0YH33','Q9BWD1;Q9BWD1-2']#list(common)[0:100]+
    #print common
    
    drug=drug.loc[common]
    control=control.loc[common]
    
    drug = drug.drop_duplicates()
    control = control.drop_duplicates()
    
    drug = drug.apply(norm_first,1)
    drug = drug.iloc[:,1:]
    drug = drug.apply(resize,axis=1)
    control = control.apply(norm_first,1)
    control = control.iloc[:,1:]
    control = control.apply(resize,axis=1)
    exp_1 = Experiment(control_df=control,
                     drug_df=drug,
                     tm=np.array(_TM[1:]))
    
    exp_1.fit()
    exp_1.peak_analysis()
    exp_1.quality_check()
    exp_1.plot('Q9Y3F4;Q9Y3F4-2;H0YH33')
    exp_1.plot('Q9BWD1;Q9BWD1-2')
    exp_1.select()
    exp_1.make_report(drug_name+'_'+control_name+'_report.csv',selected=True)
    print 'selected', len(exp_1.select())






if __name__ == '__main__':
    #main(drug_name='drug1.r1.csv', control_name='dmso.r1.csv')
    #main(drug_name='drug1.r2.csv', control_name='dmso.r2.csv')
    #main(drug_name='drug1.r3.csv', control_name='dmso.r3.csv')
    
    #main(drug_name='drug2.r1.csv', control_name='dmso.r1.csv')
    #main(drug_name='drug2.r2.csv', control_name='dmso.r2.csv')
    main(drug_name='drug2.r3.csv', control_name='dmso.r3.csv')
    
    
    
    
    
    '''
    #main(drug='drug1.r1.csv',control='dmso.r1.csv')
    drug = pd.DataFrame.from_csv('drug1.r1.csv')
    control = pd.DataFrame.from_csv('dmso.r1.csv')
    common = set(drug.index.values) & set(control.index.values)
    
    common = list(common)#[0:2]+['Q9Y3F4;Q9Y3F4-2;H0YH33']#list(common)[0:100]+
    #print common
    
    drug=drug.loc[common]
    control=control.loc[common]
    
    drug = drug.drop_duplicates()
    control = control.drop_duplicates()
    
    drug = drug.apply(norm_first,1)
    drug = drug.iloc[:,1:]
    drug = drug.apply(resize,axis=1)
    control = control.apply(norm_first,1)
    control = control.iloc[:,1:]
    control = control.apply(resize,axis=1)
    exp_1 = Experiment(control_df=control,
                     drug_df=drug,
                     tm=np.array(_TM[1:]))
    
    
    exp_1.fit()
    exp_1.peak_analysis()
    exp_1.select()
    exp_1.quality_check()
    exp_1.plot('Q9Y3F4;Q9Y3F4-2;H0YH33')
    exp_1.make_report('drug1.r1_dmso.r1_report.csv')
    #print 'selected', len(exp_1.select())
    '''
