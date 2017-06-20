# -*- coding: utf-8 -*-
"""
Created on Fri Oct 07 11:05:49 2016

@author: mtinti
"""

import os
import numpy as np
import pandas as pd
from string import strip
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')

#maxuont data field
_QUANT_METHODS = ['Reporter intensity corrected ']

#experiment temperature range
_TM = [37,41,44,47,50,53,56,59,62,65]

#helps to keep track
#of proteins removed from dataset
def print_result(start_df_shape, shape_before, df, what):
    removed = shape_before[0]- df.shape[0]
    removed_from_beginning = start_df_shape[0]-df.shape[0]
    if removed > 0:
        print 'removed ',removed, what   
        print 'tot ', removed_from_beginning, ' entries removed' 
        print '---------------'
    else:
        print what
        print 'nothing removed'
        print '---------------'

#remove bad ids form maxquant
def clean(df, quant_methods, experiments):     
    before,start = df.shape,df.shape
    col = 'Only identified by site'
    df = df[df[col] != '+'] 
    print_result(start, before, df, col)

    before = df.shape
    col = 'Reverse'
    df = df[df[col] != '+']
    print_result(start, before, df, col)
        
    before = df.shape
    col = 'Potential contaminant'
    df = df[df[col] != '+']
    print_result(start, before, df, col)
    
    #fasta_headers = df['Fasta headers']
    #protein_name = [strip(s.split(';')[0].split(' ')[0]) for s in fasta_headers]
    #df['protein_name'] = protein_name
    cols_to_select = [ q+e for q in quant_methods for e in experiments]
    #df = df.set_index(['protein_name'])
    df = df[cols_to_select]
    print 'got: ', df.shape[0], 'protein now'
    
    before = df.shape
    df = df[(df.T != 0).any()]
    print_result(start, before, df, 'all zeros') 
    return df

#divide each element of the array
#by the sum of the elemets
def norm_sum(X):
    X = X/X.sum()
    return X

#divide each element of the array
#by the biggest element
def norm_max(X):
    X = X/X.max()
    return X

#divide each element of the array
#by the first element
def norm_first(X):
    X = X/X.values[0]
    return X
    
def norm_log(X):  
    return [np.log10(x) for x in X.values]    
   
#make plot of the dataframe   
def box_plot(df, norm_type):
    df=df.apply(norm_type,1)
    fig,ax = plt.subplots()
    df.plot(kind='box',ax=ax)
    ax.set_xticklabels(_TM)
    ax.set_xlabel('temperature')
    ax.set_ylabel('quantification')
    return fig
        
def make_input_r(df, out_name):
    rcols = ['rel_fc_131L','rel_fc_130H','rel_fc_130L','rel_fc_129H',
             'rel_fc_129L','rel_fc_128H','rel_fc_128L','rel_fc_127H',
             'rel_fc_127L','rel_fc_126']   
    df = df.apply(norm_first,1)
    df.columns = rcols
    df['gene_name']=df.index.values
    df['qssm'] = 6
    df['qupm'] = 6
    df = df[['gene_name','qssm','qupm']+rcols]
    df.to_csv(out_name,index=False)     
    

def get_data(tag, df):
    experiment = [str(n)+' '+tag for n in np.arange(0,10,1)]
    df = clean(df, _QUANT_METHODS, experiment)
    df.columns = _TM
    return df
    
     
if __name__ == '__main__':
    
    

    file_name = os.path.join('in_data','proteinGroups.txt')
    df = pd.DataFrame.from_csv(file_name, sep='\t')
    #tags = ['dmso.r1','dmso.r2','dmso.r3','drug1.r1','drug1.r2','drug1.r3']
    tags = ['drug2.r1','drug2.r2','drug2.r3']
    for tag in tags:
        temp_df = get_data(tag, df)
        temp_df.to_csv(tag+'.csv')
        box_plot(temp_df, norm_sum)
        plt.ylim(0,0.3)
    
    fig,ax=plt.subplots()
    for tag in tags:
        temp_df = get_data(tag, df)
        ax.plot(_TM, temp_df.median(),label=tag)
    plt.legend()
    plt.show()
    
    
    
    
    

    
