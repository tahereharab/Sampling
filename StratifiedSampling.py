#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 11:56:06 2019

@author: tahereh
"""

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
from scipy.stats import entropy
from scipy.spatial import distance
import numpy as np


def computeAgeInitialFreq(original_df):

    stratified_rules = pd.read_csv('stratified_age_rules.csv', sep=",")
    stratified_rules['initial_freq'] = 0
    
    for rule in stratified_rules[:].itertuples(): 
        
        age_id = stratified_rules.loc[rule.Index, 'id']
        original_df_selected = original_df[(original_df['13'] == age_id)]
        
        initial_freq = len(original_df_selected)
        stratified_rules.loc[rule.Index, 'initial_freq'] = initial_freq

    stratified_rules.to_csv('stratified_age_rules.csv', index=False)


def computeGenderFinalFreq(stra_percentage, ideal_total):
     stratified_rules = pd.read_csv('stratified_gender_rules.csv', sep=",")
     stratified_rules['final_freq'] = 0
     stratified_rules['final_percentage'] = 0
     stratified_rules['final_freq'] = round(stratified_rules['initial_freq'] * stra_percentage)
     stratified_rules['final_percentage'] = round((stratified_rules['final_freq']/ideal_total)*100,3)
     stratified_rules.to_csv('stratified_gender_rules.csv', index=False)
     
     
def computeGenderInitialFreq(original_df):

    stratified_rules = pd.read_csv('stratified_gender_rules.csv', sep=",")
    stratified_rules['initial_freq'] = 0
    
    for rule in stratified_rules[:].itertuples(): 
        
        age_id = stratified_rules.loc[rule.Index, 'id']
        original_df_selected = original_df[(original_df['14'] == age_id)]
        
        initial_freq = len(original_df_selected)
        stratified_rules.loc[rule.Index, 'initial_freq'] = initial_freq

    stratified_rules.to_csv('stratified_gender_rules.csv', index=False)


def computeAgeFinalFreq(stra_percentage, ideal_total):
     stratified_rules = pd.read_csv('stratified_age_rules.csv', sep=",")
     stratified_rules['final_freq'] = 0
     stratified_rules['final_percentage'] = 0
     stratified_rules['final_freq'] = round(stratified_rules['initial_freq'] * stra_percentage)
     stratified_rules['final_percentage'] = round((stratified_rules['final_freq']/ideal_total)*100,3)
     stratified_rules.to_csv('stratified_age_rules.csv', index=False)     


def computeStrataInitialFreq(original_df):

    stratified_rules = pd.read_csv('stratified_rules.csv', sep=",")
    stratified_rules['initial_freq'] = 0
    
    for rule in stratified_rules[:].itertuples(): 
        
        age_id = stratified_rules.loc[rule.Index, 'age_id']
        gender_id = stratified_rules.loc[rule.Index, 'gender_id']
        original_df_selected = original_df[(original_df['13'] == age_id) & (original_df['14'] == gender_id)]
        
        initial_freq = len(original_df_selected)
        stratified_rules.loc[rule.Index, 'initial_freq'] = initial_freq

    stratified_rules.to_csv('stratified_rules.csv', index=False)


def computeStrataFinalFreq(stra_percentage, ideal_total):
     stratified_rules = pd.read_csv('stratified_rules.csv', sep=",")
     stratified_rules['final_freq'] = 0
     stratified_rules['final_percentage'] = 0
     stratified_rules['final_freq'] = round(stratified_rules['initial_freq'] * stra_percentage)
     stratified_rules['final_percentage'] = round((stratified_rules['final_freq']/ideal_total)*100,3)
     stratified_rules.to_csv('stratified_rules.csv', index=False)


def edulideanDistance():
    diversity_rules = pd.read_csv('stratified_rules.csv', sep=",")
    ideal = diversity_rules['joint_ideal_percentage'] 
    final = diversity_rules['final_percentage']
    dist = distance.euclidean(ideal/100, final/100)    
    #print('Euclidean Distance = ', round(dist,2))
    return round(dist,3)

def cosineDistance():
    from scipy import spatial
    diversity_rules = pd.read_csv('stratified_rules.csv', sep=",")
    ideal = diversity_rules['joint_ideal_percentage']
    final = diversity_rules['final_percentage']
    dist = spatial.distance.cosine(final/100, ideal/100)
    return round(dist,3)

def shannonEntropy():
    pk = []
    diversity_rules = pd.read_csv('stratified_rules.csv', sep=",") 
    final = diversity_rules['final_percentage']
    for f in final:
        pk.append(f/100)
        
    ent = entropy(pk)
    return ent    

def computeJointIdealPercentage(original_df, diversity_rules, rule_file):
    
    diversity_rules['joint_ideal_percentage'] = 0
    diversity_age_rules = pd.read_csv('stratified_age_rules.csv', sep=",")
    diversity_gender_rules = pd.read_csv('stratified_gender_rules.csv', sep=",")

    
    rule_ids = list(diversity_rules['id'])
    for rule_id in rule_ids:
        
        rule_age_id = diversity_rules[(diversity_rules['id'] == rule_id)]['age_id'].iloc[0]
        rule_gender_id = diversity_rules[(diversity_rules['id'] == rule_id)]['gender_id'].iloc[0]
           
        age_ideal = diversity_age_rules[(diversity_age_rules['id'] == rule_age_id)]['ideal_percentage'].iloc[0]
        gender_ideal = diversity_gender_rules[(diversity_gender_rules['id'] == rule_gender_id)]['ideal_percentage'].iloc[0]
        diversity_rules.loc[(diversity_rules['id'] == rule_id), 'joint_ideal_percentage'] = (age_ideal/100* gender_ideal/100)*100

    count = diversity_rules.loc[((diversity_rules['initial_percentage'] == 0) & (diversity_rules['joint_ideal_percentage'] != 0)), 'joint_ideal_percentage'].count()
    if count != 0:
        diversity_rules = divideIdealPercentageToOtherGroups(diversity_rules, count)
    #-----------
    diversity_rules.to_csv(rule_file, index=False)    
    return diversity_rules

def divideIdealPercentageToOtherGroups(diversity_rules, count):
    
    sum_ideal = diversity_rules.loc[((diversity_rules['initial_percentage'] == 0) & (diversity_rules['joint_ideal_percentage'] != 0)), 'joint_ideal_percentage'].sum()
    diversity_rules.loc[((diversity_rules['initial_percentage'] == 0) & (diversity_rules['joint_ideal_percentage'] != 0)), 'joint_ideal_percentage'] = 0  
    # divide value of the group porportional to the percentage of other groups
    selected_df = diversity_rules[(diversity_rules['initial_percentage'] != 0) & (diversity_rules['joint_ideal_percentage'] != 0)]
    for rule in selected_df[:].itertuples(): 
        joint_ideal_perc = selected_df.loc[rule.Index, 'joint_ideal_percentage']
        portion = sum_ideal * (joint_ideal_perc/(100 - sum_ideal))
        diversity_rules.loc[rule.Index, 'joint_ideal_percentage'] = joint_ideal_perc + portion
    return diversity_rules

def computeGenderIdealPercentageEnummerations():
    #this is only for gender ideal range example
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_gender.csv', sep=",")
    ideal_percentage_sumup_to_100 = []
    
    ideal_min_percentages = diversity_gender_rules['ideal_min_percentage']
    ideal_max_percentages = diversity_gender_rules['ideal_max_percentage']


    for i in np.arange(ideal_min_percentages[0], ideal_max_percentages[0]+1, 0.25):
        for j in np.arange(ideal_min_percentages[1], ideal_max_percentages[1]+1,0.25):
            for k in np.arange(ideal_min_percentages[2], ideal_max_percentages[2]+1,0.25):
                    if i + j + k == 100:
                        ideal_percentage_sumup_to_100.append([i,j,k])
                        
    return ideal_percentage_sumup_to_100


def computeAgeIdealPercentageEnummerations():
    #this is only for gender ideal range example
    diversity_age_rules = pd.read_csv('stratified_age_rules.csv', sep=",")
    ideal_percentage_sumup_to_100 = []
    
    ideal_min_percentages = diversity_age_rules['ideal_min_percentage']
    ideal_max_percentages = diversity_age_rules['ideal_max_percentage']
    
    for i in range(ideal_min_percentages[0], ideal_max_percentages[0]+1):
        for j in range(ideal_min_percentages[1], ideal_max_percentages[1]+1):
            for k in range(ideal_min_percentages[2], ideal_max_percentages[2]+1):
                for l in range(ideal_min_percentages[3], ideal_max_percentages[3]+1):
                    for m in range(ideal_min_percentages[4], ideal_max_percentages[4]+1):
                        for n in range(ideal_min_percentages[5], ideal_max_percentages[5]+1):
                            for o in range(ideal_min_percentages[6], ideal_max_percentages[6]+1):
                                
                                if i + j + k + l + m + n + o == 100:
                                    ideal_percentage_sumup_to_100.append([i,j,k,l,m,n,o])
    
    return ideal_percentage_sumup_to_100



def setGenderIdealPercentageFromIdealRange(enummeration):
    #this is only for gender ideal range 
    diversity_gender_rules = pd.read_csv('stratified_gender_rules.csv', sep=",")
    diversity_gender_rules['ideal_percentage'] = 0
    diversity_gender_rules['ideal_percentage'] = enummeration
    diversity_gender_rules.to_csv('stratified_gender_rules.csv', index=False) 
    

def setAgeIdealPercentageFromIdealRange(enummeration):
    #this is only for age ideal range 
    diversity_gender_rules = pd.read_csv('stratified_age_rules.csv', sep=",")
    diversity_gender_rules['ideal_percentage'] = 0
    diversity_gender_rules['ideal_percentage'] = enummeration
    diversity_gender_rules.to_csv('stratified_age_rules.csv', index=False) 

if __name__ == "__main__": 
    
    is_ideal_range = True
    original_df = pd.read_csv('PSLSurvey.csv', sep=",")
    original_df = original_df[(original_df['13'] != -99) & (original_df['14'] != -99)]
    original_df.loc[(original_df['14'] >= 3), '14'] = 3    
    ideal_total = 100
    total_size = len(original_df)
    stra_percentage = ideal_total / total_size
    rule_file = 'stratified_rules.csv'
    diversity_rules = pd.read_csv(rule_file, sep=",")
    computeStrataInitialFreq(original_df)
    computeStrataFinalFreq(stra_percentage, ideal_total)
    
    #******************************
    if is_ideal_range == False:  #when running this, use the original dim0, dim1, dim2, dim3 with correct ideal percentages
    #******************************  
        computeJointIdealPercentage(original_df, diversity_rules, rule_file)
        eud = cosineDistance()
        print('Distance = ', eud) 
        
    else: # when both have ranges 
            
            gender_enums = computeGenderIdealPercentageEnummerations()
            age_enums = computeAgeIdealPercentageEnummerations()
            min_dis = 100
            
            print(len(gender_enums), len(age_enums))
            i = 0
            for aenum in age_enums:
                for genum in gender_enums:
                    
                    i = i+1
                    print(i)
                    setGenderIdealPercentageFromIdealRange(genum)
                    setAgeIdealPercentageFromIdealRange(aenum)    
                    
                    computeJointIdealPercentage(original_df, diversity_rules, rule_file)
    
                    dis = cosineDistance()
                                
                    if dis < min_dis:
                        min_dis = dis
                        print('min Distance = ', min_dis)
                        print('aenum enummeration: ', aenum)
                        print('genum enummeration: ', genum)   
                        #entro = shannonEntropy()
                        #print('entropy = ', round(entro,3))

   
    
    
    