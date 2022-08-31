#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:16:56 2019

@author: tahereh
"""
import pandas as pd
import numpy as np
from scipy.spatial import distance
from scipy.stats import wasserstein_distance, entropy
from scipy import spatial
import csv


def computeFinalPercentage(weighted_df, random_selected_df, diversity_rules , rule_file, dimension):
  
    selected_rows = weighted_df[weighted_df['ID'].isin(random_selected_df)]  
    diversity_rules['final_freq'] = 0
    diversity_rules['final_percentage'] = 0
    
    for rule in diversity_rules[:].itertuples(): 
        if dimension == 0:
            group_id = diversity_rules.loc[rule.Index, 'id']
            original_df_selected = selected_rows[(selected_rows['13'] == group_id)]
        
        elif dimension == 1:
            group_id = diversity_rules.loc[rule.Index, 'id']    
            if group_id == 3:
                original_df_selected = selected_rows[(selected_rows['14'] == group_id) | (selected_rows['14'] == group_id+1) | (selected_rows['14'] == group_id+2)] 
            else:           
                original_df_selected = selected_rows[(selected_rows['14'] == group_id)]
        else:    
            age_id = diversity_rules.loc[rule.Index, 'age_id']
            gender_id = diversity_rules.loc[rule.Index, 'gender_id']
            original_df_selected = selected_rows[(selected_rows['13'] == age_id) & (selected_rows['14'] == gender_id)]
            
        final_freq = len(original_df_selected)
        diversity_rules.loc[rule.Index, 'final_freq'] = final_freq

    diversity_rules['final_percentage'] = round((diversity_rules['final_freq']/ideal_total)*100,2)
    diversity_rules.to_csv(rule_file, index=False)
     

def assignWeightsToCandidates(original_df, diversity_rules, dimension):
    
    original_df['weight'] = 0
    
    rule_ids = list(diversity_rules['id'])
    for rule_id in rule_ids:
        rule_weight = diversity_rules[(diversity_rules['id'] == rule_id)]['weight'].iloc[0]
        
        if dimension == 0:
            original_df.loc[original_df['13'] == rule_id, 'weight'] = rule_weight
        elif dimension == 1:   
            
            if rule_id == 3:
                original_df.loc[(original_df['14'] == rule_id) | (original_df['14'] == rule_id+1) | (original_df['14'] == rule_id+2), 'weight'] = rule_weight
            else:    
                original_df.loc[original_df['14'] == rule_id, 'weight'] = rule_weight
            
        else:    
            rule_age_id = diversity_rules[(diversity_rules['id'] == rule_id)]['age_id'].iloc[0]
            rule_gender_id = diversity_rules[(diversity_rules['id'] == rule_id)]['gender_id'].iloc[0]
            original_df.loc[(original_df['13'] == rule_age_id) & (original_df['14'] == rule_gender_id), 'weight'] = rule_weight                    
            
    original_df.to_csv('PSLSurvey.csv', index=False)    
    return original_df

def assignJointWeightsToCandidates(original_df, diversity_rules, dimension):
    
    original_df['weight'] = 0
    
    rule_ids = list(diversity_rules['id'])
    for rule_id in rule_ids:
        
        rule_weight = diversity_rules[(diversity_rules['id'] == rule_id)]['jointweight'].iloc[0]
        rule_age_id = diversity_rules[(diversity_rules['id'] == rule_id)]['age_id'].iloc[0]
        rule_gender_id = diversity_rules[(diversity_rules['id'] == rule_id)]['gender_id'].iloc[0]
        original_df.loc[(original_df['13'] == rule_age_id) & (original_df['14'] == rule_gender_id), 'weight'] = rule_weight                   
            
    original_df.to_csv('PSLSurvey.csv', index=False)    
    return original_df
    

def computeDiversityWeights(diversity_rules, ideal_total, rule_file):
    
    total_size = diversity_rules['initial_freq'].sum()

    try:
        diversity_rules['weight'] = (diversity_rules['ideal_percentage'] / diversity_rules['initial_percentage']) * (1/total_size)
     
    except ZeroDivisionError:
        print('error')
    
    diversity_rules = diversity_rules.replace([np.inf, -np.inf], np.nan)
    diversity_rules['weight'].fillna(0, inplace=True)
    diversity_rules.to_csv(rule_file, index=False)
    
    return diversity_rules

def computeDiversityWeightsJoint(diversity_rules, ideal_total, rule_file):
    
    total_size = diversity_rules['initial_freq'].sum() 
     
    try:
        diversity_rules['jointweight'] = (diversity_rules['joint_ideal_percentage'] / diversity_rules['initial_percentage']) * (1/total_size)
    except ZeroDivisionError:
        print('error')
    
    diversity_rules = diversity_rules.replace([np.inf, -np.inf], np.nan)
    diversity_rules['jointweight'].fillna(0, inplace=True)
    diversity_rules.to_csv(rule_file, index=False)
    
    return diversity_rules


def computeDiversityWeightsJointOld(diversity_rules, ideal_total, rule_file):
    
    diversity_rules['jointweight'] = 0
    diversity_age_rules = pd.read_csv('PSL_diversity_rule_age.csv', sep=",")
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_gender.csv', sep=",")
    total_size = diversity_rules['initial_freq'].sum()
    
    rule_ids = list(diversity_rules['id'])
    for rule_id in rule_ids:
        
        rule_age_id = diversity_rules[(diversity_rules['id'] == rule_id)]['age_id'].iloc[0]
        rule_gender_id = diversity_rules[(diversity_rules['id'] == rule_id)]['gender_id'].iloc[0]
        
        age_weight = diversity_age_rules[(diversity_age_rules['id'] == rule_age_id)]['weight'].iloc[0]
        gender_weight = diversity_gender_rules[(diversity_gender_rules['id'] == rule_gender_id)]['weight'].iloc[0]
        diversity_rules.loc[(diversity_rules['id'] == rule_id), 'jointweight'] = ((age_weight * gender_weight)*(total_size))
        
    diversity_rules.to_csv('PSL_diversity_rules.csv', index=False)    
    return diversity_rules


def computeInitialFreqPercentage(original_df, diversity_rules, rule_file, dimension):

    diversity_rules['initial_freq'] = 0
    diversity_rules['initial_percentage'] = 0
    
    for rule in diversity_rules[:].itertuples(): 
        if dimension == 0:
            group_id = diversity_rules.loc[rule.Index, 'id']
            original_df_selected = original_df[(original_df['13'] == group_id)]
        
        elif dimension == 1:
            group_id = diversity_rules.loc[rule.Index, 'id']
            
            if group_id == 3:
                original_df_selected = original_df[(original_df['14'] == group_id) | (original_df['14'] == group_id+1) | (original_df['14'] == group_id+2)]
                
            else:
                original_df_selected = original_df[(original_df['14'] == group_id)]
                
        else:
            age_id = diversity_rules.loc[rule.Index, 'age_id']
            gender_id = diversity_rules.loc[rule.Index, 'gender_id']
            original_df_selected = original_df[(original_df['13'] == age_id) & (original_df['14'] == gender_id)]
            
        initial_freq = len(original_df_selected)
        diversity_rules.loc[rule.Index, 'initial_freq'] = initial_freq
    
    diversity_rules['initial_percentage'] = (diversity_rules['initial_freq'] / len(original_df))*100
    diversity_rules.to_csv(rule_file, index=False)
    return diversity_rules   

def computeJointIdealPercentage(original_df, diversity_rules, rule_file):
    
    diversity_rules['joint_ideal_percentage'] = 0
    diversity_age_rules = pd.read_csv('PSL_diversity_rule_age.csv', sep=",")
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_gender.csv', sep=",")

    rule_ids = list(diversity_rules['id'])
    for rule_id in rule_ids:
        
        rule_age_id = diversity_rules[(diversity_rules['id'] == rule_id)]['age_id'].iloc[0]
        rule_gender_id = diversity_rules[(diversity_rules['id'] == rule_id)]['gender_id'].iloc[0]
           
        age_ideal = diversity_age_rules[(diversity_age_rules['id'] == rule_age_id)]['ideal_percentage'].iloc[0]
        gender_ideal = diversity_gender_rules[(diversity_gender_rules['id'] == rule_gender_id)]['ideal_percentage'].iloc[0]

        diversity_rules.loc[(diversity_rules['id'] == rule_id), 'joint_ideal_percentage'] = (age_ideal/100* gender_ideal/100)*100
        
    #-----------    
    count = diversity_rules.loc[((diversity_rules['initial_percentage'] == 0) & (diversity_rules['joint_ideal_percentage'] != 0)), 'joint_ideal_percentage'].count()
    if count != 0:
        diversity_rules = divideIdealPercentageToOtherGroups(diversity_rules, count)  
    #-----------
    diversity_rules.to_csv('PSL_diversity_rules.csv', index=False)    
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

def computeJointInitialPercentage(original_df, diversity_rules, rule_file):
    
    diversity_rules['joint_initial_percentage'] = 0
    diversity_age_rules = pd.read_csv('PSL_diversity_rule_age.csv', sep=",")
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_gender.csv', sep=",")

    
    rule_ids = list(diversity_rules['id'])
    for rule_id in rule_ids:
        
        rule_age_id = diversity_rules[(diversity_rules['id'] == rule_id)]['age_id'].iloc[0]
        rule_gender_id = diversity_rules[(diversity_rules['id'] == rule_id)]['gender_id'].iloc[0]
           
        age_initial = diversity_age_rules[(diversity_age_rules['id'] == rule_age_id)]['initial_percentage'].iloc[0]
        gender_initial = diversity_gender_rules[(diversity_gender_rules['id'] == rule_gender_id)]['initial_percentage'].iloc[0]
        diversity_rules.loc[(diversity_rules['id'] == rule_id), 'joint_initial_percentage'] = (age_initial* gender_initial)/100
        
    diversity_rules.to_csv('PSL_diversity_rules.csv', index=False)    
    return diversity_rules

def randomSelection(weighted_df, ideal_total):
    
    probabilities = weighted_df['weight']
    ids = weighted_df['ID']
    random_selection = np.random.choice(ids, ideal_total, p = list(probabilities), replace=False)
    return random_selection


def finalWeights(original_df, diversity_rules, rule_file, ideal_total, dimension):
    
    diversity_rules = computeDiversityWeights(diversity_rules, ideal_total, rule_file)
    weighted_df = assignWeightsToCandidates(original_df, diversity_rules, dimension)
    return weighted_df


def finalWeightsJoint(original_df, diversity_rules, rule_file, ideal_total, dimension):
    
    diversity_rules = computeJointIdealPercentage(original_df, diversity_rules, rule_file)
    diversity_rules = computeDiversityWeightsJoint(diversity_rules, ideal_total, rule_file)
    weighted_df = assignJointWeightsToCandidates(original_df, diversity_rules, dimension)
    return weighted_df


def finalWeightsJointOld(original_df, diversity_rules, rule_file, ideal_total, dimension):
    
    diversity_rules = computeDiversityWeightsJointOld(diversity_rules, ideal_total, rule_file)
    weighted_df = assignJointWeightsToCandidates(original_df, diversity_rules, dimension)
    return weighted_df


def EMD(rule_file, dimension):
    diversity_rules = pd.read_csv(rule_file, sep=",")
    if dimension == 2:
        dist1 = diversity_rules['joint_ideal_percentage']
    else:
        dist1 = diversity_rules['ideal_percentage']
        
    dist2 = diversity_rules['final_percentage']
    distance = round(wasserstein_distance(dist1, dist2), 2)
    return distance
    

def computeEntropy(random_selected_df):
    selected_rows = weighted_df[weighted_df['ID'].isin(random_selected_df)]  
    probabilities = selected_rows['weight']
    entr = entropy(probabilities)
    print('entropy = ', entr) 


def computeAvgFinalPercentage(rule_file, df, count):
    diversity_rules = pd.read_csv(rule_file, sep=",")
    col_name = 'final_percentage'+ str(count)
    df[col_name] = diversity_rules['final_percentage']
    return df
  
def computeGenderIdealPercentageEnummerations():
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_gender.csv', sep=",")
    ideal_percentage_sumup_to_100 = []
    
    ideal_min_percentages = diversity_gender_rules['ideal_min_percentage']
    ideal_max_percentages = diversity_gender_rules['ideal_max_percentage']

    for i in np.arange(0, 101, 0.25):
        for j in np.arange(0, 101,0.25):
            for k in np.arange(ideal_min_percentages[2], ideal_max_percentages[2]+1,0.25):
                    if (i + j + k == 100) & (j == 2*i) :
                        ideal_percentage_sumup_to_100.append([i,j,k])                     
                        
    print(ideal_percentage_sumup_to_100)
    return ideal_percentage_sumup_to_100


def computeAgeIdealPercentageEnummerations():
    #this is only for gender ideal range example
    diversity_age_rules = pd.read_csv('PSL_diversity_rule_age.csv', sep=",")
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
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_gender.csv', sep=",")
    diversity_gender_rules['ideal_percentage'] = 0
    diversity_gender_rules['ideal_percentage'] = enummeration
    diversity_gender_rules.to_csv('PSL_diversity_rule_gender.csv', index=False) 
    

def setAgeIdealPercentageFromIdealRange(enummeration):
    #this is only for age ideal range 
    diversity_gender_rules = pd.read_csv('PSL_diversity_rule_age.csv', sep=",")
    diversity_gender_rules['ideal_percentage'] = 0
    diversity_gender_rules['ideal_percentage'] = enummeration
    diversity_gender_rules.to_csv('PSL_diversity_rule_age.csv', index=False) 
    

def edulideanDistance(rule_file, dimension):
    diversity_rules = pd.read_csv(rule_file, sep=",")
    if dimension == 2:
        ideal = diversity_rules['joint_ideal_percentage']
    else:
        ideal = diversity_rules['ideal_percentage']
        
    final = diversity_rules['final_percentage']
    dist = distance.euclidean(ideal/100, final/100)    
    return round(dist,3)

def cosineDistance(rule_file, dimension):

    diversity_rules = pd.read_csv(rule_file, sep=",")
    if dimension == 2:
        ideal = diversity_rules['joint_ideal_percentage']
    else:
        ideal = diversity_rules['ideal_percentage']
        
    final = diversity_rules['final_percentage']
    dist = spatial.distance.cosine(final/100, ideal/100)
    return round(dist,3)

def cosineDistance2(rule_file):

    diversity_rules = pd.read_csv(rule_file, sep=",")
    
    ideal = diversity_rules['joint_ideal_percentage']
    initial = diversity_rules['initial_percentage']
    dist = spatial.distance.cosine(initial/100, ideal/100)
    return round(dist,3)
    
def shannonEntropy(rule_file):
    pk = []
    diversity_rules = pd.read_csv(rule_file, sep=",") 
    final = diversity_rules['final_percentage']
    for f in final:
        pk.append(f/100)
        
    ent = entropy(pk)
    return ent


def kl_divergence():
    p =[]
    q = []
    diversity_rules = pd.read_csv('PSL_diversity_rules.csv', sep=",") 
    final = diversity_rules['final_percentage']
    ideal =  diversity_rules['joint_ideal_percentage']
    
    for f in final:
        p.append(f/100)
        
    for i in ideal:
        q.append(i/100)    
    
    
    kl = 0
    for i in range(len(p)):
        if p[i] != 0:
            kl = kl +  (p[i] * np.log(p[i]/q[i]))
    return round(kl,3)        
    #OR
    """
    p = np.asarray(p, dtype=np.float)
    q = np.asarray(q, dtype=np.float)

    return round(np.sum(np.where(p != 0, p * np.log(p / q), 0)),3)
    """
    
#--------------------------------------------------------------------------------------------
if __name__ == "__main__": 
    
    dimension = 2  #0=age, 1=gender, 2=age-gender
    is_ideal_range = False
    is_Nout_unknown = False
    
    
    if dimension == 0:
        rule_file = 'PSL_diversity_rule_age.csv'
    elif dimension == 1:    
        rule_file = 'PSL_diversity_rule_gender.csv'
    else:    
        rule_file = 'PSL_diversity_rules.csv'
        
    input_file = 'PSLSurvey.csv'
    diversity_rules = pd.read_csv(rule_file, sep=",")
    original_df = pd.read_csv(input_file, sep=",", encoding = "utf-8")
    original_df = original_df[(original_df['13'] != -99) & (original_df['14'] != -99)]
    print(len(original_df))
    #---------------------------
    original_df.loc[(original_df['14'] >= 3), '14'] = 3      
    ideal_total = 100
    #---------------------------
    iteration = 100
    distance_sum = 0
    kld_sum = 0
    entro_sum = 0
    
    if dimension == 0:
        index_range = 7
    elif dimension == 1:
        index_range = 2
    else:
        index_range = 21

    final_perc_avg_df = pd.DataFrame(index=range(index_range)) 
    diversity_rules = computeInitialFreqPercentage(original_df, diversity_rules, rule_file, dimension)
    #---------------------------
    #---------------------------
    if (is_ideal_range == False) & (is_Nout_unknown == True):
        
        min_EMD = 100
        max_EMD = -100
        print(len(original_df))
        
        for ideal_total in range(1, len(original_df)+1):
            
            distance_sum = 0
        
            if (dimension == 0) | (dimension == 1) :
                    weighted_df = finalWeights(original_df, diversity_rules, rule_file, ideal_total, dimension) #when age-gender percentages are given
                
            else:
                    weighted_df = finalWeightsJoint(original_df, diversity_rules, rule_file, ideal_total, dimension) #when given any ideal percentages, we use joint probabilities of given ideal percentages for each group
                    
            for count in range(0,iteration):
                    
                random_selected_df = randomSelection(weighted_df, ideal_total)
                computeFinalPercentage(weighted_df, random_selected_df, diversity_rules, rule_file, dimension)
                
                distance = EMD(rule_file, dimension)
                distance_sum = distance_sum + distance
                
            EMD_avg = round(distance_sum / iteration,2)
            
            with open('minmax_Nout.csv','a') as file:
                writer = csv.writer(file)
                writer.writerow([ideal_total , EMD_avg])
            
            if EMD_avg < min_EMD:
                min_EMD = EMD_avg
                print('min EMD_average for N_out = ', ideal_total, 'is ', EMD_avg)
                
            if EMD_avg > max_EMD:
                max_EMD = EMD_avg
                print('max EMD_average for N_out = ', ideal_total, 'is ', EMD_avg)
                
        
        print('min= ', min_EMD, 'max= ' , max_EMD)
        
    #---------------------------
    #---------------------------
    if (is_ideal_range == False) & (is_Nout_unknown == False):
        
        if (dimension == 0) | (dimension == 1) : # | (dimension == 1) ##when age-gender percentages are given by census or uniform
                weighted_df = finalWeights(original_df, diversity_rules, rule_file, ideal_total, dimension) 
            
        else:
                #weighted_df = finalWeightsJointOld(original_df, diversity_rules, rule_file, ideal_total, dimension) #the formula was incorrect so stopped using it. when age percentages and gender percentages are given separately 
                weighted_df = finalWeightsJoint(original_df, diversity_rules, rule_file, ideal_total, dimension) #when given any ideal percentages, we use joint probabilities of given ideal percentages for each group
                
        for count in range(0,iteration):
                
            random_selected_df = randomSelection(weighted_df, ideal_total)
            computeFinalPercentage(weighted_df, random_selected_df, diversity_rules, rule_file, dimension)
            distance = cosineDistance(rule_file, dimension)
            distance_sum = distance_sum + distance
            kld = kl_divergence()
            kld_sum = kld_sum + kld
            final_perc_avg_df = computeAvgFinalPercentage(rule_file, final_perc_avg_df, count)
            
    
        final_perc_avg_df['mean'] = final_perc_avg_df.mean(axis=1)    
        cos_avg = round(distance_sum / iteration,3)
        cos_ini = cosineDistance2(rule_file)
    #---------------------------
    # range ideals
    #---------------------------
    # using enummerations 
    if (is_ideal_range == True) & (is_Nout_unknown == False): 
        
        both_dim_range = True
        
        #---- only when gender has range byt age has fixed ideals---------
        if both_dim_range == False:
            
            enums = computeGenderIdealPercentageEnummerations()
            min_EMD = 100
            
            for enum in enums:
                
                distance_sum = 0
                setGenderIdealPercentageFromIdealRange(enum)
                weighted_df = finalWeightsJoint(original_df, diversity_rules, rule_file, ideal_total, dimension)
                
                for count in range(0,iteration):
                    
                    random_selected_df = randomSelection(weighted_df, ideal_total)
                    computeFinalPercentage(weighted_df, random_selected_df, diversity_rules, rule_file, dimension)
                
                    distance = EMD(rule_file, dimension)
                    distance_sum = distance_sum + distance
                
                    final_perc_avg_df = computeAvgFinalPercentage(rule_file, final_perc_avg_df, count)
                    
                
                final_perc_avg_df['mean'] = final_perc_avg_df.mean(axis=1)    
                EMD_avg = round(distance_sum / iteration,2)
                #print('enummeration: ', enum, ' EMD_average = ', EMD_avg, '\n')
    
                if EMD_avg < min_EMD:
                    min_EMD = EMD_avg
                    print('min emd = ', min_EMD)
                    print('enummeration: ', enum)
                
            print(enums)
            print('min_EMD of all enummerations = ', min_EMD) 
        
        else: # when both have ranges 
            
            gender_enums = computeGenderIdealPercentageEnummerations()
            age_enums = computeAgeIdealPercentageEnummerations()
            min_EuD = 100
            
            print(len(gender_enums), len(age_enums))
            
            i = 0
            
            for aenum in age_enums:
                for genum in gender_enums:
                    
                    i = i  + 1
                    print(i)
                    
                    setGenderIdealPercentageFromIdealRange(genum)
                    setAgeIdealPercentageFromIdealRange(aenum)
                    weighted_df = finalWeightsJoint(original_df, diversity_rules, rule_file, ideal_total, dimension)
                    
                    distance_sum = 0
                    for count in range(0,iteration):
                        random_selected_df = randomSelection(weighted_df, ideal_total)
                        computeFinalPercentage(weighted_df, random_selected_df, diversity_rules, rule_file, dimension)
                    
                        #EuD = edulideanDistance(rule_file,dimension)
                        EuD = cosineDistance(rule_file,dimension)
                        distance_sum = distance_sum + EuD
                        #entro = shannonEntropy(rule_file)
                        #entro_sum = entro_sum  + entro
                    
                    cos_avg = round(distance_sum / iteration,3)
                    #entro_avg = round(entro_sum / iteration,3)
                    if cos_avg < min_EuD:
                        min_EuD = cos_avg
                        print('min Distance = ', min_EuD)
                        print('gender enummeration: ', genum)      
    #---------------------------
    #---------------------------
    
    
    