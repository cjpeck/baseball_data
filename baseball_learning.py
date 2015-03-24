# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 11:59:25 2015

@author: cjpeck
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import scipy as sp
import scipy.stats

class BaseballData(object):
    
    def __init__(self):
        self.tFrame = [1960, 2014]        
        self.directory = '/Users/cjpeck/Dropbox/spyder2/baseball/mydata/'       
        
    def create_df(self):                
        
        # master data
        master_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Master.csv'
        ms = pd.DataFrame.from_csv(master_fname)
        # batting data
        batting_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Batting.csv'
        bt = pd.DataFrame.from_csv(batting_fname)
        bt['AVG'] = bt['H'] / bt['AB']
        bt['SLG'] = (bt['H'] + 1*bt['2B'] + 2*bt['3B'] + 3*bt['HR']) / bt['AB']
        # pitching data
        pitching_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Pitching.csv'
        pt = pd.DataFrame.from_csv(pitching_fname)
        # fielding data
        fielding_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Fielding.csv'
        fd = pd.DataFrame.from_csv(fielding_fname)
        # team data
        team_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Teams.csv'
        tm = pd.DataFrame.from_csv(team_fname)
                                 
        # years of interest
        years = list(range(self.tFrame[0], self.tFrame[1]+1))
        # excluding strike years, years before 162 games
        indices = [[], []]
        remove = []
        for year in years:
            tm_year = tm[str(year)]
            if tm_year['G'].mean() < 160:
                print('skipping year ' + str(year))
                remove.append(year)
                continue
            teams = tm_year['teamID']
            indices[0].extend([year for _ in teams])
            indices[1].extend([team for team in teams])
        for year in remove:
            years.remove(year)
        
        # data types of interest
        tm_keys = ['G', 'W', 'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin']
        bt_keys = ['AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 
                   'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
        pt_keys = ['CG', 'SHO', 'SV', 'IPouts', 'H', 'ER', 'HR', 'BB', 
                   'SO',  'BFP', 'R']
        fd_keys = ['E', 'SB', 'CS']
        keys = [x + '_bt' for x in bt_keys]
        keys += [x + '_pt' for x in pt_keys]
        keys += [x + '_fd' for x in fd_keys]
        
        # create the DataFrame
        self.dfx = pd.DataFrame(index=indices, columns=keys)
        self.dfy = pd.DataFrame(index=indices, columns=tm_keys)  
        for year in years:
            tm_year = tm[str(year)]
            bt_year = bt[bt['yearID']==year]
            pt_year = pt[pt['yearID']==year]
            fd_year = fd[fd['yearID']==year]
            teams = tm_year['teamID']            
            for team in teams:                
                # team results            
                team_data = tm_year[tm_year['teamID']==team]
                s = pd.Series(team_data[tm_keys].sum(skipna=True))  
                self.dfy.ix[(year, team)] = s                                                    
                #append batting information            
                team_data = bt_year[bt_year['teamID']==team]
                s = pd.Series(team_data[bt_keys].sum(skipna=True))  
                s.index = [s.index[i] + '_bt' for i in range(len(s))]
                count = len(s)                
                #append pitching information
                team_data = pt_year[pt_year['teamID']==team]
                s = s.append(team_data[pt_keys].sum(skipna=True))  
                s.index = [s.index[i] + '_pt' if i >= count else s.index[i] 
                           for i in range(len(s))]
                count = len(s)                
                #append fielding information
                team_data = fd_year[fd_year['teamID']==team]
                s = s.append(team_data[fd_keys].sum(skipna=True))  
                s.index = [s.index[i] + '_fd' if i >= count else s.index[i] 
                           for i in range(len(s))]
                count = len(s) 
                #append to dataframe
                self.dfx.ix[(year, team)] = s
                
    def get_predictors(self):
        ''' Return dataframe for doing the analysis 
            
            Some less informative correlations:
            -'AB_bt' is negatively correlated with wins
            -'IPouts_pt' is positively correlated with wins
            -'SV_pt' is positively correlated with wins
            -'BFP_pt' is positively correlated with wins
            ** most stats should be considered as rate per 'AB' or 'G'            
        '''
        pass
    
    def save_df(self):
        with open(self.directory + 'predictors.pickle', 'wb') as f:        
            pickle.dump(self.dfx, f, pickle.HIGHEST_PROTOCOL)
        with open(self.directory + 'responses.pickle', 'wb') as f:        
            pickle.dump(self.dfy, f, pickle.HIGHEST_PROTOCOL)
            
    def load_df(self):
        with open(self.directory + 'predictors.pickle', 'rb') as f:        
            self.dfx = pickle.load(f)
        with open(self.directory + 'responses.pickle', 'rb') as f:        
            self.dfy = pickle.load(f)
            
    def normalize_predictors(self):
        #normalize
        self.dfx = \
            self.dfx.apply(lambda x: (x - self.dfx.ix[x.name[0]].mean()) / 
                                      self.dfx.ix[x.name[0]].std(), axis=1)
                                      
if __name__ == '__main__':
    b = BaseballData()
    b.create_df()    
    b.normalize_predictors()
    b.save_df()           
    b.load_df()
    