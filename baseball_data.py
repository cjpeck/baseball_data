# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp

'''
Git command line:

git add 'experiment.py'
git commit -m 'updated experiment.py'
git remote add origin https://github.com/cjpeck/Mapping.git
git push -u origin master
'''

# Load data
master_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Master.csv'
ms = pd.DataFrame.from_csv(master_fname)

batting_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Batting.csv'
bt = pd.DataFrame.from_csv(batting_fname)
bt['AVG'] = bt['H'] / bt['AB']
bt['SLG'] = (bt['H'] + 1*bt['2B'] + 2*bt['3B'] + 3*bt['HR']) / bt['AB']

pitching_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Pitching.csv'
pt = pd.DataFrame.from_csv(pitching_fname)

fielding_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Fielding.csv'
fd = pd.DataFrame.from_csv(fielding_fname)

team_fname = '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Teams.csv'
tm = pd.DataFrame.from_csv(team_fname)

sal = pd.DataFrame.from_csv(
    '/Users/cjpeck/Dropbox/spyder2/baseball/lahman-csv_2015-01-24/Salaries.csv')
       
# error in the salaries CSV which includes teams 'SFG' and 'NYM' which
# should instead be 'SFN' and 'NYN' - applies to 2014 only
sal.ix[(sal.index=='2014') & (sal['teamID']=='NYM'), 'teamID'] = 'NYN'
sal.ix[(sal.index=='2014') & (sal['teamID']=='SFG'), 'teamID'] = 'SFN'

for year in range(bt['yearID'].min(), bt['yearID'].max() + 1):
    bt_teams = bt.ix[bt['yearID']==year, 'teamID'].unique()
    pt_teams = pt.ix[pt['yearID']==year, 'teamID'].unique()
    fd_teams = fd.ix[fd['yearID']==year, 'teamID'].unique() 
    tm_teams = tm[(tm.index==str(year))]['teamID'].unique()
    sal_teams = sal[(sal.index==str(year))]['teamID'].unique()
    n_teams = [len(bt_teams), len(pt_teams), len(fd_teams), len(tm_teams), len(sal_teams)]
    if len(sal_teams)==0:
        n_teams.pop()
    if len(np.unique(n_teams)) > 1:
        print(str(year), str(n_teams))
    

def get_team_stats(bt, pt, fd, tm):
       
    tFrame = [1985, 2014]
    years = list(range(tFrame[0], tFrame[1]+1))
        
    tm_keys = ['G', 'W', 'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin']
    bt_keys = ['AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 
               'SO', 'IBB', 'HBP', 'SH', 'SF', 'GIDP']
    pt_keys = ['CG', 'SHO', 'SV', 'IPouts', 'H', 'ER', 'HR', 'BB', 
               'SO',  'BFP', 'R']
    fd_keys = ['E', 'SB', 'CS']
    bt_keys.remove('AB')
    bt_keys.remove('R')
    bt_keys.remove('H')
    pt_keys.remove('IPouts')
    pt_keys.remove('SV')
    pt_keys.remove('R')
    pt_keys.remove('BFP')
            
    keys = [x + '_bt' for x in bt_keys]
    keys += [x + '_pt' for x in pt_keys]
    keys += [x + '_fd' for x in fd_keys]
    keys += ['salary']
    
    indices = [[], []]
    remove = []
    for year in years:
        tm_year = tm[str(year)]
        if np.sum(tm_year['G'] < 160):
            print('skipping year ' + str(year))
            remove.append(year)
            continue
        teams = tm_year['teamID']
        #indices.extend([(year, team) for team in teams])
        indices[0].extend([year for _ in teams])
        indices[1].extend([team for team in teams])
    for year in remove:
        years.remove(year)
        
    dfx = pd.DataFrame(index=indices, columns=keys)
    dfy = pd.DataFrame(index=indices, columns=tm_keys)    
    
    for year in years:
        tm_year = tm[str(year)]
        sal_year =  sal[str(year)]            
        bt_year = bt[bt['yearID']==year]
        pt_year = pt[pt['yearID']==year]
        fd_year = fd[fd['yearID']==year]
        teams = tm_year['teamID']
        
        for team in teams:
            
            # team results            
            team_data = tm_year[tm_year['teamID']==team]
            s = pd.Series(team_data[tm_keys].sum(skipna=True))  
            dfy.ix[(year, team)] = s
                                                
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

            #append salaray information
            salary = sal_year[sal_year['teamID']==team]['salary'].sum()
            s =  s.append(pd.Series(salary, index=['salary']))
            
            #append to dataframe
            dfx.ix[(year, team)] = s
                
    return dfx, dfy            

### ANALYSIS OF SALARY PREDICTING WINSS
def salary_figures(dfx, dfy):
    
    # parameters for sliding window regression    
    years = dfx.index.levels[0]
    tInt = 10
    tShift = 1
    tStart = []
    tEnd = []
    i = 0
    while i + tInt < len(years):
        tStart.append(years[i])
        tEnd.append(years[i+tInt])
        i += tShift
 
    #normalize
    dfx_z = dfx.apply(lambda x: (x - dfx.ix[x.name[0]].mean()) / 
                                 dfx.ix[x.name[0]].std(), axis=1)
                                         
    # overall regression (for all years) and sliding regression    
    beta = sp.stats.linregress(dfx_z['salary'], dfy['W'])
    beta_sliding = []
    for t in range(len(tStart)):
        tmp =sp.stats.linregress(dfx_z.ix[tStart[t]:tEnd[t], 'salary'],
                                 dfy.ix[tStart[t]:tEnd[t], 'W'])
        beta_sliding.append(tmp[0])
    
    # scatter of salary predicting wins
    plt.figure()
    xmin = dfx_z['salary'].min()
    xmax = dfx_z['salary'].max()
    plt.plot((xmin, xmax), (xmin * beta[0] + beta[1], xmax * beta[0] + beta[1]), 
                     color='r', linestyle='-')
    plt.scatter(dfx_z['salary'], dfy['W'])
    plt.title('b1=%1.2f, p=%1.4f' % (beta[0], beta[3]))    
    plt.show()
    
    # set up figure
    plt.figure()
    fig, ax = plt.subplots(nrows=2, ncols=2)
    
    # mean salary as a function of year
    plt.sca(ax[0,0])
    plt.plot(years, dfx.mean(axis=0, level=0)/1e6)
    plt.xlabel('Year')
    plt.ylabel('Mean salary (millions $)')
    
    # change in z-score required as function of increase in wins desired
    plt.sca(ax[0,1])
    wins_desired = np.array([1, 5, 10, 20])
    plt.plot(wins_desired, wins_desired / beta[0])
    plt.xlabel('Increase in predicted wins')
    plt.ylabel('Change in z-score')
    fig.tight_layout()
    
    # change in z-score as function to time (sliding window regression) to win
    # 'wins_desired' more games
    plt.sca(ax[1,0])
    wins_desired = 10
    money_needed = wins_desired / np.array(beta_sliding)
    plt.plot(np.mean(np.c_[tStart, tEnd], 1), money_needed)
    plt.xlabel('Year')
    plt.ylabel('Change in z-score, to get %d wins' % wins_desired)
    
    # salary as a function of year for common z-scores
    z_vals = np.array([-2, -1, 0, 1, 2])
    z_chart = pd.DataFrame()
    for i in z_vals:
        z_chart[i] = (dfx.mean(level=0)['salary'] + 
                      dfx.std(level=0)['salary'] * i) / 1e6
    plt.sca(ax[1,1])
    plt.plot(z_chart)
    plt.xlabel('Year')
    plt.ylabel('Salary')
    
    plt.show()
    
    # save
    directory = '/Users/cjpeck/Dropbox/spyder2/baseball/figures'
    plt.savefig(directory + 'salary.eps', bbox_inches='tight')
    
    # how does salary increase predict an increase in probability of winning
    # the World Series
    winner = dfy['WSWin'] == 'Y'
    prob = []
    z_vals = np.linspace(dfx_z['salary'].min(), dfx_z['salary'].max(), 100)
    for z in z_vals:
        prob.append(np.sum(winner[dfx_z['salary'] <= z]) / len(years))
    plt.plot(z_vals, prob)

   
if __name__ == '__main__':
    dfx, dfy = get_team_stats(bt, pt, fd, tm)
    salary_figures(dfx, dfy)
