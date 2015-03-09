# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
import scipy.stats

'''
Git command line:

git add 'baseball_data.py'
git commit -m 'initial commit'
git remote add origin https://github.com/cjpeck/baseball_data.git
git push -u origin master
'''

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
    keys += ['salary', 'n_over_2SD', 'n_over_3SD', 'n_over_4SD']
    
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
            team_data = sal_year[sal_year['teamID']==team]['salary']            
            salary = float(team_data.sum())
            mean = float(sal[str(year)].mean())
            std = float(sal[str(year)].std()) 
            n_over_2SD = (team_data >= mean + 2*std).sum()
            n_over_3SD = (team_data >= mean + 3*std).sum()
            n_over_4SD = (team_data >= mean + 4*std).sum()
            s =  s.append(pd.Series([salary, n_over_2SD, n_over_3SD, n_over_4SD], 
                                    index=['salary', 'n_over_2SD', 'n_over_3SD', 'n_over_4SD']))

            #append to dataframe
            dfx.ix[(year, team)] = s
            
    return dfx, dfy            

### ANALYSIS OF SALARY PREDICTING WINSS
def salary_figures(dfx, dfy):
    
    directory = '/Users/cjpeck/Dropbox/spyder2/baseball/figures/'
    
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
    plt.savefig(directory + 'salary_wins.eps', bbox_inches='tight')
    plt.show()
    
    # mean salary as a function of year
    plt.figure()
    plt.plot(years, dfx.mean(axis=0, level=0)/1e6)
    plt.xlabel('Year')
    plt.ylabel('Mean salary (millions $)')
    plt.savefig(directory + 'mean_salary.eps', bbox_inches='tight')
    plt.show()
    
    # change in z-score as function to time (sliding window regression) to win
    # 'wins_desired' more games
    plt.figure()
    wins_desired = 4
    money_needed = wins_desired / np.array(beta_sliding)
    plt.plot(np.mean(np.c_[tStart, tEnd], 1), money_needed)
    plt.xlabel('Year')
    plt.ylabel('Change in z-score, to get %d wins' % wins_desired)
    plt.savefig(directory + 'year_wins.eps', bbox_inches='tight')
    plt.show()
    
    # salary as a function of year for common z-scores
    z_vals = np.array([0, 1, 2])
    z_chart = pd.DataFrame()
    for i in z_vals:
        z_chart[i] = (dfx.mean(level=0)['salary'] + 
                      dfx.std(level=0)['salary'] * i) / 1e6
    z_chart.plot()
    plt.xlabel('Year')
    plt.ylabel('Salary')    
    plt.savefig(directory + 'year_salary.eps', bbox_inches='tight')
    plt.show()

    # how does salary increase predict an increase in probability of winning
    # the World Series
    plt.figure()
    
    keys = ['DivWin', 'WCWin', 'LgWin', 'WSWin']
    nkeys = ['nDivWin', 'nWCWin', 'nLgWin', 'nWSWin']
    
    bin_size = 10 # need to be an int divisble by 2 and a factor of 100
    x_vals = list(range(int(bin_size/2), 100, bin_size))
    df = pd.DataFrame(index=x_vals, columns=keys+nkeys)
    for i in range(len(x_vals)):
        lb = np.percentile(dfx_z['salary'], i * bin_size)
        ub = np.percentile(dfx_z['salary'], (i+1) * bin_size)        
        winners = dfy.ix[(dfx_z['salary'] > lb) & (dfx_z['salary'] <= ub), keys] == 'Y'
        losers = dfy.ix[(dfx_z['salary'] > lb) & (dfx_z['salary'] <= ub), keys] == 'N'
        n = winners.sum().add(losers.sum())
        df.ix[x_vals[i], keys] = winners.sum().divide(n)  
        df.ix[x_vals[i], nkeys] = n.rename({keys[i]: nkeys[i] for i in range(len(n))})    
    df[keys].plot(xlim=[0,100])
    plt.savefig(directory + 'playoff_prob.eps', bbox_inches='tight')
    

    plt.figure()
    x2 = dfx['n_over_2SD'] + (.1 * np.random.random(size=(len(dfx),)) - 0.05)
    x3 = dfx['n_over_3SD'] + (.1 * np.random.random(size=(len(dfx),)) - 0.05)
    x4 = dfx['n_over_4SD'] + (.1 * np.random.random(size=(len(dfx),)) - 0.05)
    plt.scatter(x2, dfy['W'], c='b', s=5)
    plt.scatter(x3, dfy['W'], c='r')
    plt.scatter(x4, dfy['W'], c='g')
    b2 = sp.stats.linregress(x2, dfy['W'])
    b3 = sp.stats.linregress(x3, dfy['W'])
    b4 = sp.stats.linregress(x4, dfy['W'])
    plt.show()

if __name__ == '__main__':
    
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
    sal.ix[(sal.index==pd.Timestamp('2014-01-01')) & 
           (sal['teamID']=='NYM'), 'teamID'] = 'NYN'
    sal.ix[(sal.index==pd.Timestamp('2014-01-01')) &
           (sal['teamID']=='SFG'), 'teamID'] = 'SFN'
    
    for year in range(bt['yearID'].min(), bt['yearID'].max() + 1):
        bt_teams = bt.ix[bt['yearID']==year, 'teamID'].unique()
        pt_teams = pt.ix[pt['yearID']==year, 'teamID'].unique()
        fd_teams = fd.ix[fd['yearID']==year, 'teamID'].unique() 
        tm_teams = tm.ix[str(year), 'teamID'].unique()
        n_teams = [len(bt_teams), len(pt_teams), len(fd_teams), len(tm_teams)]
        if str(year) in sal.index:
            sal_teams = sal.ix[str(year), 'teamID'].unique()
            n_teams.append(len(sal_teams))
        if len(np.unique(n_teams)) > 1:
            print(str(year), str(n_teams))
        
        
    dfx, dfy = get_team_stats(bt, pt, fd, tm)
    salary_figures(dfx, dfy)

