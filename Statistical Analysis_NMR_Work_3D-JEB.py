'''This code is to plot 3D surfaces of net metabolic rate vs 
walking speed and terrain amplitude. 
Also makes 2D plots of metabolic rate vs speed and terrain.
Exports some spreadsheets to summarize the tables. 
'''

import numpy                   as np
import pandas                  as pd
import matplotlib.pyplot       as plt
import statsmodels.api         as sm
import statsmodels.formula.api as smf

from mat4py               import loadmat
from scipy.io             import savemat
from scipy.signal         import argrelextrema, savgol_filter
from tabulate             import tabulate
from IPython.display      import Markdown
from scipy.stats          import ttest_ind, ttest_rel
from mpl_toolkits.mplot3d import Axes3D

# Loading dataframes
df_ya_w = pd.read_csv('W_YA_df_export.csv')

subjectsinds = range(10)

# hard-coded leg length data; see also Young Adults Subjects information.csv
ya_ll = np.array([0.900, 0.890, 0.895, 0.950, 0.898, 0.895, 0.905, 0.965, 0.950, 0.978]);
g = 9.81

# All analysis is performed with non-dimensional data, but for plotting and
# presentation, they are usually converted to dimensional, using a 
# single conversion factor for all subjects. 
ya_lengthfactor = np.mean(ya_ll)
ya_speedfactor = np.mean(g*ya_ll)**0.5
ya_powerfactor = (g**1.5) * (np.mean(ya_ll))**0.5 # g^1.5*l^0.5 for power (already mass normalized)

ya_workfactor = g * (np.mean(ya_ll)) # gL for work (already mass normalized)

## Selection of a sub data frame where the walking velocity was 1.2m/s or the Max_h = 0.032m
crossconds = (df_ya_w['Velocity'] == 1.2) | (df_ya_w['Max_h'] == 0.032)

# continuous ranges for predictions but including specific conditions
nmesh = 20
v_conds = [0.8, 1.0, 1.2, 1.4]
h_conds = [0, 0.019, 0.032, 0.045]
velocity_range = np.sort(np.unique(np.concatenate((np.linspace(0.8, 1.4, nmesh), v_conds ))))
h_range = np.sort(np.unique(np.concatenate((np.linspace(0., 0.045, nmesh), h_conds))))

plt.rcParams['svg.fonttype'] = 'none' # to use embedded fonts and avoid converting to outline
plt.rcParams['pdf.use14corefonts'] = True # same thing for saving to pdf

'''Statistical functions ----------------------------------------------------'''
from statsfunctions import rsquaredmixedlm, remove_offsets
# Still need to modify Tabulate to include the new analysis

def Tabulate (Params, Pvalues, R2, table_title, file_name, terrain, ave_data):
    
    params_list = Params.index[:-1]
    params_vals = Params.tolist()[:-1]
    pvalues     = Pvalues.tolist()[:-1]
    #
    Params_List = [];
    
    ## defining the significant values
    for idx, p_list in enumerate(params_list):
        if pvalues[idx] < 0.05:
            Params_List = np.append(Params_List, params_list[idx] + '*')
        else:
            Params_List = np.append(Params_List, params_list[idx])
    
    ## adding the r2 value
    Params_List = np.append(Params_List, 'R2')
    params_vals = np.append(params_vals, R2[1])
    pvalues     = np.append(pvalues, 0)
    
    ##      
    data_frame = pd.DataFrame(np.column_stack((params_vals, pvalues,
                                               terrain,
                                               ave_data)),
                              columns = ['Coefficients', 'p_values', 'h', 'Ave'],
                              index = Params_List)
    
    ## tabulation
    table = tabulate(data_frame, headers=data_frame.columns, tablefmt='grid')
    table = f"{table_title}\n{table}"
    print('\n\n', table)
    data_frame.to_excel(file_name + '.xlsx') # La
        
    return

# Suppress warnings for convergence; models converge satisfactorily.
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning
warnings.simplefilter('ignore', ConvergenceWarning)

'''Statistical analysis ----------------------------------------------------'''

# Linear model fits, starting with net metabolic rate only
# Dimensionless fits metabolic rate with amplitude squared and speed cubed
yawfitdl = smf.mixedlm('Net_MR ~ h2v_dl + v3_dl', data=df_ya_w, groups=df_ya_w['uniqsubs']).fit()
r2_yawfitdl = rsquaredmixedlm(yawfitdl) 
nmr_yaw_df = df_ya_w.copy() # remove offsets using these two lines
nmr_yaw_df['Edot_no'] = remove_offsets(yawfitdl, nmr_yaw_df) # Edot no offset

# Group by speed and amplitude conditions and compute means and stds
# for all conditions, into a condensed dataframe of just means and stds across subjects
nmr_yawg_df  = nmr_yaw_df.groupby(['Velocity', 'Max_h', 'Age', 'Vision']).agg({'Net_MR': ['mean', 'std'],'Edot_no':['mean','std']}).reset_index()

# Generate combinations of velocity and h values, also dimensionless values for predictions in the grid
vel_values, h_values = np.meshgrid(velocity_range, h_range)
h2vfactor = np.mean(np.sqrt(g*np.array(ya_ll))*np.array(ya_ll)**2)
yawpredict = pd.DataFrame({'Velocity': vel_values.flatten(), 'Max_h' : h_values.flatten()})
yawpredict['v_dl'] = (yawpredict['Velocity']/ np.mean(np.sqrt(g*np.array(ya_ll)))) # use mean conversions
yawpredict['h_dl'] = (yawpredict['Max_h']/ np.mean(np.array(ya_ll)))
yawpredict['v2_dl'] = (yawpredict['Velocity']**2 / np.mean(g*np.array(ya_ll))) # use mean conversions
yawpredict['h2_dl'] = (yawpredict['Max_h']**2 / np.mean(np.array(ya_ll))**2)
yawpredict['v3_dl'] = (yawpredict['Velocity'] / np.mean(np.sqrt(g*np.array(ya_ll))))**3
yawpredict['h2v_dl'] = (yawpredict['Velocity']*yawpredict['Max_h']**2)/ h2vfactor
yawpredict['Edot'] = yawfitdl.predict(yawpredict) # predict using dimensionless values
yaw_predevalues = {'YAW': yawpredict.pivot(index='Max_h', columns='Velocity', values='Edot').values} # 2D array for surface plot

# COM work fits for also plotting COM work rate 
# dropping rows with nan
CoM_Works_df = df_ya_w[['ave_positive_work', 'ave_negative_work', 'ave_pushoff_work', 'ave_collision_work',
                        'std_positive_work', 'std_negative_work', 'std_pushoff_work', 'std_collision_work',
                       'h2_dl', 'v2_dl', 'h_dl', 'v_dl', 'uniqsubs', 'Velocity', 'Max_h']]
CoM_Works_df = CoM_Works_df.dropna()

# Work per step fitted to amplitude squared and velocity squared
# All fits treat individuals as having offsets as random effects (group).
posworkfit=smf.mixedlm('ave_positive_work ~ h2_dl + v2_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit(method="bfgs") 
negworkfit=smf.mixedlm('ave_negative_work ~ h2_dl + v2_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit() 
poworkfit=smf.mixedlm('ave_pushoff_work ~ h2_dl + v2_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit()    
coworkfit=smf.mixedlm('ave_collision_work ~ h2_dl + v2_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit() 

## COM work variabilities as linear fits (there is no mechanics here)
posworkfit_std=smf.mixedlm('std_positive_work ~ h_dl + v_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit() 
negworkfit_std=smf.mixedlm('std_negative_work ~ h_dl + v_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit() 
poworkfit_std =smf.mixedlm('std_pushoff_work ~ h_dl + v_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit()   
coworkfit_std =smf.mixedlm('std_collision_work ~ h_dl + v_dl', data=CoM_Works_df, groups=CoM_Works_df['uniqsubs']).fit() 

## calculating the r2 for each work component
r2_posworkfit = rsquaredmixedlm(posworkfit)
r2_negworkfit = rsquaredmixedlm(negworkfit)
r2_poworkfit  = rsquaredmixedlm(poworkfit)
r2_coworkfit  = rsquaredmixedlm(coworkfit)

## calculating r2 for each work variability
r2_posworkfit_std = rsquaredmixedlm(posworkfit_std)
r2_negworkfit_std = rsquaredmixedlm(negworkfit_std)
r2_poworkfit_std  = rsquaredmixedlm(poworkfit_std)
r2_coworkfit_std  = rsquaredmixedlm(coworkfit_std)

##
yawpredict['ave_positive_work'] = posworkfit.predict(yawpredict)
yawpredict['ave_negative_work'] = negworkfit.predict(yawpredict)
yawpredict['ave_pushoff_work'] = poworkfit.predict(yawpredict)
yawpredict['ave_collision_work'] = coworkfit.predict(yawpredict)
yawpredict['posworkstd'] = posworkfit_std.predict(yawpredict) # and standard deviations
yawpredict['negworkstd'] = negworkfit_std.predict(yawpredict)
yawpredict['poworkstd'] = poworkfit_std.predict(yawpredict)
yawpredict['coworkstd'] = coworkfit_std.predict(yawpredict)

## Plot surfaces for predictions with data
yaw_surfvalues = {'ave_positive_work': yawpredict.pivot(index='Max_h', columns='Velocity', values='ave_positive_work').values,
    'ave_negative_work': yawpredict.pivot(index='Max_h', columns='Velocity', values='ave_negative_work').values,
    'ave_pushoff_work': yawpredict.pivot(index='Max_h', columns='Velocity', values='ave_pushoff_work').values,
    'ave_collision_work': yawpredict.pivot(index='Max_h', columns='Velocity', values='ave_collision_work').values}

# com work rate fits for also plotting COM work rate
# dropping rows with nan
CoM_Workrate_df = df_ya_w[['pos_work_rate', 'neg_work_rate', 'PO_work_rate', 'CO_work_rate',
                           'ave_positive_work', 'ave_negative_work', 'ave_pushoff_work', 'ave_collision_work',
                           'std_positive_work', 'std_negative_work', 'std_pushoff_work', 'std_collision_work',
                           'h2v_dl', 'v3_dl', 'uniqsubs', 'Velocity', 'Max_h']]
CoM_Workrate_df = CoM_Workrate_df.dropna()

# Work rate fited to amplitude squared and speed cubed
posworkratefit=smf.mixedlm('pos_work_rate ~ h2v_dl + v3_dl', data=CoM_Workrate_df, groups=CoM_Workrate_df['uniqsubs']).fit() 
negworkratefit=smf.mixedlm('neg_work_rate ~ h2v_dl + v3_dl', data=CoM_Workrate_df, groups=CoM_Workrate_df['uniqsubs']).fit() 
poworkratefit=smf.mixedlm('PO_work_rate ~ h2v_dl + v3_dl', data=CoM_Workrate_df, groups=CoM_Workrate_df['uniqsubs']).fit()   
coworkratefit=smf.mixedlm('CO_work_rate ~ h2v_dl + v3_dl', data=CoM_Workrate_df, groups=CoM_Workrate_df['uniqsubs']).fit() 
yawpredict['pos_work_rate'] = posworkratefit.predict(yawpredict)
yawpredict['neg_work_rate'] = negworkratefit.predict(yawpredict)
yawpredict['PO_work_rate'] = poworkratefit.predict(yawpredict)
yawpredict['CO_work_rate'] = coworkratefit.predict(yawpredict)
# following is a bunch of 2D arrays for making surface prediction plots
yaw_surfvalues.update({'pos_work_rate': yawpredict.pivot(index='Max_h', columns='Velocity', values='pos_work_rate').values,
    'neg_work_rate': yawpredict.pivot(index='Max_h', columns='Velocity', values='neg_work_rate').values,
    'PO_work_rate': yawpredict.pivot(index='Max_h', columns='Velocity', values='PO_work_rate').values,
    'CO_work_rate': yawpredict.pivot(index='Max_h', columns='Velocity', values='CO_work_rate').values})

CoM_Workrate_df['WR_no'] = remove_offsets(posworkratefit, CoM_Workrate_df, 'pos_work_rate')
# grouped condensed data with means and stds for velocity and amplitude conditions, for work and work rates
CoM_Workrateg_df  = CoM_Workrate_df.groupby(['Velocity', 'Max_h']).agg({
    'pos_work_rate': ['mean', 'std'],'WR_no':['mean','std'],'neg_work_rate': ['mean', 'std'],
    'PO_work_rate': ['mean', 'std'],'CO_work_rate': ['mean', 'std'],
    'ave_positive_work': ['mean', 'std'], 'ave_negative_work': ['mean', 'std'],
    'ave_collision_work': ['mean', 'std'], 'ave_pushoff_work': ['mean', 'std']}).reset_index()

# Make some predictions for the main conditions only, especially nominal
with4h = yawpredict['Max_h'].isin(h_conds) & (yawpredict['Velocity']==1.2)
with4v = yawpredict['Velocity'].isin(v_conds) & (yawpredict['Max_h']==0.032)

## Tabulation
## write out tables and spreadsheets, starting with work components

Tabulate(posworkfit.params, posworkfit.pvalues, r2_posworkfit, 'Positive work trends for YA_W', 
         'Positive work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_positive_work', 'mean')] * ya_workfactor)

Tabulate(negworkfit.params, negworkfit.pvalues, r2_negworkfit, 'Negative work trends for YA_W', 
         'Negative work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_negative_work', 'mean')] * ya_workfactor)

Tabulate(poworkfit.params, poworkfit.pvalues, r2_poworkfit, 'Push-off work trends for YA_W', 
         'Push-off work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_pushoff_work', 'mean')] * ya_workfactor)

Tabulate(coworkfit.params, coworkfit.pvalues, r2_coworkfit, 'Collision work trends for YA_W', 
         'Collision work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_collision_work', 'mean')] * ya_workfactor)

## Variabilities for work components
Tabulate(posworkfit_std.params, posworkfit_std.pvalues, r2_posworkfit_std, 'STD Positive work trends for YA_W', 
         'STD Positive work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_positive_work', 'std')] * ya_workfactor)

Tabulate(negworkfit_std.params, negworkfit_std.pvalues, r2_negworkfit_std, 'STD Negative work trends for YA_W', 
         'STD Negative work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_negative_work', 'std')] * ya_workfactor)

Tabulate(poworkfit_std.params, poworkfit_std.pvalues, r2_poworkfit_std, 'STD Push-off work trends for YA_W', 
         'STD Push-off work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_pushoff_work', 'std')] * ya_workfactor)

Tabulate(coworkfit_std.params, coworkfit_std.pvalues, r2_coworkfit_std, 'STD Collision work trends for YA_W', 
         'STD Collision work_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         CoM_Workrateg_df[(CoM_Workrateg_df['Velocity'] == 1.2) & (CoM_Workrateg_df['Max_h'] != 0.005)][('ave_collision_work', 'std')] * ya_workfactor)

## Tabulation and Excel output for metabolic rate
Tabulate(yawfitdl.params, yawfitdl.pvalues, r2_yawfitdl, 'Energetics trends for YA_W', 
         'Net Metabolic Rates_Trends_YA_W', 
         np.array([0, 0.019, 0.032, 0.045]),
         nmr_yawg_df[(nmr_yawg_df['Velocity'] == 1.2) &(nmr_yawg_df['Max_h'] != 0.005)][('Net_MR', 'mean')] * ya_powerfactor)

''' Plotting ------------------------------------------------'''
# Individual dimensional surfaces for the different groups 
#### YA 1x1 --------------------------------------------####
# Use all the combinations of speed and amplitude, whereas
# we had previously used sub data frame where the walking velocity was 1.2m/s or the Max_h = 0.032m

# Create a 3D plot of all dless metabolic rate vs speed and amplitude
fig1 = plt.figure(1)
axew  = fig1.add_subplot(111, projection='3d') # axew means axis energy watching

# Plot the surface predictions from dimensionless model, but on dimensional grid
axew.plot_surface(vel_values, h_values, yaw_predevalues['YAW']*ya_powerfactor, cmap='plasma',
                alpha=0.7, linewidth=0.5)

axew.errorbar(nmr_yawg_df['Velocity'],nmr_yawg_df['Max_h'],nmr_yawg_df['Edot_no']['mean']*ya_powerfactor,
             zerr=nmr_yawg_df['Edot_no']['std']*ya_powerfactor, fmt='o', color = 'b', ecolor='b', 
             markersize='5',  capsize=5, alpha=0.5)

# trend lines for each velocity
for v in v_conds: # 0.8, 1.0, 1.2, 1.4
    indices = vel_values == v # we've ensured that the velocity range includes the specific conditions
    axew.plot(vel_values[indices], h_values[indices], yaw_predevalues['YAW'][indices]*ya_powerfactor,
             linewidth = 1.0)
# and each amplitude
for h in h_conds: # 0, 0.019, 0.032, 0.045
    indices = h_values == h # we've ensured that the velocity range includes the specific conditions
    axew.plot(vel_values[indices], h_values[indices], yaw_predevalues['YAW'][indices]*ya_powerfactor,
             linewidth = 1.0)
    
axew.legend()
# Label the axes
axew.set_xlabel('Walking speed (m/s)', fontsize = 12.5); 
axew.set_ylabel('Terrain amplitude (m)', fontsize = 12.5)
axew.set_zlabel(r'Net metabolic rate (W/kg)', fontsize = 12.5)

axew.tick_params(axis='x', labelsize=10)
axew.tick_params(axis='y', labelsize=10)
axew.tick_params(axis='z', labelsize=10)

#### plotting traces
fig, axs = plt.subplots(2, 3, figsize=(14, 9))
axs = axs.flatten()

titles = ['PO & CO work', 'Pos & Neg work', 'Net metabolic rate', '', '', '']
y_labels = (['Work (J/kg)'] * 2 + ['Metabolic rate (W/kg)']) * 2
x_labels = ['Speed (m/s)'] * 3 + ['Terrain amplitude (m)'] * 3

# Colors for different speeds/amplitudes
colors = ['navy', 'darkgreen', 'darkorange', 'darkred']; 
trend_colors = ['navy', 'darkgreen', 'darkorange', 'darkred']

# Subplot 1: Energy vs. Amplitude (one line for each speed)
ax = plt.subplot(2,3,6)
for idx, v in enumerate(v_conds): # 0.8, 1.0, 1.2, 1.4
    indices = vel_values == v # we've ensured that the velocity range includes the specific conditions
    ax.plot(h_values[indices], yaw_predevalues['YAW'][indices]*ya_powerfactor,linewidth=2.5, color=colors[idx],label=f'{v} m/s')
    eindices = (nmr_yawg_df['Velocity']==v) & (nmr_yawg_df['Max_h'] != 0.005)

    ax.errorbar(nmr_yawg_df[eindices]['Max_h'], nmr_yawg_df[eindices]['Edot_no']['mean']* ya_powerfactor, yerr=nmr_yawg_df[eindices]['Edot_no']['std']* ya_powerfactor,
                 fmt='o', color=colors[idx], ecolor=colors[idx], markersize=3, capsize=2, 
                 elinewidth=1.5, markerfacecolor=colors[idx], linewidth=0.25, markeredgecolor='black')
    roughindices = (nmr_yawg_df['Velocity']==v) & (nmr_yawg_df['Max_h'] == 0.005)
    if sum(roughindices) > 0:
        usecolor = 'lightgray'
        ax.errorbar(nmr_yawg_df[roughindices]['Max_h'], nmr_yawg_df[roughindices]['Edot_no']['mean']* ya_powerfactor, yerr=nmr_yawg_df[roughindices]['Edot_no']['std']* ya_powerfactor,
                 fmt='o', color=usecolor, ecolor=usecolor, markersize=3, capsize=2, 
                 elinewidth=1.5, markerfacecolor=usecolor, linewidth=0.25, markeredgecolor='black')
ax.legend()

# Subplot 2: Energy vs. Speed (one line for each amplitude)
# trend lines for each amplittude   
ax = plt.subplot(2, 3, 3)
for idx, h in enumerate(h_conds): # 0.8, 1.0, 1.2, 1.4
    indices = h_values == h # we've ensured that the amplitude range includes the specific conditions
    ax.plot(vel_values[indices], yaw_predevalues['YAW'][indices]*ya_powerfactor,linewidth=2.5, color=trend_colors[idx], label=f'{h} m')
    eindices = nmr_yawg_df['Max_h']==h
    ax.errorbar(nmr_yawg_df[eindices]['Velocity'], nmr_yawg_df[eindices]['Edot_no']['mean']* ya_powerfactor, yerr=nmr_yawg_df[eindices]['Edot_no']['std']* ya_powerfactor,
                 fmt='o', color=trend_colors[idx], ecolor=trend_colors[idx], markersize=3, capsize=2, 
                 elinewidth=1.5, markerfacecolor=trend_colors[idx], linewidth=0.25, markeredgecolor='black')

ax.legend()
    
## Continuing Fig 6 2D work component fits vs speed and vs amplitude, push-off & collision, positive & negative
ax = plt.subplot(2,3, 2) # for young with lookahead only  # a remake of Fig 6
for ind, h in enumerate(h_conds): # positive and negative work
    ax.plot(velocity_range, yawpredict[yawpredict['Max_h']==h]['ave_positive_work']*ya_workfactor, color=colors[ind], linewidth=2)
    data = CoM_Workrateg_df[CoM_Workrateg_df['Max_h']==h]['ave_positive_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(v_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

    ax.plot(velocity_range, yawpredict[yawpredict['Max_h']==h]['ave_negative_work']*ya_workfactor, color=colors[ind], linewidth=2, linestyle='--')
    data = CoM_Workrateg_df[CoM_Workrateg_df['Max_h']==h]['ave_negative_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(v_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 


ax = plt.subplot(2,3, 1)
for ind, h in enumerate(h_conds): # push-off and collision
    ax.plot(velocity_range, yawpredict[yawpredict['Max_h']==h]['ave_pushoff_work']*ya_workfactor, color=colors[ind], linewidth=2)
    data = CoM_Workrateg_df[CoM_Workrateg_df['Max_h']==h]['ave_pushoff_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(v_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

    ax.plot(velocity_range, yawpredict[yawpredict['Max_h']==h]['ave_collision_work']*ya_workfactor, color=colors[ind], linewidth=2, linestyle='--')
    data = CoM_Workrateg_df[CoM_Workrateg_df['Max_h']==h]['ave_collision_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(v_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 


# 2D work component fits vs amplitude, for each speed
ax = plt.subplot(2, 3, 5)
h_valueext = [0, 0.005, 0.019, 0.032, 0.045] # adding in the rough terrain
for ind, v in enumerate(v_conds): # positive and negative work
    roughindices = (CoM_Workrateg_df['Velocity']==v) & (CoM_Workrateg_df['Max_h']==0.005)
    if sum(roughindices) > 0:
        data = CoM_Workrateg_df[roughindices]['ave_positive_work']*ya_workfactor
        usecolor = 'lightgray'
        markers, caps, bars = ax.errorbar([0.005], data['mean'], yerr=data['std'],
                                fmt = 'o', markersize = 5,  color = usecolor, ecolor = usecolor, elinewidth = 3, capsize=5)
        [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 
        data = CoM_Workrateg_df[roughindices]['ave_negative_work']*ya_workfactor
        markers, caps, bars = ax.errorbar([0.005], data['mean'], yerr=data['std'],
                                fmt = 'o', markersize = 5,  color = usecolor, ecolor = usecolor, elinewidth = 3, capsize=5)
        [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 
    ax.plot(h_range, yawpredict[yawpredict['Velocity']==v]['ave_positive_work']*ya_workfactor, color=colors[ind], linewidth=2)
    data = CoM_Workrateg_df[(CoM_Workrateg_df['Velocity']==v)&(CoM_Workrateg_df['Max_h']!=0.005)]['ave_positive_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(h_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

    ax.plot(h_range, yawpredict[yawpredict['Velocity']==v]['ave_negative_work']*ya_workfactor, color=colors[ind], linewidth=2, linestyle='--')
    data = CoM_Workrateg_df[(CoM_Workrateg_df['Velocity']==v) & (CoM_Workrateg_df['Max_h']!=0.005)]['ave_negative_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(h_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

ax = plt.subplot(2, 3, 4)
for ind, v in enumerate(v_conds): # push-off and collision vs amplitude
    roughindices = (CoM_Workrateg_df['Velocity']==v) & (CoM_Workrateg_df['Max_h']==0.005)
    if sum(roughindices) > 0:
        data = CoM_Workrateg_df[roughindices]['ave_pushoff_work']*ya_workfactor
        usecolor = 'lightgray'
        markers, caps, bars = ax.errorbar(0.005, data['mean'], yerr=data['std'],
                                fmt = 'o', markersize = 5,  color = usecolor, ecolor = usecolor, elinewidth = 3, capsize=5)
        [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 
        data = CoM_Workrateg_df[roughindices]['ave_collision_work']*ya_workfactor
        usecolor = 'lightgray'
        markers, caps, bars = ax.errorbar(0.005, data['mean'], yerr=data['std'],
                                fmt = 'o', markersize = 5,  color = usecolor, ecolor = usecolor, elinewidth = 3, capsize=5)
        [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

    ax.plot(h_range, yawpredict[yawpredict['Velocity']==v]['ave_pushoff_work']*ya_workfactor, color=colors[ind], linewidth=2)
    data = CoM_Workrateg_df[(CoM_Workrateg_df['Velocity']==v)&(CoM_Workrateg_df['Max_h']!=0.005)]['ave_pushoff_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(h_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

    ax.plot(h_range, yawpredict[yawpredict['Velocity']==v]['ave_collision_work']*ya_workfactor, color=colors[ind], linewidth=2, linestyle='--')
    data = CoM_Workrateg_df[(CoM_Workrateg_df['Velocity']==v)&(CoM_Workrateg_df['Max_h']!=0.005)]['ave_collision_work']*ya_workfactor
    markers, caps, bars = ax.errorbar(h_conds, data['mean'], yerr=data['std'],
                      fmt = 'o', markersize = 5,  color = colors[ind], ecolor = colors[ind], elinewidth = 3, capsize=5)
    [bar.set_alpha(0.35) for bar in bars]; [cap.set_alpha(0.35) for cap in caps] 

x_vel = v_conds
x_ter = h_conds
for i in range(6):
    ax = axs[i]
    ax.set_title(titles[i], fontsize=12.5)
    ax.set_xlabel(x_labels[i], fontsize=12.5)
    ax.set_ylabel(y_labels[i], fontsize=12.5)
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    if i < 3: # plots vs speed
        ax.set_xticks(x_vel); 
    else: # plots vs amplitude
        ax.set_xticks(x_ter); 

    if (i == 2) | (i == 5): # metabolic plots y ticks
        ax.set_yticks(range(5))
    else: # work plots y ticks
        ax.set_yticks([-1.5, -1, -0.5, 0, 0.5, 1, 1.5])
        ax.set_ylim([-1.6, 1.6])        
        
plt.tight_layout()

