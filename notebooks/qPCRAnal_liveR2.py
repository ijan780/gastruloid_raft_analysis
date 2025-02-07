#%% [markdown]
# Gene panel

#%%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

ploid_types = ['Euploid', 'Aneuploid']
ploid = []
samples = []
assay_types = ['NOG', 'TP53', 'KRT7', 'SPINT1']
assays = []
for p in ploid_types:
    for j in range(0, 3):
        for assay in assay_types:
            ploid.append(p)
            samples.append(j + 1)
            assays.append(assay)
relex = [0.68,1.46,1.14,0.94,
        1.64,0.73,0.74,1.24,
        0.91,0.93,1.18,0.86,
        5.76,0.95,5.59,3.27,
        7.67,1.12,5.52,4.20,
        8.42,1.16,8.00,3.48]
df = pd.DataFrame({'Ploid': ploid,
                  'Sample': samples,
                  'Assay': assays,
                  'RelEx': relex})

plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 1, figsize=(4, 4))

sns.swarmplot(x='Assay',
              y='RelEx',
              data= df,
              hue='Ploid',
              size=10,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              zorder=1)
plt.legend(loc='upper center',
           bbox_to_anchor=(0.5, 1.17),
           ncols=2,
           handletextpad=0)

axes.set_xticks([0, 1, 2, 3], assay_types, fontweight='bold')
axes.set_xlabel('')
axes.set_ylabel('Relative Expression', fontweight='bold')

plt.tight_layout()
plt.show()

#%% [markdown]
# qPCR run (triplicate)

#%%
from pathlib import Path
from parse import parse
import pandas as pd
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt

import scipy.stats as stats

dataDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Genetic Analysis\Manuscript 1')
df_raw = pd.read_csv(dataDir / '2024_06_20_NOG&KRT7.csv')

plate_set = ['NOG', 'KRT7']
arr_set = ['1', '2', '3']
ploid_set = ['Eu', 'An']
num_set = ['1', '2', '3']
target_set = [['GAPDH', 'NOG'], ['GAPDH', 'KRT7']]

arrs = []
ploids = []
samp_nums = []
pattern = '{arr}_{ploid}_{num}'
for samp in df_raw['Sample Name'].tolist():
    patternResult = parse(pattern, samp)
    arrs.append(patternResult['arr'])
    ploids.append(patternResult['ploid'])
    samp_nums.append(patternResult['num'])
df_raw['Arr'] = arrs
df_raw['Ploid'] = ploids
df_raw['Samp_Num'] = samp_nums

plates2 = []
arrs2 = []
ploids2 = []
samp_nums2 = []
targets2 = []
means_CT = []
for i in range(len(plate_set)):
    for a in arr_set:
        for p in ploid_set:
            for n in num_set:
                for t in target_set[i]:
                    # Separate ctrls for ea plate.
                    # df_sub = df_raw[(df_raw['Plate']==plate_set[i])
                    #                 & (df_raw['Arr']==a) 
                    #                 & (df_raw['Ploid']==p) 
                    #                 & (df_raw['Samp_Num']==n) 
                    #                 & (df_raw['Target Name']==t)
                    #                ]

                    # Mixed ctrls for both plates
                    df_sub = df_raw[(df_raw['Arr'] == a)
                                    & (df_raw['Ploid'] == p)
                                    & (df_raw['Samp_Num'] == n)
                                    & (df_raw['Target Name'] == t)
                                    ]

                    if df_sub.empty:
                        continue
                    else:
                        plates2.append(plate_set[i])
                        arrs2.append(a)
                        ploids2.append(p)
                        samp_nums2.append(n)
                        targets2.append(t)
                        mean_CT = np.mean(df_sub['CT'].tolist())
                        means_CT.append(mean_CT)
df_mean_CT = pd.DataFrame({'Plate': plates2,
                           'Arr': arrs2,
                           'Ploid': ploids2,
                           'Samp_Num': samp_nums2,
                           'Target': targets2,
                           'Mean_CT': means_CT
                           })

samp_names3 = []
arrs3 = []
ploids3 = []
samp_nums3 = []
targets3 = []
dCTs = []

for pl in plate_set:
    for a in arr_set:
        for p in ploid_set:
            for n in num_set:
                df_sub = df_mean_CT[(df_mean_CT['Plate'] == pl)
                                    & (df_mean_CT['Arr'] == a)
                                    & (df_mean_CT['Ploid'] == p)
                                    & (df_mean_CT['Samp_Num'] == n)
                                    ]

                flag_emptyData = df_sub[df_sub['Target'] == pl].empty
                # print(f'{a}_{p}_{n}_{pl}: {flag_emptyData}')

                if flag_emptyData:
                    continue
                else:
                    gapdh = df_sub[df_sub['Target'] == 'GAPDH']['Mean_CT'].item()

                    if pl == 'NOG':
                        goi = df_sub[df_sub['Target'] == 'NOG']['Mean_CT'].item()

                    elif pl == 'KRT7':
                        goi = df_sub[df_sub['Target'] == 'KRT7']['Mean_CT'].item()

                    samp_names3.append(f'{a}_{p}_{n}')
                    arrs3.append(a)
                    ploids3.append(p)
                    samp_nums3.append(n)
                    targets3.append(pl)
                    dCTs.append(goi - gapdh)
df_final = pd.DataFrame({'Samp_Name': samp_names3,
                         'Arr': arrs3,
                         'Ploid': ploids3,
                         'Samp_Num': samp_nums3,
                         'Target': targets3,
                         'dCT': dCTs
                         })

ctrl_dCTs = []
ddCTs = []
rel_exps = []
for t in plate_set:
    for a in arr_set:
        df_sub1 = df_final[(df_final['Arr'] == a)
                           & (df_final['Ploid'] == 'Eu')
                           & (df_final['Target'] == t)]
        dCT_ctrl = np.mean(df_sub1['dCT'].tolist())
        for p in ploid_set:
            for n in num_set:

                df_sub2 = df_final[(df_final['Arr'] == a)
                                   & (df_final['Ploid'] == p)
                                   & (df_final['Samp_Num'] == n)
                                   & (df_final['Target'] == t)
                                   ]
                if df_sub2.empty:
                    continue
                else:
                    dCT = df_sub2['dCT'].item()
                    ddCT = dCT - dCT_ctrl
                    rel_exp = 2 ** (-ddCT)

                    ctrl_dCTs.append(dCT_ctrl)
                    ddCTs.append(ddCT)
                    rel_exps.append(rel_exp)

df_final['Average dCT'] = ctrl_dCTs
df_final['ddCTs'] = ddCTs
df_final['2^-ddCT'] = rel_exps

#%%
df_final.to_csv((dataDir / '2024_06_20_NOG&KRT7_RelExp.csv'),
                header=True, index=False)

#%%
arrs4 = []
ploids4 = []
targets4 = []
means_relexp = []
stdevs_relexp = []

for a in arr_set:
    for p in ploid_set:
        for t in plate_set:
            df_sub = df_final[(df_final['Arr'] == a)
                               & (df_final['Ploid'] == p)
                               & (df_final['Target'] == t)
                               ]
            arrs4.append(a)
            ploids4.append(p)
            targets4.append(t)
            mean_relexp = np.mean(df_sub['2^-ddCT'].tolist())
            means_relexp.append(mean_relexp)
            stdev_relexp = np.std(df_sub['2^-ddCT'].tolist())
            stdevs_relexp.append(stdev_relexp)

df_mean_relexp = pd.DataFrame({'Arr': arrs4,
                               'Ploid': ploids4,
                               'Target': targets4,
                               'Mean_RelExp': means_relexp,
                               'Stdev_RelExp': stdevs_relexp
                               })

ttest_nog = stats.ttest_ind(a=df_mean_relexp[(df_mean_relexp['Ploid'] == 'Eu')
                                             & (df_mean_relexp['Target'] == 'NOG')]['Mean_RelExp'],
                            b=df_mean_relexp[(df_mean_relexp['Ploid'] == 'An')
                                             & (df_mean_relexp['Target'] == 'NOG')]['Mean_RelExp'],
                            equal_var=True)
ttest_krt = stats.ttest_ind(a=df_mean_relexp[(df_mean_relexp['Ploid'] == 'Eu')
                                             & (df_mean_relexp['Target'] == 'KRT7')]['Mean_RelExp'],
                            b=df_mean_relexp[(df_mean_relexp['Ploid'] == 'An')
                                             & (df_mean_relexp['Target'] == 'KRT7')]['Mean_RelExp'],
                            equal_var=True)
print(f'NOG: {ttest_nog}')
print(f'KRT7: {ttest_krt}')

#%%
df_mean_relexp.to_csv((dataDir / '2024_06_20_NOG&KRT7_RelExp_Mean.csv'),
                      index=False)

#%%
ploids5 = []
targets5 = []
means_relexp2 = []
stdevs_relexp2 = []

for p in ploid_set:
    for t in plate_set:
        df_sub = df_mean_relexp[(df_mean_relexp['Ploid'] == p)
                           & (df_mean_relexp['Target'] == t)
                           ]
        ploids5.append(p)
        targets5.append(t)
        mean_relexp2 = np.mean(df_sub['Mean_RelExp'].tolist())
        means_relexp2.append(mean_relexp2)
        stdev_relexp2 = np.std(df_sub['Mean_RelExp'].tolist())
        stdevs_relexp2.append(stdev_relexp2)

df_mean_relexp2 = pd.DataFrame({'Ploid': ploids5,
                                'Target': targets5,
                                'Mean_RelExp': means_relexp2,
                                'Stdev_RelExp': stdevs_relexp2
                                })

#%%
df_mean_relexp2.to_csv((dataDir / '2024_06_20_NOG&KRT7_RelExp_Mean2.csv'),
                      index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

sns.stripplot(ax=axes[0],
              x='Ploid',
              y='2^-ddCT',
              data = df_final[df_final['Target']=='NOG'],
              hue='Arr',
              size=12,
              alpha=0.5,
              palette='colorblind',
              jitter=0.3,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes[0],
              x='Ploid',
              y='Mean_RelExp',
              data= df_mean_relexp[df_mean_relexp['Target']=='NOG'],
              hue='Arr',
              size=17,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes[0].set_xticks([0, 1], ['Euploid', 'Aneuploid'], fontweight='bold')
axes[0].set_xlabel('', fontweight='bold')
axes[0].set_xlim(-0.5,1.5)
axes[0].set_ylabel('NOG Relative Expression', fontweight='bold')
axes[0].set_ylim(0,
                 max(df_final[df_final['Target']=='NOG']['2^-ddCT']) * 1.125)

sns.stripplot(ax=axes[1],
              x='Ploid',
              y='2^-ddCT',
              data = df_final[df_final['Target']=='KRT7'],
              hue='Arr',
              size=12,
              palette='colorblind',
              alpha=0.5,
              jitter=0.3,
              legend=False,
              zorder=1)
sns.swarmplot(ax=axes[1],
              x='Ploid',
              y='Mean_RelExp',
              data= df_mean_relexp[df_mean_relexp['Target']=='KRT7'],
              hue='Arr',
              size=17,
              palette='colorblind',
              edgecolor='k',
              linewidth=2,
              legend=False,
              zorder=2)
axes[1].set_xticks([0, 1], ['Euploid', 'Aneuploid'], fontweight='bold')
axes[1].set_xlabel('', fontweight='bold')
axes[1].set_xlim(-0.5,1.5)
axes[1].set_ylabel('KRT7 Relative Expression', fontweight='bold')
axes[1].set_ylim(0,
                 max(df_final[df_final['Target']=='KRT7']['2^-ddCT']) * 1.125)

nog = df_final[df_final['Target']=='NOG']['2^-ddCT'].tolist()
axes[0].plot([0, 0, 1, 1], 
             [max(nog)+0.2,
              max(nog)+0.3,
              max(nog)+0.3,
              max(nog)+0.2],
             linewidth=2, 
             color='k')
axes[0].text((0+1)*0.5, 
             max(nog)+0.07,
             "**", 
             ha='center', 
             va='bottom', 
             color='k',
             fontsize=20
            )
krt = df_final[df_final['Target']=='KRT7']['2^-ddCT'].tolist()
axes[1].plot([0, 0, 1, 1], 
             [max(krt)+0.4,
              max(krt)+0.6,
              max(krt)+0.6,
              max(krt)+0.4],
             linewidth=2, 
             color='k')
axes[1].text((0+1)*0.5, 
             max(krt)+0.2,
             "***", 
             ha='center', 
             va='bottom', 
             color='k',
             fontsize=20
            )

plt.tight_layout()
plt.show()

#%%
nucDir = Path(r'C:\Users\MVX\OneDrive - UW\Allbritton Lab\Manuscripts\Manuscript 1\Data\Eu & Aneu Gastruloid Fluor Analysis')
arrays = [4, 5, 6]

rafts_eu = [
    [484, 284, 75],
    [466, 480, 433],
    [189, 314, 331]
]
rafts_an = [
    [222, 178, 332],
    [268, 195, 127],
    [40, 125, 154]
]

normDNAs = []
for samp in df_final['Samp_Name'].tolist():
    a = samp[0]
    p = samp[2:4]
    n = samp[-1]

    file_name = nucDir / f'{p.upper()}_array{arrays[int(a) - 1]}_FluorAnalysis_CalcHoechst_Raw.csv'
    df_fluor = pd.read_csv(file_name)

    if p == 'Eu':
        raft = rafts_eu[int(a) - 1][int(n) - 1]
    elif p == 'An':
        raft = rafts_an[int(a) - 1][int(n) - 1]

    normDNA = df_fluor[df_fluor['Raft'] == raft]['NormDNA'].item()

    # print(file_name, raft, normDNA)

    normDNAs.append(normDNA)
df_final['NormDNA'] = normDNAs

#%%
df_final.to_csv((dataDir / '2024_06_20_NOG&KRT7_RelExp.csv'),
                index=False)

#%%
plt.rcdefaults()
sns.set_theme(style='white', font_scale=1.25)
plt.rcParams['ytick.left'] = True
fig, axes = plt.subplots(1, 2, figsize=(8, 4))

size = 200
data_nog = df_final[(df_final['Target']=='NOG')]
sns.scatterplot(ax=axes[0],
                x='NormDNA',
                y='2^-ddCT',
                data=data_nog[data_nog['Ploid']=='Eu'],
                marker='o',
                s=size,
                hue='Arr',
                palette='colorblind',
                legend=False)
sns.scatterplot(ax=axes[0],
                x='NormDNA',
                y='2^-ddCT',
                data=data_nog[data_nog['Ploid']=='An'],
                marker='^',
                s=size,
                hue='Arr',
                palette='colorblind',
                legend=False)
m, b = np.polyfit(data_nog['NormDNA'], data_nog['2^-ddCT'], 1)
# X_plot = np.linspace(axes[0].get_xlim()[0],axes[0].get_xlim()[1],100)
X_plot = np.linspace(0,1,100)
axes[0].plot(X_plot, m*X_plot + b, '--', color='orchid', linewidth=2)
axes[0].set_xlabel('Normalized DNA Content', fontweight='bold')
axes[0].set_ylabel('NOG Relative Expression', fontweight='bold')
axes[0].set_xlim(0.2,1)

data_krt = df_final[(df_final['Target']=='KRT7')]
sns.scatterplot(ax=axes[1],
                x='NormDNA',
                y='2^-ddCT',
                data=data_krt[data_krt['Ploid']=='Eu'],
                marker='o',
                s=size,
                hue='Arr',
                legend=False)
sns.scatterplot(ax=axes[1],
                x='NormDNA',
                y='2^-ddCT',
                data=data_krt[data_krt['Ploid']=='An'],
                marker='^',
                s=size,
                hue='Arr',
                legend=False)
m, b = np.polyfit(data_krt['NormDNA'], data_krt['2^-ddCT'], 1)
# X_plot = np.linspace(axes[0].get_xlim()[0],axes[0].get_xlim()[1],100)
X_plot = np.linspace(0,1,100)
axes[1].plot(X_plot, m*X_plot + b, '--', color='orchid', linewidth=2)
axes[1].set_xlabel('Normalized DNA Content', fontweight='bold')
axes[1].set_ylabel('KRT7 Relative Expression', fontweight='bold')
axes[1].set_xlim(0.2,1)

pearson_nog = stats.pearsonr(df_final[df_final['Target']=='NOG']['NormDNA'],
                             df_final[df_final['Target']=='NOG']['2^-ddCT'])
pearson_krt = stats.pearsonr(df_final[df_final['Target']=='KRT7']['NormDNA'],
                             df_final[df_final['Target']=='KRT7']['2^-ddCT'])

print(f'NOG: {pearson_nog}')
print(f'KRT7: {pearson_krt}')

plt.tight_layout()
plt.show()
