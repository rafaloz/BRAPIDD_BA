import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score
import statsmodels.api as sm
import pingouin as pg

# Load and process the dataframe
predsAll = pd.read_csv('/home/rafa/Paper_UCL/predicciones/Predictions_All.csv', sep='\t')

model = 'PyMent'  # Options: ['BrainAgeR', 'DBN', 'PyBrainAge', 'ENIGMA', 'PyMent', 'MCCQR_MLP']

# Calculate BrainPAD
predsAll['BrainPAD'] = predsAll[model] - predsAll['Age']

# Prepare the plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
plt.tight_layout(pad=4.0)

# First plot (Model_MCCQR_MLP vs Age grouped by 'acq')
x_vals = np.linspace(predsAll['Age'].min(), predsAll['Age'].max(), 100)

# Get unique acquisition values
acq_values = predsAll['acq'].unique()
colors = plt.cm.get_cmap('Set1', 2)
acq_color_dict = {acq_value: colors(i % 2) for i, acq_value in enumerate(acq_values)}

for i, acq_value in enumerate(acq_values):
    group = predsAll[predsAll['acq'] == acq_value]
    color = acq_color_dict[acq_value]

    # Scatter plot for each acquisition type
    axs[0, 0].scatter(group['Age'], group[model], label=acq_value, alpha=1, color=color)

    # Linear regression for each acquisition type
    X_group = sm.add_constant(group['Age'])
    model_group = sm.OLS(group[model], X_group).fit()

    # Plot regression line
    x_vals = np.linspace(group['Age'].min(), group['Age'].max(), 100)
    axs[0, 0].plot(x_vals, model_group.params[0] + model_group.params[1] * x_vals, color=color)

    # Calculate MAE, r, R² for each group
    mae = mean_absolute_error(group['Age'], group[model])
    r_value, _ = stats.pearsonr(group['Age'], group[model])
    r2 = r2_score(group['Age'], group[model])

    # Annotate plot with metrics
    axs[0, 0].text(0.05, 0.95 - 0.1 * i, f'{acq_value} - MAE: {mae:.2f}, r: {r_value:.2f}, R²: {r2:.2f}', transform=axs[0, 0].transAxes)

# Add dashed line for equality
axs[0, 0].plot(x_vals, x_vals, 'k--')

axs[0, 0].set_title(model + ' vs Age')
axs[0, 0].set_xlabel('Age')
axs[0, 0].set_ylabel(model)
axs[0, 0].legend(title='acq')

# Second plot (Model_MCCQR_MLP Rapid vs Standard)
rapid_values = predsAll[predsAll['acq'] == 'rapid'][model].values
standard_values = predsAll[predsAll['acq'] == 'standard'][model].values

# Ensure same length for comparison
min_length = min(len(rapid_values), len(standard_values))
rapid_values = rapid_values[:min_length]
standard_values = standard_values[:min_length]

axs[0, 1].scatter(standard_values, rapid_values)
axs[0, 1].set_xlabel(model + ' (Standard)')
axs[0, 1].set_ylabel(model + ' (Rapid)')
axs[0, 1].set_title(model + ' Rapid vs Standard Prediction')

# Add dashed line for equality
axs[0, 1].plot(x_vals, x_vals, 'k--')

ICC_preds = predsAll[['ID', 'acq', model]]
ICC_preds['ID'] = ICC_preds['ID'].str[:10]

# Calculate ICC and Pearson's r
icc = pg.intraclass_corr(data=ICC_preds, targets='ID', raters='acq', ratings=model)['ICC'].iloc[1]
r_icc, _ = stats.pearsonr(standard_values, rapid_values)

# Add text inside the plot
axs[0, 1].text(0.05, 0.95, f"r: {r_icc:.2f}, ICC: {icc:.2f}",
               transform=axs[0, 1].transAxes, verticalalignment='top')

# Third plot (Bland-Altman plot)
mean_model = np.mean([standard_values, rapid_values], axis=0)
diff_model = standard_values - rapid_values

axs[1, 0].scatter(mean_model, diff_model)
axs[1, 0].axhline(np.mean(diff_model), color='gray', linestyle='--')
axs[1, 0].axhline(np.mean(diff_model) + 1.96 * np.std(diff_model), color='gray', linestyle=':')
axs[1, 0].axhline(np.mean(diff_model) - 1.96 * np.std(diff_model), color='gray', linestyle=':')

# Check for normality using Kolmogorov-Smirnov test
normality_standard = stats.kstest(standard_values, 'norm', args=(np.mean(standard_values), np.std(standard_values)))
normality_rapid = stats.kstest(rapid_values, 'norm', args=(np.mean(rapid_values), np.std(rapid_values)))

# Check for homoscedasticity using Levene's test
levene_test = stats.levene(standard_values, rapid_values)

# Display results of normality and homoscedasticity tests
print("Kolmogorov-Smirnov Test for Normality (Standard Values):", normality_standard)
print("Kolmogorov-Smirnov Test for Normality (Rapid Values):", normality_rapid)
print("Levene's Test for Homoscedasticity:", levene_test)

# Evaluate assumptions
if normality_standard.pvalue > 0.05 and normality_rapid.pvalue > 0.05:
    if levene_test.pvalue > 0.05:
        # Perform paired t-test if assumptions are satisfied
        t_stat, p_value = stats.ttest_rel(standard_values, rapid_values)
        print("Paired t-test Results:", t_stat, p_value)
    else:
        # Perform Welch's t-test if homoscedasticity is violated
        t_stat, p_value = stats.ttest_rel(standard_values, rapid_values, alternative='two-sided')
        print("Welch's t-test Results:", t_stat, p_value)
else:
    # Perform Wilcoxon signed-rank test if normality assumption is violated
    wilcoxon_stat, p_value = stats.wilcoxon(standard_values, rapid_values)
    print("Wilcoxon Signed-Rank Test Results:", wilcoxon_stat, p_value)

# Adjust p-value text
p_text = "< 1x10e-3" if p_value < 1e-3 else f"{p_value:.3f}"

# Add the text to the plot
axs[1, 0].set_title(f"Bland-Altman Plot (t-test: p={p_text})")
axs[1, 0].set_xlabel('Mean of Standard and Rapid ' + model)
axs[1, 0].set_ylabel('Difference (Standard - Rapid)')

# Fourth plot (MMSE vs BrainPAD grouped by 'acq')
for i, acq_value in enumerate(acq_values):
    # Filter group by acquisition type and MMSE <= 98
    group = predsAll[(predsAll['acq'] == acq_value) & (predsAll['MMSE'] <= 98)]
    color = acq_color_dict[acq_value]

    # Scatter plot for each acquisition type
    axs[1, 1].scatter(group['BrainPAD'], group['MMSE'], label=acq_value, alpha=1, color=color)

    # Linear regression
    X_group = sm.add_constant(group['BrainPAD'])
    model_group = sm.OLS(group['MMSE'], X_group).fit()

    # Plot regression line
    x_vals_brainpad = np.linspace(group['BrainPAD'].min(), group['BrainPAD'].max(), 100)
    axs[1, 1].plot(x_vals_brainpad, model_group.params[0] + model_group.params[1] * x_vals_brainpad, color=color)

    # Calculate statistics
    r_value, _ = stats.pearsonr(group['BrainPAD'], group['MMSE'])
    r2 = model_group.rsquared
    aic = model_group.aic
    bic = model_group.bic

    # Annotate plot with statistics
    axs[1, 1].text(0.05, 0.95 - 0.1 * i,  f'{acq_value} - r: {r_value:.2f}, R²: {r2:.2f}, AIC: {aic:.2f}, BIC: {bic:.2f}', transform=axs[1, 1].transAxes)

axs[1, 1].set_title('MMSE vs BrainPAD')
axs[1, 1].set_xlabel('BrainPAD (' + model + ' - Age)')
axs[1, 1].set_ylabel('MMSE')
axs[1, 1].legend(title='acq')

plt.savefig(model+'.svg', format='svg')

plt.show()

# Display top 10 entries with highest absolute BrainPAD
predAll_cut = predsAll.copy()
predAll_cut['BrainPAD'] = predAll_cut['BrainPAD'].abs()
predAll_cut.sort_values(by='BrainPAD', ascending=False, inplace=True)
predAll_cut = predAll_cut[['ID', 'Age', 'BrainPAD']]

# Set pandas options for display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
print(predAll_cut.iloc[0:10, :])

print('pause')
