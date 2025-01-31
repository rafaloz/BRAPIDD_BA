import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.api import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from scipy.stats import shapiro
import rpy2.robjects as ro
from rpy2.robjects import pandas2ri
import warnings

from utils import *

# Suppress FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Activate automatic conversion between pandas and R
pandas2ri.activate()
ro.r('library(robustlmm)')
ro.r('library(emmeans)')

# Read the data
df = pd.read_csv('.../predictions.csv')
df['SMC_status'] = df['Disease'].apply(lambda x: 'SMC' if x == 'SMC' else 'notSMC')
df['ICV'] = df['ICV']/1000000 # resize ICV from mm3 to L

brain_age_models = ['BrainAgeR', 'DBN', 'PyBrainAge', 'ENIGMA', 'PyMent', 'MCCQR_MLP']

# List of categorical variables in the formula
categorical_vars = ['SMC_status', 'acq', 'Sex']

# Ensure categorical variables are converted to the 'category' dtype
for var in categorical_vars:
    if var in df.columns:
        df[var] = df[var].astype('category')

summaryOutput_list = []

# Loop over each brain age model
for model in brain_age_models:
    print(f'\n=== Analyzing {model} ===')
    # Calculate brain age gap
    df[f'{model}_gap'] = df[model] - df['Age']
    gap_column = f'{model}_gap'

    # Convert DataFrame to R
    r_df = pandas2ri.py2rpy(df)

    # Define formula for robust linear mixed-effects model
    formula = f'{gap_column} ~ C(SMC_status) * C(acq) + Age + C(Sex) + ICV + (1|ID)'

    # Assign the R DataFrame
    ro.r.assign('data_r', r_df)

    # Fit the robust linear mixed model
    ro.r(f""" model_rlmm <- rlmer(formula = {formula}, data = data_r)""")

    # Extract and print summary
    summary_output = ro.r('summary(model_rlmm)')
    print(summary_output)

    # Add post-hoc analysis after model fitting
    posthoc_script = """
    emm <- emmeans(model_rlmm, ~ C(SMC_status) | C(acq))
    posthoc_results <- pairs(emm, simple = 'each')
    summary(posthoc_results)
    """

    posthoc_output = ro.r(posthoc_script)
    print(posthoc_output)

    # Extract residuals
    residuals = ro.r('residuals(model_rlmm)')

    # Extract Exogenous Variables
    fitted_values = ro.r('fitted(model_rlmm)')
    exog = sm.add_constant(fitted_values)

    # QQ Plot of residuals
    fig, ax = plt.subplots()
    qqplot(residuals, line='s', ax=ax)
    plt.title(f'QQ Plot of Residuals for {model}')
    # plt.savefig(f'.../{model}_qqplot_residuals.svg')
    plt.close()

    # Shapiro-Wilk test on residuals
    stat, p_value_shap = shapiro(residuals)
    print("Shapiro-Wilk Test Statistic:", stat)
    print("p-value:", p_value_shap)

    # Plot interaction effects
    plt.figure(figsize=(10, 6))
    sns.pointplot(x='SMC_status', y=gap_column, hue='acq', data=df, dodge=True, markers=['o', 's'], capsize=.1, palette='colorblind')
    plt.title(f'Interaction Plot for {model} Brain Age Gap')
    # plt.savefig(f'.../{model}_interaction_plot.svg')
    plt.close()

    # Perform Breusch-Pagan test
    bp_test = het_breuschpagan(residuals, exog)

    # Unpack test statistics
    bp_stat, bp_pvalue, bp_fvalue, bp_f_pvalue = bp_test

    # Print the results using a single formatted string
    print(f"""Breusch-Pagan Test Statistic: {bp_stat} p-value: {bp_pvalue}, F-Statistic: {bp_fvalue}, F-test p-value: {bp_f_pvalue}""")

    # preapre data for output:
    # Extract the coefficient summary as a data frame
    coef_df_r = ro.r('as.data.frame(coef(summary(model_rlmm)))')
    coef_df = coef_df_r

    coef_df['CI_lower'] = coef_df['Estimate'] - 1.96 * coef_df['Std. Error']
    coef_df['CI_upper'] = coef_df['Estimate'] + 1.96 * coef_df['Std. Error']

    summaryOutput_list.append(coef_df)

# Initialize dictionaries to store results
cohens_d = {}
cliffs_delta = {}
aucs = {}

results = []
fpr_tpr_data = {}

# Loop through acquisition types and compute Cohen's d, Cliff's Delta, and AUC
for acq_type in df['acq'].unique():
    df_acq = df[df['acq'] == acq_type]

    for model in brain_age_models:
        gap_column = f'{model}_gap'
        d, delta = calculate_effect_size(df_acq, gap_column)
        auc_val = calculate_auc(df_acq, gap_column)
        key = (model, acq_type)
        cohens_d[key] = d
        cliffs_delta[key] = delta
        aucs[key] = auc_val

        results.append({
            'Acquisition': acq_type,
            'Model': model,
            'Cohen\'s d': d,
            'AUC': auc_val
        })

        fpr, tpr, _ = calculate_auc_with_roc(df_acq, gap_column)

        # Build FPR-TPR dictionary
        if model not in fpr_tpr_data:
            fpr_tpr_data[model] = {}

        fpr_tpr_data[model][acq_type] = (fpr, tpr, auc_val)

# Now compare Cohen's d and AUCs between rapid and standard acquisitions
comparison_results = []
for model in brain_age_models:
    gap_column = f'{model}_gap'

    df_rapid = df[df['acq'] == 'rapid'][gap_column]
    df_standard = df[df['acq'] == 'standard'][gap_column]

    d_rapid_real = cohens_d.get((model, 'rapid'))
    d_standard_real = cohens_d.get((model, 'standard'))

    # Standardize Cohen's d values
    d_rapid = standardize_d(d_rapid_real)
    d_standard = standardize_d(d_standard_real)

    auc_rapid = aucs.get((model, 'rapid'))
    auc_standard = aucs.get((model, 'standard'))

    n_rapid = len(df_rapid)
    n_standard = len(df_standard)

    comparison_results.append({
        'Model': model,
        'Cohen\'s d Rapid': d_rapid_real,
        'Cohen\'s d Standard': d_standard_real,
        'AUC Rapid': auc_rapid,
        'AUC Standard': auc_standard,
    })

results_df = pd.DataFrame(results)
comparison_results_df = pd.DataFrame(comparison_results)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print(comparison_results_df)

# Prepare data for plotting
long_results = []
for index, row in results_df.iterrows():
    model = row['Model']
    acquisition = row['Acquisition']

    long_results.append({
        'Model': model,
        'Acquisition': acquisition.capitalize(),
        'Metric': "Cohen's d",
        'Value': row["Cohen's d"]
    })
    long_results.append({
        'Model': model,
        'Acquisition': acquisition.capitalize(),
        'Metric': 'AUC',
        'Value': row['AUC']
    })

long_results_df = pd.DataFrame(long_results)

long_results_df['Acquisition_Metric'] = long_results_df['Acquisition'].str.lower() + ' - ' + long_results_df['Metric']

order = [
    "standard - Cohen's d", "standard - AUC",
    "rapid - Cohen's d", "rapid - AUC"
]
long_results_df['Acquisition_Metric'] = pd.Categorical(
    long_results_df['Acquisition_Metric'], categories=order, ordered=True
)

model_order = ['BrainAgeR', 'DBN', 'PyBrainAge', 'ENIGMA', 'PyMent', 'MCCQR_MLP']
long_results_df['Model'] = pd.Categorical(
    long_results_df['Model'], categories=model_order, ordered=True
)

plt.figure(figsize=(14, 6))

palette = {
    "standard - Cohen's d": '#999999',
    'standard - AUC': '#BFBFBF',
    "rapid - Cohen's d": '#E41A1C',
    'rapid - AUC':  '#F07A7C'
}

plt.figure(figsize=(10, 3))  # Width = 8, Height = 4 (adjust as needed)

sns.barplot(
    data=long_results_df,
    x='Model',
    y='Value',
    hue='Acquisition_Metric',
    palette=palette,
    edgecolor=None
)

plt.title('Metrics by Model and Acquisition Type')
plt.xlabel('Model')
plt.ylabel('Value (Cohen\'s d / AUC)')
plt.xticks(rotation=45)
plt.legend(title='Acquisition and Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

brain_age_models = ['BrainAgeR', 'DBN', 'PyBrainAge', 'ENIGMA', 'PyMent', 'MCCQR_MLP']

# Parameters to plot
plot_params = [
    "C(SMC_status)notSMC",
    "C(acq)standard",
    "C(SMC_status)notSMC:C(acq)standard",
    "Age",
    "C(Sex)M",
    "ICV"
]

rename_dict = {
    "C(SMC_status)notSMC": "SMC Status (PND)",
    "C(acq)standard": "Acquisition: Standard",
    "C(SMC_status)notSMC:C(acq)standard": "SMC x Acquisition Interaction",
    "Age": "Age (Years)",
    "C(Sex)M": "Sex (Male)",
    "ICV": "Intracranial Volume (L)"
}

sns.set_style("whitegrid")

# Create a figure with one row and 6 columns
fig, axes = plt.subplots(nrows=1, ncols=len(brain_age_models), figsize=(5 * len(brain_age_models), 4), sharey=True)

for ax, (model_name, summaryOutput_forPlot) in zip(axes, zip(brain_age_models, summaryOutput_list)):
    # Reset the index and include it as a column named 'Parameter'
    summaryOutput_forPlot = summaryOutput_forPlot.reset_index().rename(columns={'index': 'Parameter'})
    summaryOutput_forPlot.index = range(len(summaryOutput_forPlot))

    # Filter dataframe for Parameter elements
    plot_data = summaryOutput_forPlot[summaryOutput_forPlot['Parameter'].isin(plot_params)]

    # Replace the parameter names using rename_dict
    plot_data.loc[:, 'Parameter'] = plot_data['Parameter'].replace(rename_dict)

    param_order = [
        "SMC x Acquisition Interaction",
        "Acquisition: Standard",
        "SMC Status (PND)",
        "Intracranial Volume (L)",
        "Sex (Male)",
        "Age (Years)"
    ]

    # Apply the order using pd.Categorical
    plot_data.loc[:, 'Parameter'] = pd.Categorical(
        plot_data['Parameter'],
        categories=param_order,
        ordered=True
    )

    # Sort the data based on the new order
    plot_data = plot_data.sort_values("Parameter")

    # Ensure y-axis respects order
    ax.set_yticks(range(len(param_order)))
    ax.set_yticklabels(param_order)

    # Draw points and error bars
    for i, row in plot_data.iterrows():
        y_pos = param_order.index(row["Parameter"])  # Ensure correct ordering
        ax.plot([row["CI_lower"], row["CI_upper"]], [y_pos, y_pos], color='gray')
        ax.plot(row["Estimate"], y_pos, 'o', color='black')

    # Add a vertical line at 0 (no effect)
    ax.axvline(x=0, color='red', linestyle='--')
    ax.set_xlim(-20, 20)

    # Set axis labels and title
    ax.set_xlabel("Estimate (Years)")
    ax.set_title(f"{model_name}")

# Only the first subplot needs a y-label since sharey=True
axes[0].set_ylabel("Parameter")

plt.tight_layout()
plt.show()

models = ['BrainAgeR', 'DBN', 'PyBrainAge', 'ENIGMA', 'PyMent', 'MCCQR_MLP']

# Set up figure with 6 subplots in one row
fig, axes = plt.subplots(1, 6, figsize=(20, 4), sharey=True)

# Color palette
palette = {'rapid': (228 / 255, 26 / 255, 28 / 255), 'standard': (153 / 255, 153 / 255, 153 / 255)}

# Plot ROC curves for each model
for ax, model in zip(axes, models):
    data = fpr_tpr_data[model]

    # Rapid ROC curve
    ax.plot(data['rapid'][0], data['rapid'][1], label=f"rapid (AUC = {data['rapid'][2]:.2f})",
            color=palette['rapid'], lw=2)

    # Standard ROC curve
    ax.plot(data['standard'][0], data['standard'][1], label=f"standard (AUC = {data['standard'][2]:.2f})",
            color=palette['standard'], lw=2)

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)

    # Titles and labels
    ax.set_title(model, fontsize=10, weight='bold')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate" if model == models[0] else "")

    # Add legend inside plot
    ax.legend(loc="lower right", frameon=False, fontsize=8)

# Adjust layout
plt.tight_layout()
plt.show()