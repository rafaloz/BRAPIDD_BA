
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Lasso

from scipy.stats import mannwhitneyu
from sklearn.metrics import roc_curve, auc, roc_auc_score

from datetime import datetime

from feature_engine.selection import SmartCorrelatedSelection
import infoselect as inf

from scipy import stats
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from numpy import mean, std
import pickle
import os

def outlier_flattening(datos_train, datos_val, datos_test):
    datos_train_flat = datos_train.copy()
    datos_val_flat = datos_val.copy()
    datos_test_flat = datos_test.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_val_flat[col] = np.clip(datos_val[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_val_flat, datos_test_flat

def normalize_data_min_max(datos_train, datos_val, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_val = scaler.transform(datos_val)
    datos_test = scaler.transform(datos_test)

    # Save the scaler to a file using pickle
    # with open('scaler.pkl', 'wb') as file:
    #     pickle.dump(scaler, file)

    return datos_train, datos_val, datos_test

def feature_selection(data_train, data_val, data_test, ages_train, n_features, percentile):

    # 1. Select features using mutual information (percentile-based selection)
    start_time = datetime.now()
    sel_2 = SelectPercentile(mutual_info_regression, percentile=percentile)
    data_train_transformed = sel_2.fit_transform(data_train, ages_train)
    data_val_transformed = sel_2.transform(data_val)
    data_test_transformed = sel_2.transform(data_test)

    # Extract selected feature names
    selected_features = sel_2.get_feature_names_out()

    # Convert transformed data back to DataFrame with correct feature names
    data_train = pd.DataFrame(data_train_transformed, columns=selected_features)
    data_val = pd.DataFrame(data_val_transformed, columns=selected_features)
    data_test = pd.DataFrame(data_test_transformed, columns=selected_features)

    end_time = datetime.now()
    print(f"Percentile FS execution time: {(end_time - start_time).total_seconds()} seconds")

    # 2. Eliminate highly correlated features
    start_time = datetime.now()
    # Erase correlated feautures
    tr = SmartCorrelatedSelection(variables=None, method="pearson", threshold=0.8, missing_values="raise",
                                  selection_method="model_performance", estimator=MLPRegressor(max_iter=100, early_stopping=True, validation_fraction=0.1),
                                  scoring='neg_mean_absolute_error',  cv=3)

    data_train_transformed = tr.fit_transform(data_train, ages_train)
    data_val_transformed = tr.transform(data_val)
    data_test_transformed = tr.transform(data_test)

    selected_features = data_train_transformed.columns.tolist()

    end_time = datetime.now()
    print(f"Correlation FS execution time: {(end_time - start_time).total_seconds()} seconds")

    # 3. Perform additional selection using mutual information-based forward selection
    start_time = datetime.now()
    gmm = inf.get_gmm(data_train_transformed.values, ages_train)
    select = inf.SelectVars(gmm, selection_mode='forward')
    select.fit(data_train_transformed.values, ages_train, verbose=False)

    # Transform data to final selected features
    data_train_filtered = select.transform(data_train_transformed.values, rd=n_features)
    data_val_filtered = select.transform(data_val_transformed.values, rd=n_features)
    data_test_filtered = select.transform(data_test_transformed.values, rd=n_features)

    # Get final selected feature indices and map them to names
    indices = select.feat_hist[n_features]
    final_feature_names = [selected_features[i] for i in indices]

    end_time = datetime.now()
    print(f"MI FS execution time: {(end_time - start_time).total_seconds()} seconds")

    print("nº de features final: " + str(len(final_feature_names)))

    return data_train_filtered, data_val_filtered, data_test_filtered, final_feature_names

def define_lists_cnn():

    # defino listas para guardar los resultados y un dataframe # tab_CNN
    MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN, r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, \
    rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN, betas_tab_CNN = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_tab_CNN = pd.DataFrame()

    listas_tab_CNN = [MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN,
                  r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN,
                  betas_tab_CNN, BAG_ChronoAge_df_tab_CNN, 'tab_CNN']

    return listas_tab_CNN

def quantile_crossing_rate(f):
    """
    Computes the Quantile Crossing Rate (QCR) using NumPy.

    Parameters:
    - f: Array of shape [batch_size, num_quantiles], predictions for each quantile.

    Returns:
    - qcr: Quantile Crossing Rate.
    """
    batch_size, num_quantiles = f.shape
    num_pairs = num_quantiles * (num_quantiles - 1) / 2

    if num_quantiles < 2:
        return 0.0

    # Get pairs of quantiles (i < j)
    i, j = np.triu_indices(num_quantiles, k=1)

    # Compute crossing mask
    f_i = f[:, i]
    f_j = f[:, j]
    crossing_mask = (f_i > f_j).astype(float)

    # Crossing rate per sample
    crossing_rate_per_sample = crossing_mask.mean(axis=1)  # Mean over pairs

    # Average crossing rate over all samples
    qcr = crossing_rate_per_sample.mean()

    return qcr

def execute_in_val_and_test_MCCQR_MLP(data_train_filtered, edades_train, data_val_filtered, edades_val, data_test_filtered, edades_test, lista, regresor, n_features, save_dir, fold):

    # identifico en método de regresión
    regresor_used = lista[9]

    # hago el entrenamiento sobre todos los datos de entrenamiento
    regresor.fit(data_train_filtered, edades_train, data_val_filtered, edades_val, fold, epochs=500, lr=5e-4, weight_decay=1e-4)
    ridge_reg = Lasso(alpha=0.1)
    ridge_reg.fit(data_train_filtered, edades_train)

    data_train_filtered = pd.DataFrame(data_train_filtered)
    data_train_filtered['Edades'] = edades_train
    data_val_filtered = pd.DataFrame(data_val_filtered)
    data_val_filtered['Edades'] = edades_val
    data_test_filtered = pd.DataFrame(data_test_filtered)
    data_test_filtered['Edades'] = edades_test

    data_test_filtered_2, data_cal_filtered = train_test_split(data_test_filtered, test_size=0.3, random_state=21)

    data_train_filtered.to_csv(os.path.join(save_dir, 'datos_train.csv'), index=False)
    data_val_filtered.to_csv(os.path.join(save_dir, 'datos_val.csv'),  index=False)
    data_test_filtered.to_csv(os.path.join(save_dir, 'datos_test.csv'), index=False)

    edades_train = data_train_filtered['Edades'].values
    data_train_filtered = data_train_filtered.drop('Edades', axis=1)
    data_train_filtered = data_train_filtered.values
    edades_val = data_val_filtered['Edades'].values
    data_val_filtered = data_val_filtered.drop('Edades', axis=1)
    data_val_filtered = data_val_filtered.values

    # data_cal_filtered = data_cal_filtered[data_cal_filtered['Edades'] >= 60]
    # data_cal_filtered = data_cal_filtered[data_cal_filtered['Edades'] <= 100]

    edades_cal = data_cal_filtered['Edades'].values
    data_cal_filtered = data_cal_filtered.drop('Edades', axis=1)
    data_cal_filtered = data_cal_filtered.values
    edades_test_2 = data_test_filtered_2['Edades'].values
    data_test_filtered_2 = data_test_filtered_2.drop('Edades', axis=1)
    data_test_filtered_2 = data_test_filtered_2.values

    data_test_filtered = data_test_filtered.drop('Edades', axis=1)
    data_test_filtered = data_test_filtered.values

    regresor.calculate_calibration_constant(data_cal_filtered, edades_cal, alpha=0.1)

    pred_train_all = regresor.predict(data_train_filtered)

    pred_train = pred_train_all['median_aleatory_epistemic']
    pred_train_ridge = ridge_reg.predict(data_train_filtered)

    # Create DataFrame
    df_bias_correction = pd.DataFrame({
        'edades_train': edades_train,
        'pred_train': pred_train
    })

    df_bias_correction.to_csv(os.path.join(save_dir, 'DataFrame_bias_correction.csv'), index=False)

    # Hago la predicción de los casos de test sanos
    pred_test_all = regresor.predict(data_test_filtered_2, apply_calibration=True)

    pred_test_all_ridge = ridge_reg.predict(data_test_filtered_2)

    pred_test_84 = pred_test_all['0.840_aleatory_epistemic']
    pred_test = pred_test_all['median_aleatory_epistemic']
    pred_test_16 = pred_test_all['0.160_aleatory_epistemic']

    coverage = np.mean((edades_test_2 >= pred_test_16) & (edades_test_2 <= pred_test_84))
    print(f'Adjusted Coverage in Test (1 SD): {coverage * 100:.2f}%')
    from itertools import islice
    filtered_dict = dict(islice(pred_train_all.items(), 112, 212))
    stacked_array = np.transpose(np.vstack(list(filtered_dict.values())))


    QCR = quantile_crossing_rate(stacked_array)
    print(f'QCR: {QCR:.2f}')

    # Calculo BAG sanos val & test
    BAG_test_sanos = pred_test - edades_test_2
    BAG_test_sanos_ridge = pred_test_all_ridge - edades_test_2

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test_2, pred_test)
    MAPE_biased_test = mean_absolute_percentage_error(edades_test_2, pred_test)
    r_squared = r2_score(edades_test_2, pred_test)
    r_biased_test = stats.pearsonr(edades_test_2, pred_test)[0]
    r_bag_real_biased_test = stats.pearsonr(BAG_test_sanos, edades_test_2)[0]

    # calculo MAE, MAPE y r test
    MAE_biased_test_ridge = mean_absolute_error(edades_test_2, pred_test_all_ridge)
    MAPE_biased_test_ridge = mean_absolute_percentage_error(edades_test_2, pred_test_all_ridge)
    r_squared_ridge = r2_score(edades_test_2, pred_test_all_ridge)
    r_biased_test_ridge = stats.pearsonr(edades_test_2, pred_test_all_ridge)[0]
    r_bag_real_biased_test_ridge = stats.pearsonr(BAG_test_sanos_ridge, edades_test_2)[0]

    # Calculo r MAE para test
    print('----------- ' + regresor_used + ' r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test))
    print('MAPE test: ' + str(MAPE_biased_test))
    print('r test: ' + str(r_biased_test))
    print('R2 test: ' + str(r_squared))

    # calculo r biased test
    print('--------- ' + regresor_used + ' Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test))
    print('')

    # Calculo r MAE para test
    print('----------- LASSO r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test_ridge))
    print('MAPE test: ' + str(MAPE_biased_test_ridge))
    print('r test: ' + str(r_biased_test_ridge))
    print('R2 test: ' + str(r_squared_ridge))

    # calculo r biased test
    print('--------- LASSO Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test_ridge))
    print('')


    # Figura concordancia entre predichas y reales con reg lineal
    plt.figure(figsize=(8, 6))
    plt.scatter(edades_test_2, pred_test, color='blue', label='Predictions')
    plt.plot([edades_test_2.min(), edades_test_2.max()], [edades_test_2.min(), edades_test_2.max()], 'k--', lw=2, label='Ideal Fit')
    plt.xlabel('Real Age')
    plt.ylabel('Predicted Age')
    plt.title('Predicted Age vs. Real Age')

    # Annotate MAE, Pearson correlation r, and R² in the plot
    textstr = '\n'.join((
        f'MAE: {MAE_biased_test:.2f}',
        f'Pearson r: {r_biased_test:.2f}',
        f'R²: {r_squared:.2f}'))
    plt.text(0.05, 0.95, textstr, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # plt.show()

    plt.savefig(save_dir+'/Model_PredAge_vs_Age_fold_'+str(fold)+'.svg')

    MAEs_and_rs_test = pd.DataFrame(list(zip([MAE_biased_test], [r_biased_test], [r_bag_real_biased_test])),
                                    columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'])

    # save the model to disk
    filename = os.path.join(save_dir, 'MCQR_MLP_nfeats_' + str(n_features) + '_fold_'+ str(fold) +'.pkl')
    pickle.dump(regresor, open(filename, 'wb'))

    # results = permutation_importance(regresor, data_train_filtered, edades_train, scoring='neg_mean_absolute_error', n_jobs=-1)

    return MAEs_and_rs_test

def standardize_d(d):
    """
    Caps Cohen's d values to be within -0.999 and 0.999 to ensure valid Fisher's z-transformation.

    Parameters:
    - d (float): Cohen's d value.

    Returns:
    - float: Standardized Cohen's d.
    """
    return np.clip(d, -0.999, 0.999)

def calculate_auc_with_roc(df, gap_column):

    y_true = df['SMC_status'].apply(lambda x: 0 if x == 'SMC' else 1)
    y_score = df[gap_column]

    # Calculate FPR, TPR, and thresholds
    fpr, tpr, thresholds = roc_curve(y_true, y_score)

    # Calculate AUC
    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc

def calculate_auc(df, gap_column):
    smc_status_numeric = df['SMC_status'].apply(lambda x: 0 if x == 'SMC' else 1)
    auc_val = roc_auc_score(smc_status_numeric, df[gap_column])
    return auc_val

def calculate_effect_size(df, gap_column):
    smc_gap = df[df['SMC_status'] == 'SMC'][gap_column]
    not_smc_gap = df[df['SMC_status'] == 'notSMC'][gap_column]

    # Cohen's d (standardized mean difference)
    mean_diff = abs(mean(smc_gap) - mean(not_smc_gap))
    pooled_std = std(list(smc_gap) + list(not_smc_gap), ddof=1)
    cohens_d = mean_diff / pooled_std

    # Cliff's Delta (non-parametric effect size)
    u_stat, _ = mannwhitneyu(smc_gap, not_smc_gap, alternative='two-sided')
    n1, n2 = len(smc_gap), len(not_smc_gap)
    cliff_delta = (2 * u_stat - n1 * n2) / (n1 * n2)

    return cohens_d, cliff_delta

def outlier_flattening_4_entries(datos_train, datos_val, datos_test, otros_datos):
    datos_train_flat = datos_train.copy()
    datos_val_flat = datos_val.copy()
    datos_test_flat = datos_test.copy()
    otros_datos_flat = otros_datos.copy()

    for col in datos_train.columns:
        if col == 'sexo' or col == 'ID' or col == 'Bo' or col == 'sexo(M=1;F=0)' or col == 'Escaner' or col == 'Patologia' or col == 'DataBase' or col == 'Edad':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_val_flat[col] = np.clip(datos_val[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])
            otros_datos_flat[col] = np.clip(otros_datos[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_val_flat, datos_test_flat, otros_datos_flat

def normalize_data_min_max_4_entries(datos_train, datos_val, datos_test, otros_datos, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_val = scaler.transform(datos_val)
    datos_test = scaler.transform(datos_test)
    otros_datos = scaler.transform(otros_datos)

    return datos_train, datos_val, datos_test, otros_datos

def outliers_y_normalizacion(datos_train, datos_val, datos_test, datos_otros):

    features = datos_train.columns.tolist()
    datos_val = datos_val[features]
    datos_test = datos_test[features]
    datos_otros = datos_otros[features]

    datos_train_flat, datos_val_flat, datos_test_flat, datos_otros_flat = outlier_flattening_4_entries(datos_train, datos_val, datos_test, datos_otros)
    datos_train_norm, datos_val_norm, datos_test_norm, datos_otros_norm = normalize_data_min_max_4_entries(datos_train_flat, datos_val_flat, datos_test_flat, datos_otros_flat, (-1, 1))

    datos_train_norm = pd.DataFrame(datos_train_norm, columns=features)
    datos_val_norm = pd.DataFrame(datos_val_norm, columns=features)
    datos_test_norm = pd.DataFrame(datos_test_norm, columns=features)
    datos_otros_norm = pd.DataFrame(datos_otros_norm, columns=features)

    return datos_train_norm, datos_val_norm, datos_test_norm, datos_otros_norm





