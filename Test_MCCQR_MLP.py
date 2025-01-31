from utils import *
import ast
from sklearn.metrics import mean_absolute_error, r2_score
import scipy.stats as stats

datos_BRAP = pd.read_csv('.../data_BRAP')

# datos_BRAP = datos_BRAP.rename(columns={'Sex': 'sexo(M=1;F=0)', 'Age': 'Edad'})
.
datos_train = pd.read_csv('.../Datos_train_sample.csv')
datos_val = pd.read_csv('.../Datos_val_sample.csv')
datos_test = pd.read_csv('.../Datos_Test_sample.csv')
datos_AgeRisk = pd.read_csv('.../Datos_AgeRisk_To_Test.csv')

# Create the new 'participant_id' column with the same values as 'ID'
datos_BRAP['participant_id'] = datos_BRAP['ID'].astype(str)
datos_BRAP.loc[:123, 'participant_id'] = datos_BRAP.loc[:123, 'participant_id'] + '_rapid'
datos_BRAP.loc[124:248, 'participant_id'] = datos_BRAP.loc[124:248, 'participant_id'] + '_standard'
datos_BRAP = datos_BRAP.sort_values(by='participant_id', ascending=True)
datos_BRAP.reset_index(inplace=True, drop=True)
participants_id = datos_BRAP['participant_id'].values
AgeRisk_IDs = datos_AgeRisk['ID'].values

edades_todos_train = datos_train['Edad'].values
edades_todos_val = datos_val['Edad'].values
edades_todos_test = datos_test['Edad'].values
edades_BRAP = datos_BRAP['Edad'].values
edades_AgeRisk = datos_AgeRisk['Edad'].values

maquinas_todos_train = datos_train['Escaner'].values
maquinas_todos_val = datos_val['Escaner'].values
maquinas_todos_test = datos_test['Escaner'].values
maquinas_AgeRisk = datos_AgeRisk['Escaner'].values

Bo_todos_train = datos_train['Bo'].values
Bo_todos_val = datos_val['Bo'].values
Bo_todos_test = datos_test['Bo'].values
Bo_AgeRisk = datos_AgeRisk['Bo'].values

sex_todos_train = datos_train['sexo(M=1;F=0)'].values
sex_todos_val = datos_val['sexo(M=1;F=0)'].values
sex_todos_test = datos_test['sexo(M=1;F=0)'].values
sex_AgeRisk = datos_AgeRisk['sexo(M=1;F=0)'].values

datos_train = datos_train.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Edad', 'Patologia'], axis=1)
datos_val = datos_val.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Edad', 'Patologia'], axis=1)
datos_test = datos_test.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Edad', 'Patologia'], axis=1)
datos_AgeRisk = datos_AgeRisk.drop(['ID', 'Bo', 'sexo(M=1;F=0)', 'Escaner', 'DataBase', 'Edad', 'Patologia'], axis=1)
# datos_BRAP = datos_BRAP.drop(['File', 'participant_id', 'slow_acq', 'MMSE', 'sexo'], axis=1)

datos_train_norm, datos_val_norm, datos_test_norm, datos_AgeRisk = outliers_y_normalizacion(datos_train, datos_val, datos_test, datos_AgeRisk)

with open('/home/rafa/PycharmProjects/BA_Model/modelos/modelo_all_armo_100_MCCQR_MLP/MCQR_MLP_nfeats_100_fold_0.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

features = pd.read_csv('/home/rafa/PycharmProjects/BA_Model/modelos/modelo_all_armo_100_MCCQR_MLP/df_features_con_CoRR.csv')
features = ast.literal_eval(features['features'][0])

predictions = loaded_model.predict(datos_AgeRisk[features].values, apply_calibration=True)
final_preds = predictions['median_aleatory_epistemic']

data = {
    'ID': AgeRisk_IDs,
    'prediction': final_preds,
    'Age': edades_AgeRisk
}
df = pd.DataFrame(data)

# Calculate metrics
mae = mean_absolute_error(df['Age'], df['prediction'])
r, _ = stats.pearsonr(df['Age'], df['prediction'])
r2 = r2_score(df['Age'], df['prediction'])

# Scatter plot of real age vs predicted age
plt.figure(figsize=(8, 6))
plt.scatter(df['Age'], df['prediction'], alpha=0.9)
plt.plot([min(df['Age']), max(df['Age'])], [min(df['Age']), max(df['Age'])], color='red', linestyle='--', label='Ideal Fit')
plt.title('Real Age vs Predicted Age')
plt.xlabel('Real Age')
plt.ylabel('Predicted Age')
metrics_legend = f"MAE: {mae:.2f}\nr: {r:.2f}\nR²: {r2:.2f}"
plt.legend([metrics_legend], loc='upper left')
plt.legend()
plt.show()
