from utils import *
import pickle
import pandas as pd
from scipy.stats import ks_2samp
from MCQR_MLP import *
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load data
data = pd.read_csv('..../Training_Data.csv')

# randomize
datos_todos = data.sample(frac=1, random_state=42).reset_index(drop=True)
datos_train_all, datos_test_all = train_test_split(datos_todos, test_size=0.1, random_state=42)

# Variables to store results
ks_result, features, features_tag = [], [], []
Results_dataframe_perceptron_test = pd.DataFrame()
Results_dataframe_perceptron_val = pd.DataFrame()

MAE_val, MAE_test, r_test = [], [], []

prediction_perceptron_saved_test, prediction_perceptron_saved_val = [], []
edades_val, edades_test = [], []

save_dir = '.../savedir/'
datos_train = datos_train_all
datos_test = datos_test_all

# Split the DataFrame into training and testing sets
datos_train, datos_val = train_test_split(datos_train, test_size=0.1, random_state=42)

datos_train.to_csv(os.path.join(save_dir, 'Datos_train_sample.csv'), index=False)
datos_test.to_csv(os.path.join(save_dir, 'Datos_Test_sample.csv'), index=False)
datos_val.to_csv(os.path.join(save_dir, 'Datos_val_sample.csv'), index=False)

edades_todos_train = datos_train['Edad'].values
edades_todos_val = datos_val['Edad'].values
edades_todos_test = datos_test['Edad'].values

edades_test.append(edades_todos_test)

maquinas_todos_train = datos_train['Escaner'].values
maquinas_todos_val = datos_val['Escaner'].values
maquinas_todos_test = datos_test['Escaner'].values

Bo_todos_train = datos_train['Bo'].values
Bo_todos_val = datos_val['Bo'].values
Bo_todos_test = datos_test['Bo'].values

sex_todos_train = datos_train['sexo(M=1;F=0)'].values
sex_todos_val = datos_val['sexo(M=1;F=0)'].values
sex_todos_test = datos_test['sexo(M=1;F=0)'].values

print('[INFO] ##### Standarization #####')
datos_train = datos_train.drop(['ID', 'Bo', 'Escaner', 'DataBase', 'Edad', 'Patologia', 'sexo(M=1;F=0)'], axis=1)
datos_val = datos_val.drop(['ID', 'Bo', 'Escaner', 'DataBase', 'Edad', 'Patologia', 'sexo(M=1;F=0)'], axis=1)
datos_test = datos_test.drop(['ID', 'Bo', 'Escaner', 'DataBase', 'Edad', 'Patologia', 'sexo(M=1;F=0)'], axis=1)


ks_test = ks_2samp(edades_todos_train, edades_todos_val)
print('test de Kolmogorov-Smirnov para edades train-val')
print('si es mayor de 0.05 no puedo descartar que sean iguales: '+str(ks_test[1]))
ks_result.append(ks_test[1])

features_morphological = datos_train.columns.tolist()

print('[INFO] ##### Outliers #####')
start_time = datetime.now()
# Erase outliers
X_train, X_val, X_test = outlier_flattening(datos_train, datos_val, datos_test)
end_time = datetime.now()
print(f"Function execution time: {(end_time - start_time).total_seconds()} seconds")

# transform to array
X_train = X_train.values
X_val = X_val.values
X_test = X_test.values

print('[INFO] ##### Normalization #####')
start_time = datetime.now()
# 3.- Normalization
X_train, X_val, X_test = normalize_data_min_max(X_train, X_val, X_test, (-1, 1))
end_time = datetime.now()
print(f"Function execution time: {(end_time - start_time).total_seconds()} seconds")

print('[INFO] ##### Feature Selection #####')

# 4.- Feature Selection
X_train = pd.DataFrame(X_train, columns=features_morphological)
X_val = pd.DataFrame(X_val, columns=features_morphological)
X_test = pd.DataFrame(X_test, columns=features_morphological)
start_time = datetime.now()
X_train, X_val, X_test, features_names_SFS = feature_selection(X_train, X_val, X_test, edades_todos_train, 100, 20)

end_time = datetime.now()
print(f"Function execution time: {(end_time - start_time).total_seconds()} seconds")

print('[INFO] Nº of features selected: '+str(100))
print('[INFO] Features selected: \n')
print(features_names_SFS)
print('Training data shape: '+str(X_train.shape)+'\n')
features.append(features_names_SFS)
features_tag.append('features_nfeats_'+str(100))

# define variables to store results
listas_perceptron = define_lists_cnn()

print('[INFO] ##### Training #####')

# MCCQR_MLP
model = MCCQR_MLP_Regressor(100, 16, quantile_fits=np.arange(0.01, 1.01, 0.01), dropout_rate=0.0, device='cpu')
MAEs_and_rs_perceptron_test = execute_in_val_and_test_MCCQR_MLP(X_train, edades_todos_train, X_val, edades_todos_val, X_test, edades_todos_test, listas_perceptron, model, 100, save_dir, 0)
Results_dataframe_perceptron_test = pd.concat([MAEs_and_rs_perceptron_test, Results_dataframe_perceptron_test], axis=0)

Results_dataframe_perceptron_test.to_csv(os.path.join(save_dir, 'results_FastSurfer_perceptron_test.csv'))

with open(os.path.join(save_dir, 'lista_prueba_perceptron_test.pkl'), 'wb') as f:
    pickle.dump(prediction_perceptron_saved_test, f)

with open(os.path.join(save_dir, 'edades_test.pkl'), 'wb') as f:
    pickle.dump(edades_test, f)

df_Features = pd.DataFrame(list(zip(features_tag, features)), columns=['features_tag', 'features'])
df_Features.to_csv(os.path.join(save_dir, 'features.csv'))

print(features)
print(ks_result)


