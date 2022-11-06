from tabnanny import check
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from traitlets import default

import scipy.stats
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix

file = "C:/Users/borja/Desktop/TFG/streamlit/data/builds/buildsv3_classes.csv"

scores_GPU = pd.read_csv(
    "C:/Users/borja/Desktop/TFG/streamlit/data/scores_data/scoresGPU.csv")
scores_AMD = pd.read_csv(
    "C:/Users/borja/Desktop/TFG/streamlit/data/scores_data/scoresAMD.csv")
scores_Intel = pd.read_csv(
    "C:/Users/borja/Desktop/TFG/streamlit/data/scores_data/scoresIntel.csv")
scores_RAM = pd.read_csv(
    "C:/Users/borja/Desktop/TFG/streamlit/data/scores_data/scoresRAM.csv")
scores_Disk = pd.read_csv(
    "C:/Users/borja/Desktop/TFG/streamlit/data/scores_data/scoresDisk.csv")

st.write("""
# Hardware Finder
""")

gaming = "Esta build es rocomendable para Gaming. \n Con ella serás capaz de jugar a videojuegos con buenos rendimientos, \n a mayores podrás realizar también trabajos de oficina y usos domésticos."
oficina = "Esta build es recomendable para usos de Oficina. \n Con ella serás capaz de realizar labores de trabajo y uso de determinados software."
domestico = "Esta build es recomendable para uso Doméstico. \n Con ella serás capaz de consumir streaming online, realizar ciertas labores de ofimática y hacer demás operaciones básicas."

func_mode = st.sidebar.selectbox(
    'Selecciona modo de funcionamiento',
    ['Valorar PC', 'Buscar por componentes'])

# MODO VALORACION DE UNA BUILD
if(func_mode == 'Valorar PC'):

    cpu_type = st.sidebar.selectbox(
        'Intel o AMD',
        ['Intel', 'AMD'])
    with st.sidebar.form('val'):
        option_gpu = st.selectbox(
            'Seleccion una GPU',
            scores_GPU['Name'])


        if(cpu_type == 'Intel'):
            option_cpu = st.selectbox(
                'Selecciona procesador Intel',
                scores_Intel['Processor'])

        if(cpu_type == 'AMD'):
            option_cpu = st.selectbox(
                'Selecciona procesador AMD',
                scores_AMD['Model'])

        option_ram = st.selectbox(
            'Seleccion una memoria RAM',
            scores_RAM['Name'])

        option_disk = st.selectbox(
            'Seleccion un almacenamiento',
            scores_Disk['Name'])

        button_press = st.form_submit_button(label="Valorar")

        sc_gpu = scores_GPU.loc[scores_GPU['Name']
                                == option_gpu]['score'].values[0]

        if(cpu_type == 'Intel'):
            sc_cpu = scores_Intel.loc[scores_Intel['Processor']
                                    == option_cpu]['score'].values[0]
        else:
            sc_cpu = scores_AMD.loc[scores_AMD['Model']
                                    == option_cpu]['score'].values[0]

        sc_ram = scores_RAM.loc[scores_RAM['Name']
                                == option_ram]['score'].values[0]

        sc_disk = scores_Disk.loc[scores_Disk['Name']
                                == option_disk]['score'].values[0]

        library = pd.read_csv(
            'C:/Users/borja/Desktop/TFG/streamlit/data/builds/buildsv3_classes.csv')

        if button_press:
            # seleccionar solo columnas relevantes
            library_use = library[['GPU_score', 'CPU_score',
                                'RAM_score', 'Disk_score', 'Class']]

            df_aux = pd.DataFrame(columns=['GPU', 'CPU', 'RAM', 'Disk'])
            df_aux.loc[0] = [sc_gpu, sc_cpu, sc_ram, sc_disk]

            # st.sidebar.write(df_aux.loc[0]) para debugear que build esta cogiendo.

            for i in range(df_aux.shape[0]):
                df_aux_row = df_aux.iloc[i, :]
                distances = np.zeros(library_use.shape[0])
                # Variables dependientes
                X = library_use.drop(["Class"], axis=1)
                X = X.values

                # Variable independiente
                y = library_use["Class"]
                y = y.values

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2)

                svclassifier = SVC(kernel='poly')
                svclassifier.fit(X_train, y_train)

                y_pred = svclassifier.predict(df_aux.to_numpy().reshape(1, 4))

                distances[i] = y_pred

                aux = []

                aux.extend([option_gpu, sc_gpu, option_cpu, sc_cpu, option_ram, sc_ram, option_disk, sc_disk, distances[i]])

                #case = np.append(df_aux, distances[i])
                case = pd.Series(aux, index=library.columns)
                library = library.append(case, ignore_index=True)
                library.to_csv(file, index=False)

    if(button_press):
        st.header("Esta es la valoración de tus componentes:")
        st.table(library_use[['GPU_score', 'CPU_score', 'RAM_score', 'Disk_score']].tail(1))
        if(library_use.tail(1)['Class'].values[0] == 2):
            st.subheader(gaming)
        if(library_use.tail(1)['Class'].values[0] == 1):
            st.subheader(oficina)
        if(library_use.tail(1)['Class'].values[0] == 0):
            st.subheader(domestico)
    else:
        st.write("Selecciona componentes para valorar tu PC!")

#MODO BUSCAR COMPONENTES
if(func_mode == 'Buscar por componentes'):
    build_class = st.sidebar.selectbox(
        '¿A que quieres orientar tu PC?',
        ['Gaming', 'Oficina', 'Doméstico']
    )

    if build_class == 'Gaming':
        build_class = 2
    if build_class == 'Oficina':
        build_class = 1
    if build_class == 'Doméstico':
        build_class = 0

    cpu_type = st.sidebar.selectbox(
        'Intel o AMD',
        ['Intel', 'AMD'])

    st.sidebar.write("Selecciona aquellos componentes que sepas, la app complementará la build con los no seleccionados!")
    check_gpu = st.sidebar.checkbox('GPU')
    check_cpu = st.sidebar.checkbox('CPU')
    check_ram = st.sidebar.checkbox('RAM')
    check_disk = st.sidebar.checkbox('Disco')

    check_array = [check_gpu, check_cpu, check_ram, check_disk, True]

    with st.sidebar.form('val'):
        sc_array= [99,99,99,99,2]
        if check_gpu:
            selected_gpu = st.selectbox('GPU:', scores_GPU['Name'])
            sc_gpu = scores_GPU.loc[scores_GPU['Name']
                                == selected_gpu]['score'].values[0]
            sc_array[0] = sc_gpu

        if(check_cpu):
            if(cpu_type == "Intel"):
                selected_cpu = st.selectbox('CPU:', scores_Intel['Processor'])
                sc_cpu = scores_Intel.loc[scores_Intel['Processor']
                                    == selected_cpu]['score'].values[0]
                sc_array[1] = sc_cpu
            if(cpu_type == "AMD"):
                selected_cpu = st.selectbox('CPU:', scores_AMD['Model'])
                sc_cpu = scores_AMD.loc[scores_AMD['Model']
                                    == selected_cpu]['score'].values[0]
                sc_array[1] = sc_cpu

        if(check_ram):
            selected_ram = st.selectbox('RAM:', scores_RAM['Name'])
            sc_ram = scores_RAM.loc[scores_RAM['Name']
                                == selected_ram]['score'].values[0]
            sc_array[2] = sc_ram

        if(check_disk):
            selected_disk = st.selectbox('Disk', scores_Disk['Name'])
            sc_disk = scores_Disk.loc[scores_Disk['Name']
                                == selected_disk]['score'].values[0]
            sc_array[3] = sc_disk
        
        if(build_class):
            sc_array[4] = build_class

        ##st.write(check_array)
        ##st.write(sc_array)

        button_press = st.form_submit_button(label="Buscar")
    
        if button_press:
            builds = pd.read_csv('C:/Users/borja/Desktop/TFG/streamlit/data/builds/buildsv3_classes.csv')
            builds_use = builds[['GPU_score', 'CPU_score', 'RAM_score', 'Disk_score', 'Class']]

            data_point = np.array(sc_array)
            
            for i in range(0, 1, 1):
                toret = data_point

                for i in data_point:
                    j = 0
                    if (i == 99):
                        data_point = np.delete(data_point, j)
                        j = j+1
                
                case_row = data_point
                distances = np.zeros(builds.shape[0])
                #Variables dependientes
                depend = []
                indep = []
                for i in range(len(check_array)):
                    if check_array[i] == True:
                        depend.append(builds_use.columns.values[i])
                    else:
                        indep.append(builds_use.columns.values[i])
                X = builds_use[depend]
                X = X.values

                #Variable independiente
                y = builds_use[indep]
                y = y.values
                
                #Se calcula la distancia del nuevo data con respecto a los ya almacenados
                distances = np.linalg.norm(X - data_point, axis=1)
                
                k = 1
                nearest_neighbor_ids = distances.argsort()[:k]## el id de la recomendacion
                
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=12345)
                
                knn_model = KNeighborsRegressor(n_neighbors=k)
                
                knn_model.fit(X_train, y_train)    

                if toret[0] == 99:
                    toret[0] = builds.iloc[nearest_neighbor_ids[0]]["GPU_score"]
                if toret[1] == 99:
                    toret[1] = builds.iloc[nearest_neighbor_ids[0]]["CPU_score"]
                if toret[2] == 99:
                    toret[2] = builds.iloc[nearest_neighbor_ids[0]]["RAM_score"]
                if toret[3] == 99:
                    toret[3] = builds.iloc[nearest_neighbor_ids[0]]["Disk_score"]

                aux = []

                if(check_array[0] == True):
                    aux.extend([selected_gpu, sc_gpu])
                if(check_array[0] == False):
                    aux.extend([builds.iloc[nearest_neighbor_ids[0]]["GPU_name"], builds.iloc[nearest_neighbor_ids[0]]["GPU_score"]])

                if(check_array[1] == True):
                    aux.extend([selected_cpu, sc_cpu])
                if(check_array[1] == False):
                    aux.extend([builds.iloc[nearest_neighbor_ids[0]]["CPU_name"], builds.iloc[nearest_neighbor_ids[0]]["CPU_score"]])

                if(check_array[2] == True):
                    aux.extend([selected_ram, sc_ram])
                if(check_array[2] == False):
                    aux.extend([builds.iloc[nearest_neighbor_ids[0]]["RAM_name"], builds.iloc[nearest_neighbor_ids[0]]["RAM_score"]])

                if(check_array[3] == True):
                    aux.extend([selected_disk, sc_disk])
                if(check_array[3] == False):
                    aux.extend([builds.iloc[nearest_neighbor_ids[0]]["Disk"], builds.iloc[nearest_neighbor_ids[0]]["Disk_score"]])
                
                aux.append(build_class)

                case = pd.Series(aux, index = builds.columns)
                builds = builds.append(case, ignore_index = True)
                builds.to_csv(file, index=False)
        
    if button_press:
        if check_array[0] == False:
            st.subheader("GPU recomendada: " + builds.iloc[nearest_neighbor_ids[0]]["GPU_name"])
        if check_array[1] == False:
            st.subheader("CPU recomendada: " + builds.iloc[nearest_neighbor_ids[0]]["CPU_name"])
        if check_array[2] == False:
            st.subheader("RAM recomendada: " + builds.iloc[nearest_neighbor_ids[0]]["RAM_name"])
        if check_array[3] == False:
            st.subheader("Disco recomendado: " + builds.iloc[nearest_neighbor_ids[0]]["Disk"])
        st.write("\n \n")
        st.subheader("Build completa recomendada: ")
        st.write("CPU: "+ builds.iloc[len(builds.axes[0])-1]["GPU_name"])
        st.write("GPU: "+ builds.iloc[len(builds.axes[0])-1]["CPU_name"])
        st.write("RAM: "+ builds.iloc[len(builds.axes[0])-1]["RAM_name"])
        st.write("Disco: "+ builds.iloc[len(builds.axes[0])-1]["Disk"])

        
           
