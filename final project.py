#!/usr/bin/env python
# coding: utf-8

# # Langkah 1: Memahami Permasalahan

# # Langkah 2: Menganalisis dan Memproses Data

# # 2.1 Overview Data

# In[1]:


# Import standard library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Memuat file drug200.csv menjadi pandas dataframe
dataframe = pd.read_csv('drug200.csv')
# Menampilkan 5 baris pertama dari dataframe
dataframe.head()


# # 2.2 Mengubah Data Menjadi Numerik

# In[2]:


# Menampilkan informasi dari dataframe
dataframe.info()


# In[3]:


# Import LabelEncoder dari module sklearn
from sklearn.preprocessing import LabelEncoder
# Menyalin / copy dataframe agar dataframe awal tetap utuh
dataframe_int = dataframe.copy()
# Membuat objek/instance yang bernama encoder
encoder = LabelEncoder()
# Membuat list dari nama kolom data kategori
categorical_data = ['Sex', 'BP', 'Cholesterol', 'Drug']
# Mengubah setiap data kategori menjadi numerik dengan encoder
for kolom in categorical_data:
 dataframe_int[kolom] = encoder.fit_transform(dataframe[kolom])

# Sekarang data sudah berupa angka sepenuhnya
dataframe_int.head()


# # 2.3 Analisa Data Kategori

# In[4]:


for kolom in categorical_data:
 print(kolom,dataframe_int[kolom].unique())


# In[5]:


for kolom in categorical_data:
 print(kolom, dataframe[kolom].unique())


# # 2.4 Analisis Matrix Korelasi

# In[6]:


# Menampilkan matrix korelasi antar kolom
dataframe_int.corr()


# In[7]:


plt.figure(figsize=(10, 8))
plt.title('Matrix Korelasi Data')
sns.heatmap(dataframe_int.corr(), annot=True, linewidths=3)
plt.show()


# # 2.5 Distribusi Data

# In[8]:


def distribusi():
    fig,axes = plt.subplots(nrows=2,ncols=3,figsize=(12,8))
    plt.suptitle('Distribusi',fontsize=24)
    
    def kolom_generator():
        for kolom in dataframe_int:
            yield kolom
    kolom = kolom_generator()

    for i in range(0,2):
        for j in range(0,3):
            k = next(kolom)
            dataframe_int[k].plot(kind='hist',ax=axes[i,j])
            axes[i,j].set_title(k)
    plt.show()


# In[9]:


distribusi()


# # 2.6 Memisahkan Data

# In[10]:


# Memisahkan dataframe awal menjadi data dan label
data = dataframe_int.drop('Drug',axis=1)
label = dataframe_int['Drug']

# Memisahkan dataframe menjadi data latihan dan data tes
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data,label,test_size=0.2)

# Print dataframe.shape untuk mengetahui bentuk dataframe
print(x_train.shape,y_train.shape)
print(x_test.shape,y_test.shape)


# # Langkah 3: Membuat Machine Learning Model

# In[11]:


# import linear SVC model dari sklearn
from sklearn.svm import SVC
# Membuat objek dengan nama "model" dengan memanggil SVC ()
model = SVC(gamma='scale')


# # Langkah 4: Melatih Machine Learning Model

# In[12]:


# melatih model dengan data latihan
model.fit(x_train, y_train)


# In[13]:


SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape='ovr', degree=3, gamma='scale', kernel='rbf' ,max_iter=-1, probability=False, random_state=None, shrinking=True, tol=0.001, verbose=False)


# In[14]:


# Membuat prediksi terhadap data tes
prediction = model.predict(x_test)


# In[15]:


model_acc = 100*model.score(x_test, y_test)
print('SVM Predictions: \n', model.predict(x_test), '\n Accuracy', model_acc, '%')


# # Langkah 5: Evaluasi Model

# # 5.1 Analisa Confusion Matrix

# In[16]:


# Import confusion matrix dari sklearn
from sklearn.metrics import confusion_matrix

# Membuat funsi untuk menampilkan confusion matrix dengan seaborn dan matplotlib
def display_conf(y_test,prediction):
    sns.heatmap(confusion_matrix(y_test,prediction),annot=True,linewidths=3,cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Prediction')
    plt.show()

# Memanggil fungsi untuk menampilkan visualisasi confusion matrix
display_conf(y_test,prediction)


# # 5.2 Analisa Eror Metrics

# In[24]:


print(f'R2 Score : {r2_score(y_test,prediction)}')
print('Classification Report :')
print(classification_report(y_test,prediction))


# In[19]:


# Import r2_score dan classification_report dari sklearn
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report
print(f'R2 Score : {r2_score(y_test,prediction)}')
print('Classification Report :')
print(classification_report(y_test,prediction))


# # Langkah 6: Meningkatkan Model

# In[20]:


# Menggunakan GridSearchCV untuk menemukan model dengan parameter terbaik
from sklearn.model_selection import GridSearchCV

# SVC Model Hyperparameter
param_grid = {'C':[0.01,0.1,1,10,100],
              'gamma':[100,10,1,0,1,0.01]}

# Membuat model terbaik dari semua kemungkinan kombinasi param_grid
best_model = GridSearchCV(SVC(),param_grid,cv=5,refit=True)

# Melatih model terbaik
best_model.fit(x_train,y_train)


# In[21]:


# Model dengan parameter terbaik
best_model.best_estimator_


# In[22]:


# Membuat prediksi dengan model yang telah ditingkatkan
prediction = best_model.predict(x_test)
# Menampilkan confusion matrix pada prediksi yang baru
display_conf(y_test, prediction)


# In[23]:


print(f'R2 Score :{r2_score(y_test, prediction)}')
print('Classification Report : ')
print(classification_report(y_test, prediction))


# # Langkah 7: Mengulangi Semua proses

# # MENYIMPAN MACHINE LEARNING MODEL

# In[25]:


model_acc = 100*best_model.score(x_test, y_test)
print('SVM Predictions: \n', best_model.predict(x_test), '\n Accuracy', model_acc)


# In[26]:


import pickle
# Menyimpan model menjadi file .pkl
with open('AI_DrugClassifier.pkl','wb') as file:
 pickle.dump(best_model,file)


# In[27]:


# Memuat model dalam file .pkl
with open('AI_DrugClassifier.pkl','rb') as file:
 model = pickle.load(file)


# In[28]:


model.best_estimator_


#  # Demonstrasi Prediksi Model

# In[29]:


# Demonstrasi Prediksi Model
import pickle
with open('AI_DrugClassifier.pkl','rb') as file:
 model = pickle.load(file)

def self_prediction():
 age = input('Age : ')
 sex = input('Sex : ')
 bp = input('BP : ')
 chol = input('Cholesterol : ')
 NatoK = input('Na_to_K : ')

 # data harus berbentuk (1,5) yaitu [[age,sex,bp,chol,NatoK]]
 print('\nPrediction')
 print('Patient consumed : ',encoder.inverse_transform(model.predict([[age,sex,bp,chol,NatoK]])))

print(self_prediction())


# In[30]:


get_ipython().system('pip install -q pyngrok')


# In[31]:


get_ipython().system('pip install -q streamlit')


# In[32]:


get_ipython().system('pip install ipykernel')


# In[33]:


get_ipython().run_cell_magic('writefile', 'coba.py', 'import pickle\nimport streamlit as st\n# loading the trained model\npickle_in = open(\'AI_DrugClassifier.pkl\', \'rb\') \nclassifier = pickle.load(pickle_in)\n@st.cache()\n# defining the function which will make the prediction using the data which the user inputs\ndef self_prediction(Age, Sex, BP, Cholesterol, Na_to_K):\n    # Pre-processing user input\n    if Sex == "LAKI-LAKI":\n        Sex = 0\n    else:\n        Sex = 1\n    if BP == "RENDAH":\n        BP = 0\n    elif BP == "NORMAL":\n         BP = 1\n    else :\n        BP = 2\n    if Cholesterol == "NORMAL":\n        Cholesterol = 0\n    else:\n        Cholesterol = 1\n   \n    # Making predictions\n    self_prediction = classifier.predict([[Age, Sex, BP, Cholesterol, Na_to_K]])\n    \n    if self_prediction == 0:\n        pred = \'drugA\'\n    elif self_prediction == 1:\n        pred = \'drugB\'\n    elif self_prediction == 2:\n        pred = \'drugC\'\n    elif self_prediction == 3:\n        pred = \'drugX\'\n    else :\n        pred = \'DrugY\'\n    return pred\n# this is the main function in which we define our webpage\ndef main():\n    # front end elements of the web page\n        html_temp = """\n        <div style ="background-color:green;padding:13px">\n        <h1 style ="color:white;text-align:center;">Drug Identification</h1>\n        </div>\n        """\n    # display the front end aspect\n        st.markdown(html_temp, unsafe_allow_html = True)\n    # following lines create boxes in which user can enter data required to make prediction\n        Age = st.number_input("Age",min_value=15, max_value=74, value=15, step=1)\n        Sex = st.selectbox(\'Gender\',("MALE","FEMALE"))\n        BP = st.selectbox(\'BP / Blood Pressure\',("LOW","NORMAL","HIGH"))\n        Cholesterol = st.selectbox(\'Cholesterol\',("NORMAL","HIGH"))\n        Na_to_K = st.number_input("Enter Na_to_K",min_value=6, max_value=38, value=6, step=1)\n        result =""\n    # when \'Predict\' is clicked, make the prediction and store it\n        if st.button("Predict"):\n            result = self_prediction(Age, Sex, BP, Cholesterol, Na_to_K)\n            st.success(\'Type of Drug {}\'.format(result))\n            print(Na_to_K)\nif __name__==\'__main__\':\n    main()')


# # Menjalankan Aplikasi

# 1. Buka Anaconda promp
# 2. Masuk ke direktori tempat penyimpanan file Anda
# 3. Tuliskan kode "streamlit run coba.py" (tanpa tanda petik)
# 4. Aplikasi akan terbuka di browser default anda
# 5. Untuk membuka aplikasi secara online, jalankan perintah cell di bawah ini
# 6. Buka link nya di web browser laptop atau di HP anda

# In[34]:


from pyngrok import ngrok
public_url2 = ngrok.connect('8051')
public_url2


# In[ ]:




