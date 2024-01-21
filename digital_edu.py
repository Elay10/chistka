#создай здесь свой индивидуальный проект!
import pandas as pd 

df = pd.read_csv('train.csv')
df.drop(['id', 'has_photo', 'has_mobile', 'followers_count', 'relation', 'life_main', 'people_main', 'bdate', 'graduation', 'city'], axis = 1, inplace = True)
#print(df['occupation_type'].value_counts())
#print(df['last_seen'].value_counts())
#print(df['education_form'].value_counts())
#print(df['langs'].value_counts())
#print(df['education_status'].value_counts())
#print(df['result'].value_counts())
#df.info()
#reason = df.groupby(by = 'occupation_type')['last_seen'].mean()
#reason2 = df.groupby(by = 'education_status')['result'].mean()
#print(reason2)
#print(df['career_start'].value_counts())
#df['sex']=df['sex'].apply(sex_apply)
df['education_form'].fillna('Full-time', inplace = True)
df[list(pd.get_dummies(df['education_form']).columns)] = pd.get_dummies(df['education_form'])
df.drop(['education_form'], axis = 1, inplace = True)

def edu_status_apply(edu):
    if edu == 'Undergraduate applicant':
        return 1 
    elif edu.find('Student')!= -1:
        return 2
    elif edu.find('Alumnus')!= -1:
        return 3
    else:
        return 4
df['education_status'] = df['education_status'].apply(edu_status_apply)

def langs_apply(langs):
    if langs.find('Русский')!=-1:
        return 1
    return 0 
df['langs'] = df['langs'].apply(langs_apply)
df['occupation_type'].fillna('university', inplace = True)

def ocu_type_apply(ocu_type):
    if ocu_type == 'work':
        return 0
    return 1
df['occupation_type'] = df['occupation_type'].apply(ocu_type_apply)
print(df['occupation_type'].value_counts())
df.info()

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

X = df.drop('result', axis = 1)
y = df['result']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

classifier = KNeighborsClassifier(n_neighbors=5)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
print(y_test)
print(y_pred)
print('Процент правильных исходов:', round(accuracy_score(y_test, y_pred)*100, 2))