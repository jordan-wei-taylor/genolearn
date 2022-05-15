from sklearn.linear_model          import LogisticRegression
from sklearn.neural_network        import MLPClassifier
from sklearn.neighbors             import KNeighborsClassifier
from sklearn.svm                   import SVC
from sklearn.gaussian_process      import GaussianProcessClassifier
from sklearn.tree                  import DecisionTreeClassifier
from sklearn.ensemble              import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes           import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

valid_models = {name : obj for name, obj in globals().items() if 'sklearn' in repr(obj)}

def get_model(name):
    if name not in valid_models:
        raise Exception()
    return valid_models.get(name)    
