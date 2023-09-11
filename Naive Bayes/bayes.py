import pandas as pd 

def naive_bayes(data, observation, feature):
    # get all types of result for feature 
    types = data[feature].unique()
    proabilities = []
    # consider all types(B in the README)
    for B in types:
        # filter to just the type we're looking at
        filtered = data[data[feature] == B]
        # necerssary to iterate when the chosen 
        # feature isnt the last collumn
        filtered = filtered.drop(feature, axis=1)
        # find P(B)
        chance = len(filtered) / len(data)
        
        # consider all features(a_i in the README)
        index = 0
        for a in observation:
            matching = filtered[filtered.iloc[:, index] == a]
            chance *= (len(matching) + 1) / len(filtered)

        print(chance)
        proabilities.append(chance)

    return types[[n == max(proabilities) for n in proabilities]][0]

data = pd.read_csv('iris.csv')
print(naive_bayes(data, (7.9,3.8,6.4,2), 'variety'))