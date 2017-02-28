from sklearn.naive_bayes import MultinomialNB
import numpy as np


# Calculate Leave One Out Cross Validation Error
def LOOCV(features_to_use, train_label):
    RSS_list = []
    for i in range(0, features_to_use.shape[1]):
        temp_data = features_to_use.drop(i, axis=0)
        temp_label = np.delete(train_label.reshape(len(train_label), 1), i, 0)
        mnnb = MultinomialNB()
        mnnb.fit(temp_data, temp_label)
        pred = mnnb.predict(features_to_use.iloc[[i]])
        RSS_list.append(sum((train_label[i] - pred) ** 2))
    LOOCV = float(sum(RSS_list)) / len(RSS_list)
    return LOOCV


# Learn naive bayes model from feature set of feature_list
def naive_bayes_with_some_features(all_city_data, all_city_label, feature_list):
    all_city_label = all_city_label.reshape(len(all_city_label), )
    features_to_use = all_city_data.loc[:, feature_list]
    mnnb = MultinomialNB()
    mnnb.fit(features_to_use, all_city_label)
    pred = mnnb.predict(features_to_use)
    print("Number of mislabeled points out of a total " + str(features_to_use.shape[0]) + ' points: ' + (
        str((all_city_label != pred).sum())))
    # LOOCV risk
    print('Feature set: ' + str(feature_list) + '\nLOOCV: ' + str(LOOCV(features_to_use, all_city_label)))
    print('')
    return mnnb