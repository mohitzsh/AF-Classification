import os
import os.path
import numpy as np
import xgboost as xgb
from metrics import F1, get_data, apply_PCA, feature_selection

if __name__ == "__main__":
  training_samples, validation_samples, test_samples, training_labels, validation_labels = get_data()
  # training_samples = feature_selection(training_samples, training_labels)
  # validation_samples = feature_selection(validation_samples, validation_labels)
  # test_samples = feature_selection(test_samples)
  # training_samples = apply_PCA(training_samples)
  # validation_samples = apply_PCA(validation_samples)
  # test_samples = apply_PCA(test_samples)
  dtrain = xgb.DMatrix(training_samples, training_labels)
  # dtest = xgb.DMatrix(validation_samples)
  dtest = xgb.DMatrix(test_samples)
  # max_depths = [32, 64, 128, 256, 512]
  # num_rounds = [32, 64, 128, 256, 512]
  # learning_rates = [0.8, 1]
  max_depths = [64]
  num_rounds = [64]
  learning_rates = [0.8]
  for max_depth in max_depths:
    for num_round in num_rounds:
      for learning_rate in learning_rates:
        param = {'max_depth': max_depth, 'eta': learning_rate, 'silent': 1, 'objective': 'multi:softmax', 'num_class': 4}
        # bst = xgb.train(param, dtrain, num_round)
        # bst.save_model('xgboost.model')
        bst = xgb.Booster({'nthread': 4})
        bst.load_model('xgboost.model')
        predictions = bst.predict(dtest)
        if os.path.isfile('predicted_labels.txt'):
          os.remove('predicted_labels.txt')
        writer = open('predicted_labels.txt', 'w')
        for prediction in predictions:
          writer.write(str(prediction) + '\n')
        writer.close()
        print 'Max depth: ' + str(max_depth) + ' Num round: ' + str(num_round) + ' Learning rate: ' + str(learning_rate) + ' F1 Score: ' + str(F1())
