
from tool.optimization_algorithm.RFO import RFO

# 特征选择
classifiers = [
    # 'lr',
    'lda',
    # 'knn',
    # 'svm',
    # 'dt',
    # 'gnb',
               ]

for classifier in classifiers:

    print(classifier, 60*'*')
    solution = RFO(num_agents=20,
               max_iter=50,
               train_data=X_train,
               train_label=y_train,
               local_search=True,
               classifier=classifier,
               save_conv_graph=False
               )
