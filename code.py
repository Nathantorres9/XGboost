
import xgboost as xgb
import numpy as np
from sklearn.metrics import classification_report


def f_minMax_normalization(dat):
    dat_minMax  = (dat - np.min(dat)) / (np.max(dat) - np.min(dat))
    dat_minMax[np.isnan(dat_minMax)] = 0
    return dat_minMax


def f_feature_range_prop(dat, range_num = 10):
    num_i = dat.shape[0]
    range_list = []
    spacing = 1 / range_num
    for i in np.arange(0, 1, spacing):
        min_value = i
        max_value = i + spacing
        if max_value < 1:
            range_i = (np.sum(dat >= min_value) - np.sum(dat >= max_value)) / num_i
        else: 
            range_i = np.sum(dat >= min_value) / num_i
        range_list.append(range_i)
    return np.asarray(range_list).reshape(1, -1)


def define_drop_duration2(raw_dat, drop_value_threshold = 0.35, drop_num_threshold = 1):
    if len(np.where(raw_dat < drop_value_threshold)[0]) > 0:
        drop_duration = np.where(raw_dat < drop_value_threshold)[0][drop_num_threshold-1]
    else:
        drop_duration = 0
    return drop_duration


def f_feature_range_std(dat, range_num = 10):
    range_list1 = []
    range_list2 = []
    range_list3 = []
    spacing = 1 / range_num
    for i in np.arange(0, 1, spacing):
        min_value = i
        max_value = i + spacing
        if max_value >= 1:
            max_value = 1.1
        dat_range = dat[(dat >= min_value) & (dat < max_value)]
        if dat_range.shape[0] == 0:
            dat_range = np.array([0])
        range_std = np.std(dat_range)
        range_mean = np.mean(dat_range)
        range_cv = range_std / (range_mean + 1e-10)
        range_list1.append(range_std)
        range_list2.append(range_cv)

    range_np1 = np.asarray(range_list1).reshape(1, -1)
    range_np2 = np.asarray(range_list2).reshape(1, -1)

    range_list3.append(np.max(range_np1))
    range_list3.append(np.min(range_np1))
    range_list3.append(np.mean(range_np1))
    range_list3.append(np.max(range_np1) / (np.min(range_np1) + 1e-10))
    range_list3.append(np.max(range_np1) / (np.mean(range_np1) + 1e-10))

    range_list3.append(np.max(range_np2))
    range_list3.append(np.min(range_np2))
    range_list3.append(np.mean(range_np2))
    range_list3.append(np.max(range_np2) / (np.min(range_np2) + 1e-10))
    range_list3.append(np.max(range_np2) / (np.mean(range_np2) + 1e-10))

    # range_list3.append(define_drop_duration(dat))
    range_list3.append(define_drop_duration2(dat))

    range_np3 = np.asarray(range_list3).reshape(1, -1)
    return np.concatenate([range_np1, range_np2, range_np3], axis = 1)


def f_datProcess_xgb_example(dat, dat_len = 5000):
    dat = dat[:dat_len].copy()
    dat_1 = f_minMax_normalization(dat)
    dat_2 = f_feature_range_prop(dat_1)#10
    dat_3 = f_feature_range_std(dat_1)#31
    dat_4 = np.concatenate([dat_2, dat_3], axis = 1)#31+10=41
    return dat_4


# 1. 导入数据
img_folder_rootpath = "E:/Xgboost" # 数据存放文件夹
y = np.load('{}/Y_train.npy'.format(img_folder_rootpath))
x = np.load('{}/X_train.npy'.format(img_folder_rootpath))

# 2. 生成特征
x_feature_list = []
for i in range(x.shape[0]):
    x_feature_list.append(f_datProcess_xgb_example(x[i, :]))
x_feature = np.concatenate(x_feature_list, axis = 0)
np.save('{}/X_feature_train.npy'.format(img_folder_rootpath), x_feature)

# 3. 训练模型
y = np.array(y, dtype=np.int32)
dtrain = xgb.DMatrix(data=x_feature, label=y)
params = {'max_depth':5, 'booster':'gbtree', 'objective':'binary:logistic'}
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=100)

# 4. 保存模型
xgb_model.dump_model('{}/xgboost_example.txt'.format(img_folder_rootpath))
xgb_model.save_model('{}/xgboost_example.json'.format(img_folder_rootpath))

# 5. 预测
y_test = np.load('{}/Y_test.npy'.format(img_folder_rootpath))
x_test = np.load('{}/X_test.npy'.format(img_folder_rootpath))
x_feature_test_list = []
for i in range(x_test.shape[0]):
    x_feature_test_list.append(f_datProcess_xgb_example(x_test[i, :]))
x_feature_test = np.concatenate(x_feature_test_list, axis = 0)
np.save('{}/X_feature_test.npy'.format(img_folder_rootpath), x_feature_test)
y_test = np.array(y_test, dtype=np.int32)  # 将y_test转换为整数类型
dtest = xgb.DMatrix(data = x_feature_test, label = y_test)
y_hat = xgb_model.predict(dtest)

# 6. 评估
target_names = ["bad results", "good results"]
#print(classification_report(y_test, np.array(np.where(y_hat > 0.5, '1', '0')), target_names = target_names))

y_hat_numeric = np.where(y_hat > 0.5, 1, 0)  # 将y_hat的结果转换为数值类型
print(classification_report(y_test, y_hat_numeric, target_names=target_names))

#               precision    recall  f1-score   support
#  bad results       1.00      0.99      0.99       400
# good results       0.99      1.00      1.00       400
#     accuracy                           0.99       800
#    macro avg       1.00      0.99      0.99       800
# weighted avg       1.00      0.99      0.99       800







