from itertools import combinations

import numpy as np
import pandas as pd
import math

country = pd.read_excel(r"F:/合作博弈数据.xlsx")
country_list = country[2018].values.tolist()
X_list = country["协调耦合(X)"].values.tolist()
Y_list = country["Yuan/(sej/m2) (Y)"].values.tolist()
function_list = [[3.9495, -2.5038], [1.5537, -0.9001], [0.0276, -0.0105], [0.127, -0.0596], [0.6446, -0.3133],
                 [0.1981, -0.0994], [0.4689, -0.228], [1.5459, -0.9948], [0.3675, -0.1925], [0.7008, -0.3755],
                 [1.1439, -0.6422]]


def combine(temp_list, n):
    """
    根据n获得列表中的所有可能组合（n个元素为一组）
    """
    temp_list2 = []
    for c in combinations(temp_list, n):  # 其实主要用到的是这个函数
        temp_list2.append(c)
    return temp_list2


end_list = []
country_list_end = []
for i in range(len(country_list)):
    end_list.extend(combine(country_list, i))

for item in end_list:
    item0 = list(item)
    country_list_end.append(item0)

Value_list = []
country_list_end.append(country_list)

'''
end_list  :  [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3)]
里面元素可迭代，每个迭代器是元组
元组可以通过  list()  函数直接转成列表，方便判空和操作
最后可以加一个全部值的列表: [1, 2, ……, n]
'''

w_total = []
name_value_list = []
for item in country_list_end:
    # 找到对应城市名的 index ,通过  index  定位X、Y
    X_temp = []
    Y = 0
    X_max = 0
    for country_name in item:
        index = country_list.index(country_name)
        X_val = X_list[index]
        X_temp.append(X_val)
    if len(X_temp) == 0:
        X_max = 0
    else:
        X_max = max(X_temp)
    for country_name in item:
        index = country_list.index(country_name)
        y_val = function_list[index][0] * X_max + function_list[index][1]
        Y += y_val

    '''
    求权重 w: (len(country)-1)!(11-len(country))!/11!
    math.factorial 求阶乘
    '''
    if len(item) == 0:
        pass
    else:
        w = math.factorial(len(item) - 1) * math.factorial(11 - len(item)) / math.factorial(11)
        w_total.append(w)

    if len(item) == 0:
        pass
    else:
        # print("Country set {}'s value is: {}, max of the x is: {}, w value is {}".format(item, Y, X_max, w))
        name_value_list.append([item, Y, w])
    Value_list.append(Y)

'''
上面计算出各个组合的value值是Y，对应权重是w
接下来遍历求边际贡献 contribution : v(s)-v(s-i)
'''

name_value_array = np.array(name_value_list)
name_list = list(name_value_array[:, 0])
shapely_list = []

# 重要问题:每次运行之后会把 "福建" 都删除
for item in name_value_list:
    if "福建" in item[0]:
        if item[0] == ["福建"]:
            w = name_value_list[name_value_list.index(item)][2]
            contribution_shanghai = item[1]
            shapely = w * contribution_shanghai
            shapely_list.append(shapely)
        else:
            w = name_value_list[name_value_list.index(item)][2]
            item[0].remove("福建")
            index = name_list.index(item[0])
            contribution_shanghai = item[1] - name_value_list[index][1]
            shapely = w * contribution_shanghai
            shapely_list.append(shapely)

df = pd.DataFrame(shapely_list, columns=["Fujian"])
df.to_csv(r"Fujian.csv", index=False, encoding="UTF-8-SIG")

