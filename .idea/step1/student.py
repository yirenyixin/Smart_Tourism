import numpy as np
# 导入pandas库
import pandas as pd
import numpy as np



'''
读取原始数据并保存在data变量中
其中pd.read_csv函数用于读取指定的csv文件
函数的第一个参数为想要读取的csv文件的路径。
函数的第二个参数为读取csv文件时的字符集，由于csv文件中有中文，所以需要设置成utf8。



'''
data = pd.read_csv('D:\workspace\Smart_Tourism\.idea\step1\hotel_data.csv', encoding='utf-8')

'''
将data中有多少个用户统计出来并保存到users变量中
其中data.user_id表示原始数据中的user_id这一列
data.user_id.unique().shape[0]表示user_id这一列中有多少个不同的值
例如：
    data.user_id = [1, 2, 3, 4, 4, 3, 2, ]
    data.user_id.unique().shape[0]为4
'''
# users = data.user_id.unique().shape[0]
hotel_id = data['id'].astype(int).unique().tolist()
hotel_name=data['name'].unique()
# print(hotel_id)
user=data['user_id'].astype(int).unique().tolist()
# 将data中有多少个酒店统计出来并保存到items变量中
users = data.user_id.unique().shape[0]
# 将data中有多少个酒店统计出来并保存到items变量中
items = data.id.unique().shape[0]

def create_user_hotel_matrix(users, items, data, hotel_id):
    '''
    构建用户-酒店矩阵
    :param users: 用户数量，类型为整数
    :param items: 酒店数量，类型为整数
    :param data: 原始数据，类型为DataFrame
    :param hotel_id: 酒店ID的列表，类型为列表
    :return: user_hotel_matrix
    '''
    user_hotel_matrix = np.zeros((users, items))
    for line in data.itertuples():
        #********* Begin *********#
        # user_hotel_matrix[line[3], hotel_id.index(line[1])] = line[4]
        user_hotel_matrix[user.index(line[3]), hotel_id.index(line[1])] = line[4]
        #********* End *********#
    return user_hotel_matrix
data=create_user_hotel_matrix(users, items, data, hotel_id)



m,n = data.shape
'''
np.random.uniform函数用来从指定范围内按照均匀分布来采样并使用采样得到的数值来初始化指定形状的矩阵。
函数的第一个参数为范围的最小值
函数的第二个参数为范围的最大值
函数的第三个参数为需要初始化的矩阵的形状
所以这行代码的意思是构造一个m行d列的矩阵，矩阵中的值通过0-1均匀分布来初始化。
'''
d=5
B = np.random.uniform(0,1,(m,d))
'''
用和上面一样的方式初始化矩阵C
'''
C = np.random.uniform(0,1,(d,n))

#第二关
'''
np.array函数是用来构造一个矩阵
函数的第一个参数是想要构造的矩阵的值，data>0会得到一个布尔类型的矩阵，即将data中大于0的部分变成True，否则变成False。
如data中的值为：
[[22.3, 0],
 [0, 11.6]]
那么data>0的值为：
[[True, False],
 [False, True]]
 
函数的第二个参数是指定想要构造的矩阵的值的类型
这行代码的意思是将用户-酒店矩阵大于0的值改为1，并将这个矩阵保存到record变量中
'''
record = np.array(data>0, dtype=int)
n_iter=10
alpha=0.01
lr=0.01
# 总共更新n_iter次
for i in range(n_iter):
    '''
    根据公式计算loss对B的偏导
    其中np.dot用来实现矩阵相乘，比如np.dot(B, C)表示矩阵B乘以矩阵C
    
    np.multiply用来实现两个矩阵中值对应相乘。
    例如:
        B=[[1, 2], 
           [3, 4]]
        C=[[0, 1],
           [2, 1]]
        np.multiply(B, C)的结果为
        [[0, 2],
         [6, 4]]
         
    C.T表示矩阵C的转置
    '''
    B_grads = np.dot(np.multiply(record, np.dot(B,C)-data),C.T)
    # 用和上面一样的方式按公式计算loss对C的偏导
    C_grads = np.dot(B.T, np.multiply(record,np.dot(B,C)-data))
    # 根据公式更新矩阵B和矩阵C
    B = alpha*B - lr*B_grads
    C = alpha*C - lr*C_grads

# 计算矩阵A， 其中np.dot(B，C)表示矩阵B乘以矩阵C
A = np.dot(B, C)
def recommend_hotel(A, userid):
    '''
    向用户id为userid的用户推荐3家酒店
    :param A: 已经更新好了的矩阵A
    :param userid: 待推荐的userid，类型为整数
    :return: recommend
    '''
    #********* Begin *********#
    # 对矩阵A中userid对应的行进行升序排序
    ranklist =np.argsort(A[userid])
    #********* End *********#
    recommend = ranklist[-1:-4:-1]
    return recommend[-1], recommend[-2], recommend[-3]
print(user)
# print(hotel_name.shape)
# print(hotel_name)
# print(hotel_id)
# print(len(hotel_id))
for i in range(len(user)):
    x=recommend_hotel(A,i)
    print("向用户"+str(i)+"推荐："+hotel_name[x[0]]+","+hotel_name[x[1]]+","+hotel_name[x[2]])