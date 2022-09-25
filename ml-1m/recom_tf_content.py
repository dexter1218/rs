#!/usr/bin/env python
# coding: utf-8

# In[129]:


import pandas as pd
import numpy as np 
import tensorflow.compat.v1 as tf

rnames = ["UserID", "MovieID", "Rating", "TimeStamp"]
ratings = pd.read_table("C:\\Users\\dexter\\Desktop\\ml-1m\\ratings.dat", sep="::", header=None, names=rnames, engine='python')
print(ratings[:5])

unames = ["UserID", "Gender", "Age", "Occupation", "Zip-code"]
users = pd.read_table("C:\\Users\\dexter\\Desktop\\ml-1m\\users.dat", sep="::", header=None, names=unames, engine='python')
print(users[:5])

mnames = ["MovieID", "Title", "Genres"]
movies = pd.read_table("C:\\Users\\dexter\\Desktop\\ml-1m\\movies.dat", sep="::", header=None, names=mnames, engine='python', encoding='ISO-8859-1')
print(movies[:5])


# In[130]:


data = pd.merge(ratings, users, on='UserID')
print(data[:5])


# In[131]:


data1 = pd.merge(ratings,movies,on='MovieID')
print(data1[:5])


# In[132]:


ratings_df = data1[['UserID','MovieID','Rating']]
ratings_df.to_csv('ratingsProcessed.csv', index=False, header=True, encoding='utf-8')
ratings_df.head()


# In[140]:


user, movie = [], []
for index, row in ratings_df.iterrows():
    if row['UserID'] not in user:
        user.append(row['UserID'])
    if row['MovieID'] not in movie:
        movie.append(row['MovieID'])
print(user[:5],movie[:5])


# In[147]:


user.sort()
movie.sort()
m ,n = max(user), max(movie)
rating = np.zeros((max(movie), max(user)))


# In[148]:


for index, row in ratings_df.iterrows():
    rating[int(row['MovieID'])-1, int(row['UserID'])-1] = row['Rating']
    


# In[149]:


#电影评分表中，>0表示已经评分，=0表示未被评分
record = rating > 0
#bool值转换为0和1
record = np.array(record, dtype=int)


# In[150]:


def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))  #初始化对于每部电影每个用户的平均评分
    rating_norm = np.zeros((m, n))  #保存处理后的数据
    #原始评分-平均评分，最后将计算结果和平均评分返回。
    for i in range(m):
        idx = record[i, :] != 0  #获取已经评分的电影的下标
        rating_mean[i] = np.mean(rating[i,  idx])  #计算平均值，右边式子代表第i行已经评分的电影的平均值
        rating_norm[i, idx] = rating[i, idx] - rating_mean[i]
    return rating_norm, rating_mean
    
rating_norm, rating_mean = normalizeRatings(rating, record)

rating_norm = np.nan_to_num(rating_norm)
rating_mean = np.nan_to_num(rating_mean)


# In[151]:


liss=[]
for index,row in movies.iterrows():
    lis = row['Genres'].split('|')
    for i in lis :
        if i not in liss:
            liss.append(i)
liss


# In[152]:


num_features = 18
#初始化电影矩阵X，用户喜好矩阵Theta,这里产生的参数都是随机数，并且是正态分布
X_parameters = tf.Variable(tf.random_normal([n, num_features], stddev=0.35))
Theta_paramters = tf.Variable(tf.random_normal([m, num_features], stddev=0.35))
#理论课定义的代价函数
#tf.matmul(X_parameters, Theta_paramters, transpose_b=True)代表X_parameters和Theta_paramters的转置相乘
loss = 1/2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_paramters, transpose_b=True)
                             - rating_norm) * record) ** 2) \
       + 1/2 * (tf.reduce_sum(X_parameters**2)+tf.reduce_sum(Theta_paramters**2))  #正则化项，这里λ=1，可以调整来观察模型性能变化。

#创建优化器和优化目标
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(loss)


# In[155]:


#创建tensorflow绘画
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

#使用TensorFlow中的tf.summary模块，它用于将TensorFlow的数据导出，从而变得可视化，因为loss是标量，所以使用scalar函数
tf.summary.scalar('loss', loss)
#将所有summary信息汇总
summaryMerged = tf.summary.merge_all()
#定义保存信息的路径
filename = 'C:\\Users\\dexter\\Desktop\\ml-1m\\movie_tensorboard'
#把信息保存在文件当中
writer = tf.summary.FileWriter(filename)

type(summaryMerged)


# In[161]:


for i in range(1000):
    _, movie_summary = sess.run([train, summaryMerged])
    # 把训练的结果summaryMerged存在movie里
    writer.add_summary(movie_summary, i)
    # 把训练的结果保存下来


# In[162]:


Current_X_paramters, Current_Theta_parameters = sess.run([X_parameters, Theta_paramters])
#将电影内容矩阵和用户喜好矩阵相乘，再加上每一行的均值，便得到一个完整的电影评分表
predicts = np.dot(Current_X_paramters, Current_Theta_parameters.T) + rating_mean
#计算预测值与真实值的残差平方和的算术平方根，将它作为误差error,随着迭代次数增加而减少
errors = np.sqrt(np.sum((predicts - rating)**2))
errors


# In[163]:


user_id = 7
sortedResult = predicts[:, int(user_id)].argsort()[::-1]
idx = 0
#后边的center函数只是为了美观一点点，不重要
print('为该用户推荐的评分最高的20部电影是：'.center(80, '='))
for i in sortedResult:
    print('评分： %.2f, 电影名： %s' % (predicts[i, int(user_id)], movies.iloc[i]['Title']))
    idx += 1
    if idx == 20:
        break


# In[ ]:




