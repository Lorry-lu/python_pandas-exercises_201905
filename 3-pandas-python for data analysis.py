#chapter5
#Getting started with pandas
import numpy as np
import pandas as pd
#故事的开始
from pandas import DataFrame,Series

#5.1 introduction to pandas Data Structures

obj=pd.Series([3,33,4,23])
#index
print("index of obj is", obj,"\n")
print("obj value is", obj.values)
print("obj index is", obj.index)

obj2=pd.Series([323,21,323,43],index=["q","w","e","r"])
print("obj2 values is",obj2.values)
print("obj2 index is", obj2.index)
print("obj2 obj2>100 is", obj2[obj2>100])
print("obj2*2 is", obj2*2)
print("Wade in obj2", "wade" in obj2)

sdata={'Phil':32500,'Texas':41000,'Oregon':16000,'Utah':2000}
obj3=pd.Series(sdata)
print("obj3 is", obj3)
States=['California','Texas','Utah','Oregon']
print("obj3 states is",obj3[States])#好像这是个错误示范
obj4=pd.Series(sdata,index=States)
print("obj4 is", obj4)
print("obj4 is null is ",pd.isnull(obj4))
print("obj4 not null is", pd.notnull(obj4))
print("obj3+obj4 is", obj3+obj4)

#DataFrame
data={'state':["Ohio","Ohio","Ohio","Nevada","Nevada","Nevada"],
      'year':[2000,2001,2003,2001,2002,2003],
      'pop':[1.5,1.7,3.6,2.4,2.9,3.2]}
frame=pd.DataFrame(data)
print("frame is", frame)
print("frame head is", frame.head())
print("dataframe is", pd.DataFrame(data,columns=["year","state","pop"]))

#dict
frame2=pd.DataFrame(data,columns=['year','state','pop','debt']
                    ,index=['one','two','three','four','five','six'])
print("frame2 is",'\n', frame2)
print('frame2.columns\n',frame2.columns)
print('frame2.state\n',frame2['state'])
print("frame2.year\n",frame2.year)
print("frame2.loc[three] is\n",frame2.loc['three'])
frame2['debt']=16.5
print("frame2 is\n",frame2)
frame2['debt']=np.arange(6)
print("frame2[debt] is\n",frame2['debt'])
val=pd.Series([-1.2,-1.3,-1.4],index=['two','four','five'])
frame2['debt']=val
print("frame2 is\n",frame2)
#del
frame2['eastern']=frame2.state=="Ohio"
print('frame2 eastern is\n', frame2)
del frame2['eastern']
print("frame del eastern\n",frame2.columns)
#nested dict of dicts
pop={'Neveda':{2001:2.4,2002:2.9},
     'Ohio':{2000:1.5,2001:3.6}}
frame3=pd.DataFrame(pop)
print("frame3 is\n",frame3)
print('frame3.T is',frame3.T)#转置
print("pop,index\n",pd.DataFrame(pop,index=[2001,2002,2003]))
pdata={'Ohio':frame3['Ohio'][:-1],
       'Nevada':frame3['Neveda'][:2]}
print("pdata\n",pd.DataFrame(pdata))
frame3.index.name='year';frame3.columns.name='state'
print('index and columns is\n',frame3)
print('frame3 values',frame3.values)
print('frame2 values', frame2.values)

#index objects
obj5=pd.Series(range(3),index=['a','b','c'])
index=obj5.index
print('index is',index)
print('index 1: is',index[1:])
labels=pd.Index(np.arange(3))
print("labels is", labels)
obj6=pd.Series([1.5,-2.5,0],index=labels)
print('obj6 is',obj6)
#duplicate labels
dup_labels=pd.Index(['foo','foo','bar','bar'])
print('dup_labels is',dup_labels)


#5.2
#Essentional Functionality

#reindexing
obj7=pd.Series([4.5,7.2,-5.3,3.6],index=['d','b','a','c'])
print('obj7 is\n',obj7)
obj8=obj7.reindex(['a','b','c','d','e'])
print('obj8 is\n',obj8)
obj9=pd.Series(['blue','purple','yellow'],index=[0,2,4])
print('obj9 is\n',obj9)
print('obj9 reindex range 6\n',obj9.reindex(range(6),method='ffill'))

#DataFrame reindex
frame4=pd.DataFrame(np.arange(9).reshape(3,3),
                    index=['a','c','d'],
                    columns=['Ohio','Texas','California'])
print('frame4 is\n',frame4)
frame5=frame4.reindex(['a','b','c','d'])
print('frame5 is\n',frame5)

#dropping entries from an axis
obj10=pd.Series(np.arange(5),index=['a','b','c','d','e'])
print('obj10 is\n',obj10)
new_obj10=obj10.drop('c')
print('new_obj10 is\n',new_obj10)

data1=pd.DataFrame(np.arange(16).reshape((4,4)),
                   index=['Ohio','Colorado','Utah','New York'],
                   columns=['one','two','three','four'])
print('data1 is\n', data1)
data1.drop(['Colorado','Ohio'])
#print('data1 is\n',data1)
data1.drop('two',axis=1)
print('data1 is\n',data1)
#integer indexes
#arithmetic and data alignment
#function application and mapping
#sorting and ranking
#axis indexes with duplicate labels

#5.3 summarizing and computing descriptive statistics
#correlation and covariance
#unique values, value counts, and membership

#5.4 Conclusion
