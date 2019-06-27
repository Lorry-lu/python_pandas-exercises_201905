#chapter four
#numpy basics:arrays and vectorized computation

#4.1 多维数组对象
import numpy as np
#1 generate some random data
data=np.random.randn(2,3)
print("data is ", data)
print("data.shape is",data.shape)
print("data dtype is ",data.dtype)

data1=[6,7,8,9,1]
print("data1 is", data1)
arr1=np.array(data1)
print("arr1 is" ,arr1)
#1 创建n维数组

data2=[[1,2,3,4],[5,6,7,8]]
arr2=np.array(data2)
print("arr2 is", arr2)
print("arr2 ndim is", arr2.ndim)
print("arr2 shape is", arr2.shape)

#2 data type
print("arr1 dtype is", arr1.dtype)
print("arr2 dtype is", arr2.dtype)
print("np.zeros(10) is", np.zeros(10))
print("np.zeros((3,6)) is", np.zeros((3,6)))
print("np.empty((2,3,2)) is", np.empty((2,3,2)))
print("np.arange(10) is", np.arange(10))
arr3=np.array([1,2,3],dtype=np.float64)
print("float 64 is ",arr3.dtype)
arr4=np.array([3,4,1],dtype=np.int32)
print("int32 is",arr4.dtype)

arr5=np.array([1,3,4,0,3.0001])
print("arr5 is",arr5)
print("arr5 date type is", arr5.dtype)
#数据转换，把int变成float
arr5=arr5.astype(np.int32)
print("arr5 is", arr5)
print("arr5 data type is", arr5.dtype)

#向量化

#3 数组计算
arr6=np.array([[1,2,3],[4,5,6]])
print('arr6 is', arr6)
print('arr6 dtype is', arr6.dtype)
print('arr6*arr6 is',arr6*arr6)
arr7=np.array([[11,3,13],[1,34,33]])
print('arr7 is', arr7)
print('arr7 dtype is', arr7.dtype)
print('arr7 shape is',arr7.shape)
print('arr6>arr7 is', arr6>arr7)
print("arr6-arr7 is", arr6-arr7)

#4 slice(略)

#5 Boolean Indexing
names=np.array(["Af",'Ze','De','Fa','Wa'])
print("names are", names)
arr8=np.random.randn(5,4)
print("arr8 is", arr8)
print("arr8 is", arr8[names=="Fa"])
print("arr8 is",arr8[names=="Wa",2:4])

#fancy indexing

arr9=np.empty((8,4))#通过整数数组来索引
print("arr9 is",arr9)
for i in range(8):
    arr9[i]+=i

print("arr9 is", arr9)
print("arr9 4 3 0 6 is", arr9[[4,3,0,6]])
arr10=np.arange(32).reshape(4,8)
print("arr10 is", arr10)
print("arr10 1572 0312 is", arr10[[1,2,0,2],[0,3,1,2]])
print("arr10 1 3 0 2 maohao is",arr10[[1,3,0,2],])

#数组转置和轴交换
arr11=np.arange(15).reshape(5,3)
print("arr11 is",arr11)
print("arr transpose is", arr11.transpose((1,0)))
print("arr swapaxies is", arr11.swapaxes(1,0))
