import numpy as np#导入numpy库并简写为np

#print(np.__version__,np.show_config())#打印numpy的版本号

z1=np.zeros(10);print(z1)#创造一个长度为十的空向量z1

z2=np.zeros((10,10));print(z2);print("%d bytes" %(z2.size*z2.itemsize))
#找到任何一个数组z2的内存大小

#print(np.info(np.add))#从命令行得到numpy中add函数的说明文档

z3=np.zeros(10);z3[4]=1;print(z3)#创造一个长度为十且第五个值为1的空向量

z5=np.arange(10,50);print(z5)#创造一个值域范围从10到49的向量

z6=np.arange(50);z6=z6[::-1];print(z6)#反转一个向量

z7=np.arange(9).reshape(3,3);print(z7)#创造一个3✖3并且值从0到8的矩阵

z8=[1,2,0,0,4,0];nz=np.nonzero(z8);print(nz)#找到数组中非零元素的位置索引

z9=np.eye(3);print(z9)#创造一个3*3的单位矩阵

z10=np.random.random((3,3,3));print("z10=",z10)#创造一个3*3*3的随机数组

z11=np.random.random((10,10));print("z11=",z11);print("max is %f." %(z11.max()),"min is %f." %(z11.min()) )#创建一个10*10的随机数组并找到它的最大值和最小值

z12=np.random.random(30);print("z12=", z12); print("mean of z12 is %f" %(z12.mean()))#创建一个长度为30的随机向量并找到它的平均值

z13=np.zeros((10,10));z13[0,:]=1;z13[:,0]=1;z13[-1,:]=1;z13[:,-1]=1;print("z13 is ",z13)#创造一个10*10数组，其中边界值为1，其余值为0

z14=np.ones((5,5));print("z14=", z14);z14=np.pad(z14,pad_width=1,mode='constant',constant_values=0);print(z14);

z15=np.random.random((4,4));print("z15=",z15);print(np.diag(1+z14,k=-1));
#创建一个5*5的矩阵，并设置值1，2，3，4落在其对角线下方位置

z16=np.zeros((8,8));z16[1::2,::2]=1;z16[::2,:1:2]=1;print(z16)
#创建一个8*8的矩阵，并设置成棋盘样式

z17=np.unravel_index(100,(6,7,8));print(z17)
#一个（6，7，8）形状的数组，其第100个元素的索引（x,y,z）是什么

z18=np.tile(np.array([[0,1],[1,0]]),(4,4));print("z18=", z18)
#创建一个8*8棋盘样式矩阵

z19=np.random.random((5,5));z19=(z19-z19.min())/(z19.max()-z19.min());
print("z19=", z19)
#对一个5*5的随机矩阵做归一化

z20=np.dtype([("r",np.ubyte,1),("g",np.ubyte,1),("b",np.ubyte,1),("a",np.ubyte,1)])
print("color is", z20)
#创建一个将颜色描述为（RGBA）四个无符号字节的自定义dtype

z21=np.dot(((1,3,51),(3,32,12),(223,22,2),(223,22,2),(223,22,2)),((331,23,23),(23,54,32),(23,54,32)))
print(z21)
#一个5*3的矩阵乘以3*2的矩阵

z22=np.arange(11);
#z22=(1,22,3,12,6);
print(z22)
z22[(z22>3)&(z22<8)]*=-1;
print("z22=", z22)
#给定一个一维数组，对其在3到8之间的所有元素取反

z23=np.random.uniform(-10,+10,10);
print("z23 is", z23)
print(np.copysign(np.ceil(np.abs(z23)),z23))
#对从零位对浮点数组做舍入

z24=np.random.randint(0,10,10);
z25=np.random.randint(0,10,10),
print("common items are", np.intersect1d(z24,z25))
#如何找到两个数组中的共同元素

yesterday=np.datetime64('today',"D")-np.timedelta64(1,"D")
today=np.datetime64('today',"D")
tomorrow=np.datetime64('today',"D")+np.timedelta64(1,"D")
print("yesterday is", yesterday,"\n", "today is ",today,"\n","tomorrow is", tomorrow,"\n" )
#得到昨天、今天和明天的日期

z26=np.arange("2016-07",'2016-08',dtype='datetime64[D]')
print(z26)
#得到所有与2016年7月对应的日期

z27=np.ones(3)*1;z28=np.ones(3)*2;z29=np.ones(3)*3
np.add(z27,z28,out=z28);np.negative(z27,out=z27);np.divide(z27,2,out=z27)
np.multiply(z27,z28,out=z27);print("(A+B)\*(-A/2) is", z27)
#直接在位计算(A+B)\*(-A/2)

z30=np.random.uniform(0,10,10)
print("1 is ",z30-z30%1)#第一种方法
print("2 is", np.floor(z30))
print("3 is", np.ceil(z30)-1)
print("4 is", z30.astype(int))
print("5 is", np.trunc(z30))
#用五种不同方法去提取一个随机数组的整数部分

z31=np.zeros((5,5));z31+=np.arange(5);print("z31 is", z31)
#创建一个5*5的矩阵，其中每行的数值范围从0到4

def generate():
    for x in range(10):
        yield x
z32=np.fromiter(generate(),dtype=float,count=-1);
print("z32 is", z32)
#考虑一个可生成10个整数的函数，构建一个数组

z33=np.linspace(0,1,11,endpoint=False)[1:]
print("z33 is", z33)
#创建一个长度为10的随机向量，其值域范围从0到1，但是不包括0和1

z34=np.arange(10);z35=np.add.reduce(z34)
print("beat sum is", z35)
#对于一个小数组，比np.sum更快方式求和

z36=np.random.randint(0,2,5);z37=np.random.randint(0,2,5);
print("comparison of values", np.allclose(z36,z37))
#对于两个随机数组A和B，检查他们是否相等或者采用array_equal方法

#z38=np.zeros(10);z38.flags.writeable=False;z38[0]=1;print(z38)
#创建一个只读数组

z39=np.random.random((10,2));x,y=z39[:,0],z39[:,1];
R=np.sqrt(x**2+y**2);T=np.arctan2(y,x);
print("R in z39 is", R,"\n","T is", T)
#将笛卡尔坐标下的一个10*2矩阵转换为极坐标形式
#将笛卡尔坐标下一个10*2的矩阵转换为极坐标形式

z40=np.random.random(40);
z40[z40.argmax()]=0;print("z40 is", z40)
#创建一个长度为10的向量，并将向量中最大值替换为1

z41=np.zeros((5,5),[('x',float),('y',float)]);
z41['x'],z41['y']=np.meshgrid(np.linspace(0,1,5),np.linspace(0,1,5));
print("z41 is", z41)
#创建一个结构化数组，并实现x和y坐标覆盖[0,1]×[0,1]区域

x=np.arange(8);y=np.ones(8);
z42=1.0/np.subtract.outer(x,y)
print("Cauchy matrix is", np.linalg.det(z42))
#给定两个数组X和Y，构造Cauchy矩阵

for dtype in [np.int8, np.int32,np.int64]:
    print(np.iinfo(dtype).min)
    print(np.iinfo(dtype).max)

for dtype in [np.float32,np.float64]:
    print(np.finfo(dtype).min)
    print(np.finfo(dtype).max)
    print(np.finfo(dtype).eps)
#打印每个numpy标量类型的最小值和最大值

#np.set_printoptions(threshold=np.nan)
z43=np.zeros((16,16))
print("all values in z43", z43)
#打印数组中的所有数值

z44=np.arange(100);z45=np.random.uniform(0,100)
index=(np.abs(z44-z45)).argmin()
print("closest value in z44 is", z44[index])
#给定标量，找到数组中最接近标量的值

z46=np.zeros(10,[('positon',[('x',float,1),('y',float,1)]),
                 ('color',[('r',float,1),('g',float,1),('b',float,1)])])
print("position and color are", z46)
#创建一个表示位置(x,y)和颜色(r,g,b)的结构化数组

z47=np.random.random((10,2));
x,y=np.atleast_2d(z47[:,0],z47[:,1])
D=np.sqrt((x-x.T)**2+(y-y.T)**2);
print("D in z47 is", D)
#对一个表示坐标形状为(100,2)的随机向量，找到点与点的距离
#import scipy
#import scipy.spatial
#D=scipy.spatial.distance.cdist(z47,z47);
#print(D)

#将一个32位的浮点数转换为对应的整数
z48=np.arange(10,dtype=np.float32);
z48=z48.astype(np.float32,copy=False);
print("z48 is", z48);

# z49=np.genfromtxt[1,2,3,4,5]
#读取以下文件

z50=np.arange(9).reshape(3,3)
for index, value in np.ndenumerate(z50):
    print(index,value)
for index in np.ndindex(z50.shape):
    print(index,z50[index])
#对于numpy数组，enumerate的等价操作

x,y=np.meshgrid(np.linspace(-1,1,10),np.linspace(-1,1,10))
D=np.sqrt(x*x+y*y)
sigma,mu=1.0,0.0;
G=np.exp(-((D-mu)**2/(2.0*sigma**2)));print("Gaussian is ",G)
#生成通用的二维Gaussian-like数组

n=10;p=3
z51=np.zeros((n,n));
np.put(z51,np.random.choice(range(n*n),p,replace=False),1)
print(z51)
#对于一个二维数组，如何在其内部随机放置p个元素

X=np.random.rand(5,10);
z52=X-X.mean(axis=1,keepdims=True);
print("z52 is ",z52)
#减去一个矩阵中的每一个行的平均值

z53=np.random.randint(0,10,(3,3));
print("unsorted is ",z53)
print("sorted is ",z53[z53[:,1].argsort()])
#如何通过第n列对一个数组进行排序

z54=np.random.randint(0,3,(3,10));
print("z54 is", (~z54.any(axis=0)).any())
#如何检查一个二维数组是否有空列

z55=np.random.uniform(0,1,10);
z56=0.5
z56=z55.flat[np.abs(z55-z56).argmin()]
print("z56 is", z56)
#从数组中的给定值中找出最近的值

z57=np.arange(3).reshape(3,1)
z58=np.arange(3).reshape(1,3)
it=np.nditer([z57,z58,None])
for x,y,z in it:
    z[...]=x+y
print("shape13 and 31 is", it.operands[2])
#用迭代器计算两个分别具有形状（1，3）和（3，1）的数组

class NameArray(np.ndarray):
    def __new__(cls, array, name="no name"):
        obj=np.asarray(array).view(cls)
        obj.name=name
        return obj
    def __array_finalize__(self,obj):
        if obj is None:return
        self.info=getattr(obj,'name','no name')

z59=NameArray(np.arange(10),"range_10")
print("z59.name is ",z59.name)
#创建一个具有name属性的数组类

z60=np.ones(10)
z61=np.random.randint(0,len(z60),20)
z60+=np.bincount(z61,minlength=len(z60))
print("z60 is ",z60)
#考虑一个给定的向量，对第二个向量索引的每个元素加1

z63=[1,2,3,4,5,6]
z62=[1,42,64,2,2,1]
z64=np.bincount(z62,z63)
print("z64 is", z64)
#根据索引列表（z62），将向量（z63）的元素累加到数组（z64）

w,h=16,16
z65=np.random.randint(0,2,(h,w,3)).astype(np.ubyte)
z66=z65[...,0]*(256*256)+z65[...,1]*356+z65[...,2]
z67=len(np.unique(z66))
print("z67 is ",z67)
#考虑一个（dtype=ubyte）的（w,h,3）图像，计算其唯一颜色的数量

z68=np.random.randint(0,10,(3,4,3,4))
sum=z68.sum(axis=(-2,-1))
print("sum is", sum)
#考虑一个四维数组，如何一次性计算出最后两个轴的和


z69=np.random.uniform(0,1,100)
z70=np.random.randint(0,10,100)
z69_sums=np.bincount(z70,weights=z69)
z69_counts=np.bincount(z70)
z69_means=z69_sums/z69_counts
print("z69 means is", z69_means)
#考虑一个一维向量z69，如何使用相同大小的向量z70来计算z69子集的均值

z71=np.random.uniform(0,1,(5,5))
z72=np.random.uniform(0,1,(5,5))
print("zz is ",np.diag(np.dot(z71,z72)))
#获得点积dot product的对角线


#考虑一个向量[1,2,3,4,5],如何建立一个新的向量z73，在这个新向量中每个值之间有三个连续的零
z74=np.array([1,2,3,4,5])
nz=3
z73=np.zeros(len(z74)+(len(z74)-1)*(nz))
z73[::nz+1]=z74
print("z73 is", z73)

z75=np.ones((5,5,3));
z76=2*np.ones((5,5))
print("multiple is ",z75*z76[:,:,None])
#一个维度（5，3，3）的数组与一个（5，5）的数组相乘

z77=np.arange(25).reshape(5,5)
z77[[0,1]]=z77[[1,0]]
print("z77 is ",z77)
#对数组中任何两行做交换

faces=np.random.randint(0,100,(10,3))
z78=np.roll(faces.repeat(2,axis=1),-1,axis=1)
z78=z78.reshape(len(z78)*3,2)
z78=np.sort(z78,axis=1)
z79=z78.view(dtype=[('p0',z78.dtype),('p1',z78.dtype)])
z79=np.unique(z79)
print("z79 is ",z79)
#考虑一个可以描述10个三角形的triplets，找到可以分割全部三角形的ilne segment

z80=np.bincount([1,1,2,3,4,4,6])
z81=np.repeat(np.arange(len(z80)),z80)
print("z81 is ",z81)
#给定一个二进制数组z80，产生一个数组z81满足np.bincount(z81)=z80

def moving_average(z82,n=3):
    ret=np.cumsum(z82,dtype=float)
    ret[n:]=ret[n:]-ret[:-n]
    return ret[n-1:]/n
z83=np.arange(20)
print("moving average is", moving_average(z83,n=3))
#通过滑动窗口计算一个数组的平均数

from numpy.lib import stride_tricks

def rolling(a,window):
    shape=(a.size-window+1,window)
    strides=(a.itemsize,a.itemsize)
    return stride_tricks.as_strided(a,shape=shape,strides=strides)

z84=rolling(np.arange(10),3)
print("z84 is",z84)
#Consider a one-dimensional array Z, build a two-dimensional array whose first row is (Z[0],Z[1],Z[2]) and each subsequent row is shifted by 1 (last row should be (Z[-3],Z[-2],Z[-1])
