#Графики

%matplotlib inline
import numpy as np
import math
import pandas as pd 
import matplotlib.pyplot as plt 
df = pd.read_csv('ex1data1.csv', header = None) #Загружаем данные из csv-файла в скрипт с помощью метода read_csv модуля pandas.
x, y = df[0], df[1]
x1=[1,25]
y1=[0,25]
#t = arange(0.0, 25.0, 0.01)
fig = plt.figure()                 
pl1 = plt.subplot(111) #с subplot нарисуем на этом же графике кривую, «соответствующую» данным, считанным из файла. Пусть эта будет линия, соответствующая графику функции y = 2*x-10

plt.plot(x, y, 'b*') #Используя метод plot визуализируем данные, считанные из файла.
pl1.plot(x1, y1, 'g-') 
#pl1.plot(t,(2*t-10),'g-')


#Алгоритм градиентного спуска для линейной регрессии с одной переменной.

def gradient_descent(X, Y, koef, n): #вычисляем theta0 и theta1 
    l = len(x)
    theta0, theta1 = 0, 0
    for i in range(n):
        sum1 = 0
        for i in range(l):
            sum1 += theta0 + theta1 * x[i] - y[i]
        res1 = theta0 - koef * (1 / l) * sum1

        sum2 = 0
        for i in range(l):
            sum2 += (theta0 + theta1 * x[i] - y[i]) * x[i]
        res2 = theta1 - koef * (1 / l) * sum2

        theta0, theta1 = res1, res2

    return theta0, theta1

x2 = [1, 25]
y2 = [0, 0]
t0, t1 = gradient_descent(x, y, 0.01, len(x))
y2[0] = t0 + x2[0] * t1
y2[1] = t0 + x2[1] * t1
plt.plot(x2, y2, 'y-')


#polyfit

numpy_x = np.array(x)
numpy_y = np.array(y)
numpy_t1, numpy_t0 = np.polyfit(numpy_x, numpy_y, 1)

num_y1 = [0, 0]
num_y1[0] = numpy_t0 + x1[0] * numpy_t1
num_y1[1] = numpy_t0 + x1[1] * numpy_t1
plt.plot(x1, num_y1, 'b-')

plt.plot(x, y, 'b*')
plt.plot(x2, y2, 'y-')
plt.plot(x1, num_y1, 'r-')



#Линейная регрессия со множеством переменных
def sq_error(sqx, sqy, f_x=None):  #Вычисление среднеквадратичной ошибки
    squared_error = []
    for i in range(len(sqx)):
        squared_error.append((f_x(sqx[i]) - sqy[i])**2)
    return sum(squared_error)


df2 = pd.read_csv('webtraffic.csv', header = None) 
l, k = df2[0], df2[1]

lis = list(l)
lis2 = list(k)

for i in range(len(lis2)):
    if math.isnan(lis2[i]):
        lis2[i] = 0
    else:
        lis2[i] = lis2[i]

plt.plot(l, k, 'm*')

np_x = np.array(lis)
np_y = np.array(lis2)

x2 = list(range(743))

th1_1, th0_1 = np.polyfit(np_x, np_y, 1)
th2_2, th1_2, th0_2 = np.polyfit(np_x, np_y, 2)
th3_3, th2_3, th1_3, th0_3 = np.polyfit(np_x, np_y, 3)
th4_4, th3_4, th2_4, th1_4, th0_4 = np.polyfit(np_x, np_y, 4)
th5_5, th4_5, th3_5, th2_5, th1_5, th0_5 = np.polyfit(np_x, np_y, 5)

f1 = lambda x: th1_1*x + th0_1
f2 = lambda x: th2_2*x**2 + th1_2*x + th0_2
f3 = lambda x: th3_3*x**3 + th2_3*x**2 + th1_3*x + th0_3
f4 = lambda x: th4_4*x**4 + th3_4*x**3 + th2_4*x**2 + th1_4*x + th0_4
f5 = lambda x: th5_5*x**5 + th4_5*x**4 + th3_5*x**3 + th2_5*x**2 + th1_5*x + th0_5

r1 = sq_error(lis, lis2, f1)
r2 = sq_error(lis, lis2, f2)
r3 = sq_error(lis, lis2, f3)
r4 = sq_error(lis, lis2, f4)
r5 = sq_error(lis, lis2, f5)

f6 = np.poly1d(np.polyfit(np_x, np_y, 1)) #Удобный класс, используемый для инкапсуляции «естественных» операций над полиномами,
                                          #чтобы указанные операции могли принять свою обычную форму в коде.
plt.plot(x2, f6(x2))
f7 = np.poly1d(np.polyfit(np_x, np_y, 2))
plt.plot(x2, f7(x2))
f8 = np.poly1d(np.polyfit(np_x, np_y, 3))
plt.plot(x2, f8(x2))
f9 = np.poly1d(np.polyfit(np_x, np_y, 4))
plt.plot(x2, f9(x2))
f10 = np.poly1d(np.polyfit(np_x, np_y, 5))
plt.plot(x2, f10(x2))

plt.plot(l, k, 'y*')
plt.plot(x2, f6(x2))
plt.plot(x2, f7(x2))
plt.plot(x2, f8(x2))
plt.plot(x2, f9(x2))
plt.plot(x2, f10(x2))

#Чуть не забыли

print("Ср.кв.откл.", r1)
print("Ср.кв.откл.", r2)
print("Ср.кв.откл.", r3)
print("Ср.кв.откл.", r4)
print("Ср.кв.откл.", r5)
