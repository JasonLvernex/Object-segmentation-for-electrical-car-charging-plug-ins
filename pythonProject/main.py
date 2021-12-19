# This is a sample Python script.


#this script is to practise the datatype

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import math as mt
#import tensorflow as tf
from math import cos
#import keras
# we can also import user define module like import temp no! .py

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def com_compatible(arg1,arg2):
    if type(arg1) == type(arg2):
        return arg1+arg2
    else:
        print("they are not compatible")



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    a=5#int
    print(type(a))
    a=(1,2,3)#tuple it's also a solitude element that we can not modify the individual element
    print (sum(a))
    print(type(a))
    a=[1,2,3,"s",(1,2,3),[3,5]]# this is a list, list is similar as arry, but different,1. element of the list can be anything,a[2] is the tuple
    #print(sum(a)) error
    print(type(a))
    a=4.1 #float
    print(type(a))
    a='123'#string
    print(type(a))
    com_compatible(1,"2")
    print(cos(3*mt.pi))
    aa=[1,2,3,4]
    print(list(reversed(aa)))
    print(aa[::-1])
    #print(sorted(aa,reverse=True)
    b=aa.copy()
    #c=input("plz write:")
    #print(type(c))
    #when coding a function, other variables willnot be changed, however, list will be altered when coded in the function
    def add_a(k):
        k=k+1
        print(id(k))
    k=3
    print(id(k))
    add_a(k)
    print(k)
    def add_list(kl):
        kl[0]=kl[0]+1#why can modify the list?
    kl=[1,2,3,4]
    add_list(kl)
    print(kl)
    st_o="I am a good Professor"
    I=st_o[0];
    am=st_o[2:4]
    a=st_o[5]
    good=st_o[7:11]
    professor=st_o[12:]
    st_d = "Professor good a am I"
    st_t=professor+" "+good+' '+a+' '+am+' '+I
    print(st_t)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
