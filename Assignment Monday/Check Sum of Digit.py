# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 18:42:53 2020

@author: keshav
"""

num = int(input(" Enter the Number To Check The Sum : "))

sum = 0

while(num>0):
    temp = num%10
    sum = sum+temp
    num = num//10

if(sum==7):
    print(" Sum of Digits is 7 ")  
else:
    print(" Sum Of Digits is not Equal to 7")    