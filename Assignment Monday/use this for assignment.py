# -*- coding: utf-8 -*-
"""
Created on Mon Aug 31 14:57:45 2020

@author: keshav
"""
n=int(input("enter a number"))
sum=0
while(n>0):
  t=n%10
  sum=sum+t
  n=n//10
print(sum)


    
