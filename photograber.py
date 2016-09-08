# -*- coding: utf-8 -*-
"""
Created on Thu Sep 01 11:52:50 2016

@author: temp
"""

import re
import urllib
import numpy
import scipy
import matplotlib.pyplot as plt
import glob
import os

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
os.chdir(dname)

#%matplotlib qt


#      if aspect != '400':
#          imgf = numpy.empty([400*4,266*4,3])
#          for i in numpy.arange(0,301,20/4):
#              for j in numpy.arange(0,200,266/4):
#                      urllib.urlretrieve('http://marathon-photo.ru/members/image.php?id=' + idd + '&x1=0&x0=' + str(j) + '&y1=0&y0='+  str(i) +'&z=4&width=266&height=400','temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')       
#                      imgf[i*4:i*4+20,j*4:j*4+266,:] = scipy.misc.imread('temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')[0:20,0:266]
#                      if i >= 205:
#                          imgf[i*4+380:i*4+400,j*4:j*4+266,:] = scipy.misc.imread('temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')[380:400,0:266]
#                      os.remove('temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')
#          scipy.misc.imsave( idd + '.jpg', imgf)

race_number = 574   
a = urllib.urlopen('http://marathon-photo.ru/index.php?str=1&sphoto=on&search='+str(race_number)+'&competition=2016%20%D0%A2%D1%80%D0%B8%D0%B0%D1%82%D0%BB%D0%BE%D0%BD%20Ironstar%20%D0%9A%D0%B0%D0%B7%D0%B0%D0%BD%D1%8C%20113,%20%D1%81%D0%BF%D1%80%D0%B8%D0%BD%D1%82')
site_text = a.read()
list_of_id = re.findall('img.*?' +str(race_number) + '-\d*?-*?(\d+?).jpg',site_text,re.DOTALL)

for idd in list_of_id:
  a = urllib.urlopen('http://marathon-photo.ru/index.php?sphoto=on&competition=2016+%D0%A2%D1%80%D0%B8%D0%B0%D1%82%D0%BB%D0%BE%D0%BD+Ironstar+%D0%9A%D0%B0%D0%B7%D0%B0%D0%BD%D1%8C+113%2C+%D1%81%D0%BF%D1%80%D0%B8%D0%BD%D1%82&search=' + str(race_number))
  site_text = a.read()
  aspect = re.findall(idd + ".jpg',(\d\d\d)",site_text,re.DOTALL)[0]
  if aspect == '400':
      imgf = numpy.empty([266*4,400*4,3])
      for i in numpy.arange(0,301,50/4):
          for j in numpy.arange(0,200,266/4):
                  urllib.urlretrieve('http://marathon-photo.ru/members/image.php?id=' + idd + '&x1=0&x0=' + str(i) + '&y1=0&y0='+  str(j) +'&z=4&width=400&height=266','temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')       
                  imgf[j*4:j*4+266,i*4:i*4+50,:] = scipy.misc.imread('temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')[0:266,0:50]
                  if i >= 216:
                      imgf[j*4:j*4+266,i*4+350:i*4+400,:] = scipy.misc.imread('temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')[0:266,350:400]
                  os.remove('temp_' + str(j)+'_'+str(i)+'_'+str(idd)+'.jpg')
      scipy.misc.imsave( idd + '.jpg', imgf)
