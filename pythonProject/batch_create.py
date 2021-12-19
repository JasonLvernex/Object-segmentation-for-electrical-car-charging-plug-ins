# _*_ coding:utf-8 _*_
#modified by 吕添 04/11/2021
import os
path = 'C:\\Users\\lenovo\\Desktop\\331-4\\'  # path是你存放json的路径
json_file = os.listdir(path)
os.system("activate labelme")
for file in json_file:
    os.system(" F:\\acanda\\envs\\labelme\\Scripts\\labelme_json_to_dataset.exe %s"%(path + file)) #this is the path where you intalled labelme module
