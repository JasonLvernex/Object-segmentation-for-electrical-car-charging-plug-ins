#this is rename script coded by Lyu Tian
#function: create dataset and masks into two files
import os, re
import sys
i=17#initial original file number

new_path_1="F:\\data\\training_input"
new_path_2="F:\\data\\training_mask"
count=15#the file initial rename number
for j in range(1,548):
    major_path = "F:\\data\\dataset\\fig%d_json" % i
    for item in os.listdir(major_path):
        # if (re.match("img\\.png", item)):
        # for items in item:
        if (re.match("img\\.png", item)):
            print(item)
            #filelist = os.listdir(major_path)
            #for files in filelist:
            Olddir = os.path.join(major_path, item)
            if os.path.isdir(Olddir):
                continue
            Newdir = os.path.join(new_path_1, "img%d.png" %count)
            newname = re.sub("img\\.png", "img%d.png"  %count, item)
            os.rename(Olddir, Newdir)
            print("-->" + newname)
        if (re.match("label\\.png", item)):
            print(item)
            Olddir = os.path.join(major_path, item)
            if os.path.isdir(Olddir):
                continue
            Newdir = os.path.join(new_path_2, "label%d.png" %count)
            newname = re.sub("label\\.png", "label%d.png" %count, item)
            os.rename(Olddir, Newdir)
            print("-->" + newname)
            count += 1
    i+=1

print("一共修改了"+str(count)+"个文件")
