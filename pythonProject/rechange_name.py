import os, re
old_path_1="C:\\Users\\lenovo\\Desktop\\training_input"
old_path_2="C:\\Users\\lenovo\\Desktop\\training_mask"
new_path1="C:\\Users\\lenovo\\Desktop\\new_origin"
new_path2="C:\\Users\\lenovo\\Desktop\\new_mask"
count=1
for item in os.listdir(old_path_1):
        # if (re.match("img\\.png", item)):
        # for items in item:

            #filelist = os.listdir(major_path)
            #for files in filelist:
            Olddir = os.path.join(old_path_1, item)
            if os.path.isdir(Olddir):
                continue
            Newdir = os.path.join(new_path1, "img%d.png" %count)
            newname = re.sub("img\\d+png", "img%d.png"  %count, item)
            os.rename(Olddir, Newdir)
            print("-->" + newname)
            count += 1
count=1
for item in os.listdir(old_path_2):
            Olddir = os.path.join(old_path_2, item)
            if os.path.isdir(Olddir):
                continue
            Newdir = os.path.join(new_path2, "label%d.png" %count)
            newname = re.sub("label\\d+png", "label%d.png" %count, item)
            os.rename(Olddir, Newdir)
            print("-->" + newname)
            count += 1


print("一共修改了"+str(count)+"个文件")