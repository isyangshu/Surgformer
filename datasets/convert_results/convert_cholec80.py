import numpy as np
import os
phases = [
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderPackaging",
    "CleaningCoagulation",
    "GallbladderRetraction",
]

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已创建：", folder_path)
    else:
        print("文件夹已存在：", folder_path)

main_path = "/home/yangshu/Surgformer/results/Cholec80/"
file_path_0 = os.path.join(main_path, "0.txt")
file_path_1 = os.path.join(main_path, "1.txt")
anns_path = main_path + "/phase_annotations"
pred_path = main_path + "/prediction"


create_folder_if_not_exists(anns_path)
create_folder_if_not_exists(pred_path)

with open(file_path_0) as f:
    lines0 = f.readlines()

with open(file_path_1) as f:
    lines1 = f.readlines()

for i in range(41, 81):
    with open(
        anns_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        assert len(lines0) == len(lines1)
        for j in range(1, len(lines0)):
            temp0 = lines0[j].split()
            temp1 = lines1[j].split()
            if temp0[1] == "video{}".format(str(i)):
                f.write(str(temp0[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp0[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp1[1] == "video{}".format(str(i)):
                f.write(str(temp1[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp1[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations

with open(file_path_0) as f:
    lines0 = f.readlines()

with open(file_path_1) as f:
    lines1 = f.readlines()
for i in range(41, 81):
    print(i)
    with open(
        pred_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:  # phase_annotations
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        assert len(lines0) == len(lines1)
        for j in range(1, len(lines0)):
            temp0 = lines0[j].strip() # prediction
            temp1 = lines1[j].strip() # prediction
            data0 = np.fromstring(
                temp0.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data1 = np.fromstring(
                temp1.split("[")[1].split("]")[0], dtype=np.float32, sep=","
            ) # prediction
            data0 = data0.argmax() # prediction
            data1 = data1.argmax() # prediction
            temp0 = lines0[j].split()
            temp1 = lines1[j].split()
            if temp0[1] == "video{}".format(str(i)):
                f.write(str(temp0[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data0)) # prediction
                f.write('\n') # prediction
            if temp1[1] == "video{}".format(str(i)):
                f.write(str(temp1[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data1)) # prediction
                f.write('\n') # prediction
