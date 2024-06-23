import numpy as np
import os
phases = [
    "Incision",
    "Viscous agent injection",
    "Rhexis",
    "Hydrodissection",
    "Phacoemulsificiation",
    "Irrigation and aspiration",
    "Capsule polishing",
    "Lens implant setting-up",
    "Viscous agent removal",
    "Tonifying and antibiotics"
]
test_video_id = ['857', '861', '863', '865', '866', '867', '868', '871', '880', '882', '883', '884', '886', '887', '889', '890', '891', '892', '895', '896', '898', '899', '900', '901', '902', '906', '907', '908', '909', '911', '921', '922', '925', '926', '928', '929', '931', '932', '933', '934']
test_video_dict = {'857': 110, '861': 111, '863': 112, '865': 113, '866': 114, '867': 115, '868': 116, '871': 117, '880': 118, '882': 119, '883': 120, '884': 121, '886': 122, '887': 123, '889': 124, '890': 125, '891': 126, '892': 127, '895': 128, '896': 129, '898': 130, '899': 131, '900': 132, '901': 133, '902': 134, '906': 135, '907': 136, '908': 137, '909': 138, '911': 139, '921': 140, '922': 141, '925': 142, '926': 143, '928': 144, '929': 145, '931': 146, '932': 147, '933': 148, '934': 149}
id_to_video = dict(zip(test_video_dict.values(), test_video_dict.keys()))

def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        print("文件夹已创建：", folder_path)
    else:
        print("文件夹已存在：", folder_path)

main_path = "/Users/yangshu/Documents/PETL4SurgVideo/result_save/Ours/Cataract101/GA/"
file_path_0 = os.path.join(main_path, "0.txt")
file_path_1 = os.path.join(main_path, "1.txt")
anns_path = "/Users/yangshu/Documents/PETL4SurgVideo/result_save/Ours/Cataract101/GA" + "/phase_annotations"
pred_path = "/Users/yangshu/Documents/PETL4SurgVideo/result_save/Ours/Cataract101/GA" + "/prediction"


create_folder_if_not_exists(anns_path)
create_folder_if_not_exists(pred_path)

with open(file_path_0) as f:
    lines0 = f.readlines()

with open(file_path_1) as f:
    lines1 = f.readlines()


for i in range(110, 150):
    with open(
        anns_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        assert len(lines0) == len(lines1)
        video_name = id_to_video[i]
        for j in range(1, len(lines0)):
            temp0 = lines0[j].split()
            temp1 = lines1[j].split()
            if temp0[1] == "{}".format(str(video_name)):
                f.write(str(temp0[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp0[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations
            if temp1[1] == "{}".format(str(video_name)):
                f.write(str(temp1[2]))  # phase_annotations
                f.write("\t")  # phase_annotations
                f.write(str(temp1[-1]))  # phase_annotations
                f.write("\n")  # phase_annotations

with open(file_path_0) as f:
    lines0 = f.readlines()

with open(file_path_1) as f:
    lines1 = f.readlines()

# Just set a ID, which is different from Cholec80 and AutoLaparo for convenience
for i in range(110, 150):
    print(i)
    with open(
        pred_path + "/video-{}.txt".format(str(i)), "w"
    ) as f:  # phase_annotations
        f.write("Frame")
        f.write("\t")
        f.write("Phase")
        f.write("\n")
        assert len(lines0) == len(lines1)
        video_name = id_to_video[i]
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
            if temp0[1] == "{}".format(str(video_name)):
                f.write(str(temp0[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data0)) # prediction
                f.write('\n') # prediction
            if temp1[1] == "{}".format(str(video_name)):
                f.write(str(temp1[2])) # prediction
                f.write('\t') # prediction
                f.write(str(data1)) # prediction
                f.write('\n') # prediction
