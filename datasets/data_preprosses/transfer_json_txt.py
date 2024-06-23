import os
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
        return data
    
dataset_path = "/Users/yangshu/Downloads/CholecT50"
labels_path = "/Users/yangshu/Downloads/CholecT50/labels"
save_path = "/Users/yangshu/Downloads/CholecT50/verb"

if not os.path.exists(save_path):
    os.makedirs(save_path)

video_labels = sorted(os.listdir(labels_path))

total_num_triplet = 0

for video_label in video_labels:
    if "DS" in video_label:
        continue
    data_labels = dict()
    video_name = video_label.split('.')[0]
    txt_filename = os.path.join(save_path, video_name + ".txt")
    # 读取JSON文件
    label_path = os.path.join(labels_path, video_label)
    file_path = os.path.join(save_path, video_label)
    data = read_json_file(label_path)
    categpries = data['categories']
    instrument = categpries['instrument']
    verb = categpries['verb']
    target = categpries['target']
    triplet = categpries['triplet']
    phase = categpries['phase']

    rs = dict()
    anns = data['annotations']
    sorted_anns = dict(sorted(anns.items(), key=lambda x: int(x[0])))
    for k, v in sorted_anns.items():
        r = list()
        for instance in v:
            if instance[0] == -1:
                continue
            # triplet = instance[0]
            # r.append(str(triplet))
            # instrument = instance[1]
            # r.append(str(instrument))
            verb = instance[7]
            r.append(str(verb))
            # target = instance[8]
            # r.append(str(target))
            # phase = instance[14]
            # r.append(str(phase))

            # if len(r) >1:
            #     print(k,video_name)
        # if len(set(r)) != len(r):
        #     print(r)
        #     print('======')
        # rs[k] = set(r)
        rs[k] = r
    with open(txt_filename, "a") as txtfile:
            txtfile.write("Frame\tVerb\n")  # 写入标题行
            for frame, triplet in rs.items():
                if len(triplet) == 0:
                    target = -1
                else:
                    target = ",".join(triplet)
                txtfile.write(f"{frame}\t{target}\n")