import os
import cv2
import numpy as np
from numpy.lib.function_base import disp
import torch
import decord
import pickle
from PIL import Image
from torchvision import transforms
from datasets.transforms.random_erasing import RandomErasing
import warnings
from torch.utils.data import Dataset
import random
import datasets.transforms.video_transforms as video_transforms
import datasets.transforms.volume_transforms as volume_transforms

def spatial_sampling(
    frames,
    spatial_idx=-1,
    min_scale=256,
    max_scale=320,
    crop_size=224,
    random_horizontal_flip=True,
    inverse_uniform_sampling=False,
    aspect_ratio=None,
    scale=None,
    motion_shift=False,
):
    """
    Perform spatial sampling on the given video frames. If spatial_idx is
    -1, perform random scale, random crop, and random flip on the given
    frames. If spatial_idx is 0, 1, or 2, perform spatial uniform sampling
    with the given spatial_idx.
    Args:
        frames (tensor): frames of images sampled from the video. The
            dimension is `num frames` x `height` x `width` x `channel`.
        spatial_idx (int): if -1, perform random spatial sampling. If 0, 1,
            or 2, perform left, center, right crop if width is larger than
            height, and perform top, center, buttom crop if height is larger
            than width.
        min_scale (int): the minimal size of scaling.
        max_scale (int): the maximal size of scaling.
        crop_size (int): the size of height and width used to crop the
            frames.
        inverse_uniform_sampling (bool): if True, sample uniformly in
            [1 / max_scale, 1 / min_scale] and take a reciprocal to get the
            scale. If False, take a uniform sample from [min_scale,
            max_scale].
        aspect_ratio (list): Aspect ratio range for resizing.
        scale (list): Scale range for resizing.
        motion_shift (bool): Whether to apply motion shift for resizing.
    Returns:
        frames (tensor): spatially sampled frames.
    """
    assert spatial_idx in [-1, 0, 1, 2]
    if spatial_idx == -1:
        if aspect_ratio is None and scale is None:
            frames, _ = video_transforms.random_short_side_scale_jitter(
                images=frames,
                min_size=min_scale,
                max_size=max_scale,
                inverse_uniform_sampling=inverse_uniform_sampling,
            )
            frames, _ = video_transforms.random_crop(frames, crop_size)
        else:
            transform_func = (
                video_transforms.random_resized_crop_with_shift
                if motion_shift
                else video_transforms.random_resized_crop
            )
            frames = transform_func(
                images=frames,
                target_height=crop_size,
                target_width=crop_size,
                scale=scale,
                ratio=aspect_ratio,
            )
        if random_horizontal_flip:
            frames, _ = video_transforms.horizontal_flip(0.5, frames)
    else:
        # The testing is deterministic and no jitter should be performed.
        # min_scale, max_scale, and crop_size are expect to be the same.
        assert len({min_scale, max_scale, crop_size}) == 1
        frames, _ = video_transforms.random_short_side_scale_jitter(
            frames, min_scale, max_scale
        )
        frames, _ = video_transforms.uniform_crop(frames, crop_size, spatial_idx)
    return frames


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor


class PhaseDataset_AutoLaparo(Dataset):
    """Load video phase recognition dataset."""

    def __init__(
        self,
        anno_path="data/AutoLaparo/labels_pkl/train/1fpstrain.pickle",
        data_path="data/AutoLaparo",
        mode="train",  # val/test
        data_strategy="online",  # offline
        output_mode="key_frame",  # all_frame
        cut_black=False,
        clip_len=16,
        frame_sample_rate=2,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
        crop_size=224,
        short_side_size=256,
        new_height=256,
        new_width=340,
        keep_aspect_ratio=True,
        args=None,
    ):
        self.anno_path = anno_path
        self.data_path = data_path
        self.mode = mode
        self.data_strategy = data_strategy
        self.output_mode = output_mode
        self.cut_black = cut_black
        self.clip_len = clip_len
        self.frame_sample_rate = frame_sample_rate
        self.crop_size = crop_size
        self.short_side_size = short_side_size
        self.new_height = new_height
        self.new_width = new_width
        self.keep_aspect_ratio = keep_aspect_ratio
        self.args = args

        self.frame_span = self.clip_len * self.frame_sample_rate

        # Augment
        self.aug = False
        self.rand_erase = False
        if self.mode in ["train"]:
            self.aug = True
            if self.args.reprob > 0:  # default: 0.25
                self.rand_erase = True
        self.infos = pickle.load(open(self.anno_path, "rb"))
        self.dataset_samples = self._make_dataset(self.infos)

        if mode == "train":
            pass

        elif mode == "val":
            self.data_transform = video_transforms.Compose(
                [
                    video_transforms.Resize(
                        (self.short_side_size, self.short_side_size),
                        interpolation="bilinear",
                    ),
                    # video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )
        elif mode == "test":
            self.data_resize = video_transforms.Compose(
                [
                    video_transforms.Resize(
                        size=(short_side_size, short_side_size),
                        interpolation="bilinear",
                    ),
                    # video_transforms.CenterCrop(size=(self.crop_size, self.crop_size)),
                ]
            )
            self.data_transform = video_transforms.Compose(
                [
                    volume_transforms.ClipToTensor(),
                    video_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            )

    def __getitem__(self, index):
        if self.mode == "train":
            args = self.args
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
            )
            if self.data_strategy == "online":
                buffer, phase_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, False
                )  # T H W C
            elif self.data_strategy == "offline":
                (
                    buffer,
                    phase_labels,
                    sampled_list,
                ) = self._video_batch_loader_for_key_frames(
                    frames, frame_id, video_id, index, False
                )  # T H W C

            buffer = self._aug_frame(buffer, args)

            if self.output_mode == "key_frame":
                if self.data_strategy == "offline":
                    return (
                        buffer,
                        phase_labels[self.clip_len // 2],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        {},
                    )
                elif self.data_strategy == "online":
                    return (
                        buffer,
                        phase_labels[-1],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        {},
                    )
            elif self.output_mode == "all_frame":
                return (
                    buffer,
                    phase_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                    {},
                )

        elif self.mode == "val":
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
            )
            if self.data_strategy == "online":
                buffer, phase_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C
            elif self.data_strategy == "offline":
                (
                    buffer,
                    phase_labels,
                    sampled_list,
                ) = self._video_batch_loader_for_key_frames(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C

            buffer = self.data_transform(buffer)

            if len(sampled_list) == len(np.unique(sampled_list)):
                flag = False
            else:
                flag = True

            if self.output_mode == "key_frame":
                if self.data_strategy == "offline":
                    return (
                        buffer,
                        phase_labels[self.clip_len // 2],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                    )
                elif self.data_strategy == "online":
                    return (
                        buffer,
                        phase_labels[-1],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                    )
            elif self.output_mode == "all_frame":
                return (
                    buffer,
                    phase_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                    flag,
                )

        elif self.mode == "test":
            frames_info = self.dataset_samples[index]
            video_id, frame_id, frames = (
                frames_info["video_id"],
                frames_info["frame_id"],
                frames_info["frames"],
            )
            if self.data_strategy == "online":
                buffer, phase_labels, sampled_list = self._video_batch_loader(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C
            elif self.data_strategy == "offline":
                (
                    buffer,
                    phase_labels,
                    sampled_list,
                ) = self._video_batch_loader_for_key_frames(
                    frames, frame_id, video_id, index, self.cut_black
                )  # T H W C

            # dim = (int(buffer[0].shape[1] / buffer[0].shape[0] * 300), 300)
            # buffer = [cv2.resize(frame, dim) for frame in buffer]
            # buffer = [self.filter_black(frame) for frame in buffer]
            buffer = self.data_resize(buffer)
            if isinstance(buffer, list):
                buffer = np.stack(buffer, 0)

            buffer = self.data_transform(buffer)

            if len(sampled_list) == len(np.unique(sampled_list)):
                flag = False
            else:
                flag = True

            if self.output_mode == "key_frame":
                if self.data_strategy == "offline":
                    return (
                        buffer,
                        phase_labels[self.clip_len // 2],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                    )
                elif self.data_strategy == "online":
                    return (
                        buffer,
                        phase_labels[-1],
                        str(index) + "_" + video_id + "_" + str(frame_id),
                        flag,
                    )
            elif self.output_mode == "all_frame":
                return (
                    buffer,
                    phase_labels,
                    str(index) + "_" + video_id + "_" + str(frame_id),
                    flag,
                )
        else:
            raise NameError("mode {} unkown".format(self.mode))

    def filter_black(self, image):
        binary_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image2 = cv2.threshold(binary_image, 15, 255, cv2.THRESH_BINARY)
        binary_image2 = cv2.medianBlur(
            binary_image2, 19
        )  # filter the noise, need to adjust the parameter based on the dataset
        x = binary_image2.shape[0]
        y = binary_image2.shape[1]

        edges_x = []
        edges_y = []
        for i in range(x):
            for j in range(10, y - 10):
                if binary_image2.item(i, j) != 0:
                    edges_x.append(i)
                    edges_y.append(j)

        if not edges_x:
            return image

        left = min(edges_x)  # left border
        right = max(edges_x)  # right
        width = right - left
        bottom = min(edges_y)  # bottom
        top = max(edges_y)  # top
        height = top - bottom

        pre1_picture = image[left : left + width, bottom : bottom + height]
        return pre1_picture

    def _aug_frame(
        self,
        buffer,
        args,
    ):
        aug_transform = video_transforms.create_random_augment(
            input_size=(self.crop_size, self.crop_size),
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
        )
        # if self.cut_black:
        #     dim = (int(buffer[0].shape[1] / buffer[0].shape[0] * 300), 300)
        #     buffer = [cv2.resize(frame, dim) for frame in buffer]
        #     buffer = [self.filter_black(frame) for frame in buffer]
        #     buffer = [cv2.resize(frame, (250, 250)) for frame in buffer]

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]
        buffer = aug_transform(buffer)

        # for k in range(len(buffer)):
        #     img = cv2.cvtColor(np.asarray(buffer[k]), cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(k), img)
        #     cv2.waitKey()

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        # T H W C
        buffer = tensor_normalize(buffer, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        # T H W C -> C T H W.
        buffer = buffer.permute(3, 0, 1, 2)
        # Perform data augmentation.
        scl, asp = (
            [0.7, 1.0],
            [0.75, 1.3333],
        )

        buffer = spatial_sampling(
            buffer,
            spatial_idx=-1,
            min_scale=256,
            max_scale=320,
            crop_size=self.crop_size,
            random_horizontal_flip=True,
            inverse_uniform_sampling=False,
            aspect_ratio=asp,
            scale=scl,
            motion_shift=False,
        )

        if self.rand_erase:
            erase_transform = RandomErasing(
                args.reprob,
                mode=args.remode,
                max_count=args.recount,
                num_splits=args.recount,
                device="cpu",
            )
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        # Vis
        # for k in range(buffer.shape[1]):
        #     img = cv2.cvtColor(np.asarray(buffer[:,k,:,:]).transpose(1,2,0), cv2.COLOR_RGB2BGR)
        #     cv2.imshow(str(k), img)
        #     cv2.waitKey()
        return buffer

    def _make_dataset(self, infos):
        frames = []
        for video_id in infos.keys():
            data = infos[video_id]
            for line_info in data:
                # line format: unique_id, frame_id, video_id, tool_gt, phase_gt, phase_name, fps, frames
                if len(line_info) < 8:
                    raise (
                        RuntimeError(
                            "Video input format is not correct, missing one or more element. %s"
                            % line_info
                        )
                    )
                img_path = os.path.join(
                    self.data_path,
                    "frames",
                    line_info["video_id"],
                    str(line_info["original_frame_id"]).zfill(5) + ".png"
                    if "original_frame_id" in line_info
                    else str(line_info["frame_id"]).zfill(5) + ".png",
                )
                line_info["img_path"] = img_path
                frames.append(line_info)
        return frames

    def _video_batch_loader(self, duration, indice, video_id, index, cut_black):
        offset_value = index - indice
        frame_sample_rate = self.frame_sample_rate
        sampled_list = []
        frame_id_list = []
        for i, _ in enumerate(range(0, self.clip_len)):
            frame_id = indice
            frame_id_list.append(frame_id)
            if self.frame_sample_rate == -1:
                frame_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                frame_sample_rate = 2**i
            elif self.frame_sample_rate == -2:
                frame_sample_rate = 1 if 2 * i == 0 else 2 * i
            if indice - frame_sample_rate >= 0:
                indice -= frame_sample_rate
        sampled_list = sorted([i + offset_value for i in frame_id_list])
        sampled_image_list = []
        sampled_label_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                image_name_list.append(self.dataset_samples[image_index]["img_path"])
                path = self.dataset_samples[image_index]["img_path"]
                if cut_black:
                    path = path.replace('frames', 'frames_cutmargin')
                image_data = Image.open(path)
                phase_label = self.dataset_samples[image_index]["phase_gt"]
                # PIL可视化
                # image_data.show()
                # cv2可视化
                # img = cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2BGR)
                # cv2.imshow(str(num), img)
                # cv2.waitKey()
                sampled_image_list.append(image_data)
                sampled_label_list.append(phase_label)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of path {} (Unique_id: {}).".format(
                        frame_id_list[num],
                        video_id,
                        self.dataset_samples[image_index]["img_path"],
                        image_index,
                    )
                )
        video_data = np.stack(sampled_image_list)
        phase_data = np.stack(sampled_label_list)

        return video_data, phase_data, sampled_list

    def _video_batch_loader_for_key_frames(self, duration, timestamp, video_id, index, cut_black):
        # 永远控制的只有对应帧序号和整个视频序列有效视频数目，不受采样FPS影响，根据标签映射回对应image path
        # 当前视频内帧序号为timestamp,
        # 当前数据集内帧序号为index
        # 为了保证偶数输入的前序帧以及后续帧数目保持一致，中间double了关键帧
        # 如果为奇数，则中间帧位于中间，但是3D卷积不适用于偶数kernel及stride
        right_len = self.clip_len // 2
        left_len = self.clip_len - right_len
        offset_value = index - timestamp

        # load right
        right_sample_rate = self.frame_sample_rate
        cur_t = timestamp
        right_frames = []
        if right_len == left_len:
            for i, _ in enumerate(range(0, right_len)):
                right_frames.append(cur_t)
                if self.frame_sample_rate == -1:
                    right_sample_rate = random.randint(1, 5)
                elif self.frame_sample_rate == 0:
                    right_sample_rate = 2**i
                elif self.frame_sample_rate == -2:
                    right_sample_rate = 1 if 2 * i == 0 else 2 * i
                if cur_t + right_sample_rate <= duration:
                    cur_t += right_sample_rate
        else:
            for i, _ in enumerate(range(0, right_len)):
                if self.frame_sample_rate == -1:
                    right_sample_rate = random.randint(1, 5)
                elif self.frame_sample_rate == 0:
                    right_sample_rate = 2**i
                elif self.frame_sample_rate == -2:
                    right_sample_rate = 1 if 2 * i == 0 else 2 * i
                if cur_t + right_sample_rate <= duration:
                    cur_t += right_sample_rate
                right_frames.append(cur_t)

        # load left
        left_sample_rate = self.frame_sample_rate
        cur_t = timestamp
        left_frames = []
        for j, _ in enumerate(range(0, left_len)):
            left_frames = [cur_t] + left_frames
            if self.frame_sample_rate == -1:
                left_sample_rate = random.randint(1, 5)
            elif self.frame_sample_rate == 0:
                left_sample_rate = 2**j
            elif self.frame_sample_rate == -2:
                left_sample_rate = 1 if 2 * j == 0 else 2 * j
            if cur_t - left_sample_rate >= 0:
                cur_t -= left_sample_rate

        frame_id_list = left_frames + right_frames
        assert len(frame_id_list) == self.clip_len
        sampled_list = [i + offset_value for i in frame_id_list]
        sampled_image_list = []
        sampled_label_list = []
        image_name_list = []
        for num, image_index in enumerate(sampled_list):
            try:
                image_name_list.append(self.dataset_samples[image_index]["img_path"])
                path = self.dataset_samples[image_index]["img_path"]
                if cut_black:
                    path = path.replace('frames', 'frames_cutmargin')
                image_data = Image.open(path)
                phase_label = self.dataset_samples[image_index]["phase_gt"]
                # PIL可视化
                # image_data.show()
                # cv2可视化
                # img = cv2.cvtColor(np.asarray(image_data), cv2.COLOR_RGB2BGR)
                # cv2.imshow(str(num), img)
                # cv2.waitKey()
                sampled_image_list.append(image_data)
                sampled_label_list.append(phase_label)
            except:
                raise RuntimeError(
                    "Error occured in reading frames {} from video {} of path {} (Unique_id: {}).".format(
                        frame_id_list[num],
                        video_id,
                        self.dataset_samples[image_index]["img_path"],
                        image_index,
                    )
                )
        video_data = np.stack(sampled_image_list)
        phase_data = np.stack(sampled_label_list)
        return video_data, phase_data, sampled_list

    def __len__(self):
        return len(self.dataset_samples)


def build_dataset(is_train, test_mode, fps, args):
    if args.data_set == "AutoLaparo":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl", mode, fps + "val.pickle")

        dataset = PhaseDataset_AutoLaparo(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy="online",
            output_mode="key_frame",
            cut_black=False,
            clip_len=8,
            frame_sample_rate=4,  # 0表示指数级间隔，-1表示随机间隔设置, -2表示递增间隔
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 7
    assert nb_classes == args.nb_classes
    print("%s %s - %s : Number of the class = %d" % ("AutoLaparo", mode, fps, args.nb_classes))

    return dataset, nb_classes