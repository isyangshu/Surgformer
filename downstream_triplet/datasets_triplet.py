# import os
# from datasets.transforms import *
# from datasets.transforms.surg_transforms import *

# from datasets.triplet.CholecT50_triplet import TripletDataset_CholecT50

# def build_dataset(is_train, test_mode, fps, args):
#     """Load video triplet recognition dataset."""
#     if args.data_set == "CholecT50":
#         mode = None
#         anno_path = None
#         if is_train is True:
#             mode = "train"
#             anno_path = os.path.join(
#                 args.data_path, "labels_pkl", mode, fps + "train.pickle"
#             )
#             dataset = TripletDataset_CholecT50(
#                 anno_path=anno_path,
#                 data_path=args.data_path,
#                 mode=mode,
#                 data_strategy=args.data_strategy,
#                 output_mode=args.output_mode,
#                 clip_len=args.num_frames,
#                 frame_sample_rate=args.sampling_rate,
#                 keep_aspect_ratio=True,
#                 crop_size=args.input_size,
#                 short_side_size=args.short_side_size,
#                 new_height=256,
#                 new_width=320,
#                 args=args,
#             )
#         elif test_mode is True:
#             mode = "test"
#             anno_path = os.path.join(args.data_path, "labels_pkl", mode, "single")
#             pks = os.listdir(anno_path)
#             dataset_ = list()
#             for pk in pks:
#                 pk_path = os.path.join(anno_path, pk)
#                 dataset = TripletDataset_CholecT50(
#                     anno_path=pk_path,
#                     data_path=args.data_path,
#                     mode=mode,
#                     data_strategy=args.data_strategy,
#                     output_mode=args.output_mode,
#                     clip_len=args.num_frames,
#                     frame_sample_rate=args.sampling_rate,
#                     keep_aspect_ratio=True,
#                     crop_size=args.input_size,
#                     short_side_size=args.short_side_size,
#                     new_height=256,
#                     new_width=320,
#                     args=args,
#                 )
#                 dataset_.append(dataset)
#             dataset = dataset_
#         else:
#             mode = "val"
#             anno_path = os.path.join(
#                 args.data_path, "labels_pkl", mode, fps + "train.pickle"
#             )
#             dataset = TripletDataset_CholecT50(
#                 anno_path=anno_path,
#                 data_path=args.data_path,
#                 mode=mode,
#                 data_strategy=args.data_strategy,
#                 output_mode=args.output_mode,
#                 clip_len=args.num_frames,
#                 frame_sample_rate=args.sampling_rate,
#                 keep_aspect_ratio=True,
#                 crop_size=args.input_size,
#                 short_side_size=args.short_side_size,
#                 new_height=256,
#                 new_width=320,
#                 args=args,
#             )

#         nb_classes = 100
#     else:
#         raise NotImplementedError("Dataset [{}] is not implemented".format(args.data_set))

#     assert nb_classes == args.nb_classes
#     print("%s - %s : Number of the class = %d" % (mode, fps, args.nb_classes))
#     print("Data Strategy: %s" % args.data_strategy)
#     print("Output Mode: %s" % args.output_mode)
#     print("Cut Black: %s" % args.cut_black)
#     if args.sampling_rate == 0:
#         print(
#             "%s Frames with Temporal sample Rate %s (%s)"
#             % (str(args.num_frames), str(args.sampling_rate), "Exponential Stride")
#         )
#     elif args.sampling_rate == -1:
#         print(
#             "%s Frames with Temporal sample Rate %s (%s)"
#             % (str(args.num_frames), str(args.sampling_rate), "Random Stride (1-5)")
#         )
#     elif args.sampling_rate == -2:
#         print(
#             "%s Frames with Temporal sample Rate %s (%s)"
#             % (str(args.num_frames), str(args.sampling_rate), "Incremental Stride")
#         )
#     else:
#         print(
#             "%s Frames with Temporal sample Rate %s"
#             % (str(args.num_frames), str(args.sampling_rate))
#         )

#     return dataset, nb_classes

import os
from datasets.transforms import *
from datasets.transforms.surg_transforms import *

from datasets.triplet.CholecT50_triplet import TripletDataset_CholecT50

def build_dataset(is_train, test_mode, fps, args):
    """Load video triplet recognition dataset."""
    if args.data_set == "CholecT50":
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

        dataset = TripletDataset_CholecT50(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            long_side_size=args.long_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 100
    elif args.data_set == "CholecT50C":
        mode = None
        anno_path = None
        if is_train is True:
            mode = "train"
            anno_path = os.path.join(
                args.data_path, "labels_pkl_challenge", mode, fps + "train.pickle"
            )
        elif test_mode is True:
            mode = "test"
            anno_path = os.path.join(
                args.data_path, "labels_pkl_challenge", mode, fps + "test.pickle"
            )
        else:
            mode = "val"
            anno_path = os.path.join(args.data_path, "labels_pkl_challenge", mode, fps + "val.pickle")

        dataset = TripletDataset_CholecT50(
            anno_path=anno_path,
            data_path=args.data_path,
            mode=mode,
            data_strategy=args.data_strategy,
            output_mode=args.output_mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            long_side_size=args.long_side_size,
            new_height=256,
            new_width=320,
            args=args,
        )
        nb_classes = 100
    else:
        raise NotImplementedError("Dataset [{}] is not implemented".format(args.data_set))

    assert nb_classes == args.nb_classes
    print("%s - %s : Number of the class = %d" % (mode, fps, args.nb_classes))
    print("Data Strategy: %s" % args.data_strategy)
    print("Output Mode: %s" % args.output_mode)
    if args.sampling_rate == 0:
        print(
            "%s Frames with Temporal sample Rate %s (%s)"
            % (str(args.num_frames), str(args.sampling_rate), "Exponential Stride")
        )
    elif args.sampling_rate == -1:
        print(
            "%s Frames with Temporal sample Rate %s (%s)"
            % (str(args.num_frames), str(args.sampling_rate), "Random Stride (1-5)")
        )
    elif args.sampling_rate == -2:
        print(
            "%s Frames with Temporal sample Rate %s (%s)"
            % (str(args.num_frames), str(args.sampling_rate), "Incremental Stride")
        )
    else:
        print(
            "%s Frames with Temporal sample Rate %s"
            % (str(args.num_frames), str(args.sampling_rate))
        )

    return dataset, nb_classes

