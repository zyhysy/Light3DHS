import os
import sys
GRANDFA = os.path.dirname(os.path.dirname(os.path.realpath(__file__))) #/home/server1080/Documents/zyh/My_nnUNet/nnunet
sys.path.append(GRANDFA)

from utilities.file_and_folder_operations import *
from configuration import default_num_threads
from utils import split_4d
from utilities.file_endings import remove_trailing_slash


def crawl_and_remove_hidden_from_decathlon(folder):
    folder = remove_trailing_slash(folder)  # 删掉路径结尾的'/'
    assert folder.split('/')[-1].startswith("Task"), "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    subf = subfolders(folder, join=False) #文件夹下的子文件夹
    assert 'imagesTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'imagesTs' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    assert 'labelsTr' in subf, "This does not seem to be a decathlon folder. Please give me a " \
                                                     "folder that starts with TaskXX and has the subfolders imagesTr, " \
                                                     "labelsTr and imagesTs"
    
    # 删除文件夹中无用的文件
    _ = [os.remove(i) for i in subfiles(folder, prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'labelsTr'), prefix=".")]
    _ = [os.remove(i) for i in subfiles(join(folder, 'imagesTs'), prefix=".")]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="The MSD provides data as 4D Niftis with the modality being the first"
                                                 " dimension. We think this may be cumbersome for some users and "
                                                 "therefore expect 3D niftixs instead, with one file per modality. "
                                                 "This utility will convert 4D MSD data into the format nnU-Net "
                                                 "expects")
    parser.add_argument("-i", help="Input folder. Must point to a TaskXX_TASKNAME folder as downloaded from the MSD "
                                   "website", required=True)
    parser.add_argument("-p", required=False, default=default_num_threads, type=int,
                        help="Use this to specify how many processes are used to run the script. "
                             "Default is %d" % default_num_threads)
    parser.add_argument("-output_task_id", required=False, default=None, type=int,
                        help="If specified, this will overwrite the task id in the output folder. If unspecified, the "
                             "task id of the input folder will be used.")
    args = parser.parse_args()

    crawl_and_remove_hidden_from_decathlon(args.i)

    split_4d(args.i, args.p, args.output_task_id)


if __name__ == "__main__":
    main()