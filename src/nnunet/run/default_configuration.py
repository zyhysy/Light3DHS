from paths import network_training_output_dir, preprocessing_output_dir, default_plans_identifier
from utilities.file_and_folder_operations import *
from experiment_planning.summarize_plans import summarize_plans
from training.model_restore import recursive_find_python_class


def get_configuration_from_output_folder(folder):
    # split off network_training_output_dir
    folder = folder[len(network_training_output_dir):]
    if folder.startswith("/"):
        folder = folder[1:]

    configuration, task, trainer_and_plans_identifier = folder.split("/")
    trainer, plans_identifier = trainer_and_plans_identifier.split("__")
    return configuration, task, trainer, plans_identifier


def get_default_configuration(network, task, network_trainer, plans_identifier=default_plans_identifier,
                              search_in=("/home/amax/repository/zyh/My_nnUnet/nnunet", "training", "network_training"),
                              base_module='training.network_training'):
    assert network in ['3d_lowres', '3d_fullres', '3d_cascade_fullres'], \
        "network can only be one of the following: \'3d_lowres\', \'3d_fullres\', \'3d_cascade_fullres\'"

    dataset_directory = join(preprocessing_output_dir, task)


    plans_file = join(preprocessing_output_dir, task, plans_identifier + "_plans_3D.pkl")

    plans = load_pickle(plans_file)
    possible_stages = list(plans['plans_per_stage'].keys())

    if (network == '3d_cascade_fullres' or network == "3d_lowres") and len(possible_stages) == 1:
        raise RuntimeError("3d_lowres/3d_cascade_fullres only applies if there is more than one stage. This task does "
                           "not require the cascade. Run 3d_fullres instead")

    
    stage = possible_stages[-1]  # 0

    trainer_class = recursive_find_python_class([join(*search_in)], network_trainer,
                                                current_module=base_module)  # 从nnunet/training/network_training报下找到nnUNetTrainerV2类

    output_folder_name = join(network_training_output_dir, network, task, network_trainer + "__" + plans_identifier)

    print("###############################################")
    print("I am running the following nnUNet: %s" % network)
    print("My trainer class is: ", trainer_class)
    print("For that I will be using the following configuration:")
    summarize_plans(plans_file)
    print("I am using stage %d from these plans" % stage)


    batch_dice = False
    print("I am using sample dice + CE loss")

    print("\nI am using data from this folder: ", join(dataset_directory, plans['data_identifier']))
    print("###############################################")
    print(plans_file)
    print(output_folder_name)
    print(dataset_directory)
    print(plans_file)
    print(batch_dice)
    print(stage)
    return plans_file, output_folder_name, dataset_directory, batch_dice, stage, trainer_class
