import os
import cv2
import random
import numpy as np
import torch
import argparse
from shutil import copyfile
from src.config import Config
from src.nclg import NCLG
import optuna


def main(mode=None):
    r"""starts the model

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    config = load_config(mode)


    # cuda visble devices
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in config.GPU)


    # init device
    if torch.cuda.is_available():
        print('Cuda is available')
        config.DEVICE = torch.device("cuda")
        torch.backends.cudnn.benchmark = True   # cudnn auto-tuner
    else:
        print('Cuda is unavailable, use cpu')
        config.DEVICE = torch.device("cpu")

    #config.DEVICE = torch.device("cpu")

    # set cv2 running threads to 1 (prevents deadlocks with pytorch dataloader)
    cv2.setNumThreads(0)


    # initialize random seed
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed_all(config.SEED)
    np.random.seed(config.SEED)
    random.seed(config.SEED)

    # build the model and initialize
    model = NCLG(config)
    model.load()


    # model training
    if config.MODE == 1:
        config.print()
        print('\nstart training...\n')
        model.train()

    # model test
    elif config.MODE == 2:
        print('\nstart testing...\n')
        model.test()

    #### optuna
    elif config.MODE == 5:
        print('\noptuna starting...\n')
        study = optuna.create_study(direction='maximize')
        study.optimize(model.objective_variable(),n_trials=20)
        print('Best params : {}'.format(study.best_params))
        print('Best value  : {}'.format(study.best_value))
        print('Best trial  : {}'.format(study.best_trial))
        df = study.trials_dataframe()
        print(df)

    #### optuna_end


    # eval mode
    else:
        print('\nstart eval...\n')
        model.eval()



def load_config(mode=None):
    r"""loads model config

    Args:
        mode (int): 1: train, 2: test, 3: eval, reads from config file if not specified
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='/mnt/recsys/daniel/gan_inpainting/SIMPAC-2022-272/experiments/',
                        help='model checkpoints path (default: ./checkpoints)')

    parser.add_argument('--model', type=int, default='2', choices=[1, 2, 3],
                        help='1: landmark prediction model, 2: inpaint model, 3: joint model')

    parser.add_argument('--mask_type', type=int, default='2', choices=[1, 2],
                        help='1: Fixed_mask, 2: Random Mask')
    # test mode
    if mode == 2 or mode == 1:
        parser.add_argument('--experiment_name', type=str, default='celebAHQ-pretrained/', help='experiment name')

        #parser.add_argument('--input', type=str, default = '/mnt/recsys/daniel/datasets/US_surgeons_cleft/nosemask256/img', help='path to the input images directory or an input image')
        #parser.add_argument('--mask', type=str,  default= '/mnt/recsys/daniel/datasets/US_surgeons_cleft/nosemask256/black_mask', help='path to the masks directory or a mask file')
        #parser.add_argument('--landmark', type=str, default = '/mnt/recsys/daniel/datasets/US_surgeons_cleft/nosemask256/', help='path to the landmarks directory or a landmark file')

        parser.add_argument('--input', type=str,
                            default='/mnt/recsys/daniel/evaluation_miccai/present_imgs2/img',
                            help='path to the input images directory or an input image')
        parser.add_argument('--mask', type=str,
                            default='/mnt/recsys/daniel/evaluation_miccai/present_imgs2/mask',
                            help='path to the masks directory or a mask file')
        parser.add_argument('--landmark', type=str,
                            default='/mnt/recsys/daniel/evaluation_miccai/present_imgs2/keypoints',
                            help='path to the landmarks directory or a landmark file')

    if mode == 2:
        #parser.add_argument('--landmark', type=str,
        #                    default='',
        #                    help='path to the landmarks directory or a landmark file')
        parser.add_argument('--iteration', type=str, default = '100')
        parser.add_argument('--test_filelist', type=str, default='')
        parser.add_argument('--output_dir', type=str, default = '/mnt/recsys/daniel/evaluation_miccai/results_imgs_web2/UK_model_pretrained')
        #parser.add_argument('--output_dir', type=str,
        #                   default='/mnt/recsys/daniel/evaluation_miccai/results_images/UK_model/celebA_pretrained/lips_mask')
    args = parser.parse_args()
    experiment_path = os.path.join(args.path, args.experiment_name)

    config_path = os.path.join(experiment_path, 'config.yml')

    # create checkpoints path if does't exist
    if not os.path.exists(experiment_path):
        os.makedirs(experiment_path)

    # copy config template if does't exist
    if not os.path.exists(config_path):
        copyfile('./config.yml.example', config_path)

    # load config file
    config = Config(config_path)

    # train mode
    if mode == 1:
        config.MODE = 1
        if args.model:
            config.MODEL = args.model

        if args.input is not None:
            config.TRAIN_INPAINT_IMAGE_FLIST = args.input

        if ((args.mask is not None) and (args.mask_type==1)):
            config.TRAIN_MASK_FLIST = args.mask

        if args.landmark is not None:
            config.TRAIN_INPAINT_LANDMARK_FLIST = args.landmark


        config.RESULTS = experiment_path

        if args.experiment_name is not None:
            config.EXPERIMENT = args.experiment_name

        if args.experiment_name is not None:
            config.EXPERIMENT_PATH = experiment_path

        if args.mask_type == 1:
            config.MASK = 6
        elif args.mask_type == 2:
            config.MASK = 8
            config.TRAIN_MASK_FLIST = None
    # test mode
    elif mode == 2:
        config.MODE = 2

        #config.TEST_INPAINT_IMAGE_FLIST =

        config.MODEL = args.model if args.model is not None else 3
        config.PRETRAINED_MODEL = 'InpaintingModel' + args.iteration
        if args.input is not None:
            config.TEST_INPAINT_IMAGE_FLIST = args.input

        config.ITERATION = args.iteration
        if args.mask is not None:
            config.TEST_MASK_FLIST = args.mask

        if args.landmark is not None:
            config.TEST_INPAINT_LANDMARK_FLIST = args.landmark

        #if args.output is not None:
        #    config.RESULTS = args.output
        #results_dir = os.path.join(args.output_dir, args.iteration)
        #if not os.path.exists(results_dir):
        #    os.makedirs(results_dir)


        config.RESULTS = args.output_dir

        if args.experiment_name is not None:
            config.EXPERIMENT = args.experiment_name

    # eval mode
    elif mode == 3:
        config.MODE = 3
        config.MODEL = args.model if args.model is not None else 3

    return config


if __name__ == "__main__":
    main()
