import warnings
import os


class DefaultConfig:

    TASK_NAME = 'CON'

    # GPU / CPU
    GPU_IDS = None  # slipt different gpus with comma
    nTHREADS = 8
    WORKERS = 8

    # MODEL
    MODEL = 'trecg'
    ARCH = 'vgg11_bn'
    PRETRAINED = 'imagenet'
    CONTENT_PRETRAINED = 'imagenet'
    NO_UPSAMPLE = False  # set True when evaluating baseline
    FIX_GRAD = False
    IN_CONC = False  # if True, change input_nc from 3 to specific ones

    # PATH
    # DATA_DIR_TRAIN = '/data0/lzy/sunrgbd/conc_jet_rgb/con_cf_TT//train'
    # DATA_DIR_VAL = '/data0/lzy/sunrgbd/conc_jet_rgb/con_cf_TT/test'
    ROOT_DIR = '/home/lzy/'
    DATA_DIR_TRAIN =  '/data0/lzy/STL10/unlabeled/'
    DATA_DIR_VAL =  '/data0/lzy/STL10/train/'
    DATA_DIR_UNLABELED = ROOT_DIR + '/home/dudapeng/workspace/datasets/nyud2/conc_data/10k_conc_bak'
    SAMPLE_MODEL_PATH = None
    CHECKPOINTS_DIR = './checkpoints'
    LOG_PATH = ROOT_DIR + 'summary'

    # DATA
    DATA_TYPE = 'pair'  # pair | single
    WHICH_DIRECTION = None
    NUM_CLASSES = 10
    BATCH_SIZE = 20
    LOAD_SIZE = 96#256
    FINE_SIZE = 64#224
    FLIP = True
    UNLABELED = False
    FIVE_CROP = False
    FAKE_DATA_RATE = 0.3
    MULTI_SCALE = True
    MULTI_SCALE_NUM = 5

    # OPTIMIZATION
    LR = 2e-4
    WEIGHT_DECAY = 1e-4
    MOMENTUM = 0.9
    LR_DECAY_ITER = 1500
    LR_POLICY = 'plateau'  # lambda|step|plateau

    # TRAINING / TEST
    PHASE = 'train'
    RESUME = False
    RESUME_PATH = None
    RESUME_PATH_A = None
    RESUME_PATH_B = None
    NO_FC = False
    INIT_EPOCH = True  # True for load pretrained parameters, False for resume the last training
    START_EPOCH = 1
    ROUND = 1
    MANUAL_SEED = None
    NITER = 10
    NITER_DECAY = 40
    NITER_TOTAL = 50
    LOSS_TYPES = []  # SEMANTIC_CONTENT, PIX2PIX, GAN
    EVALUATE = True
    USE_FAKE_DATA = False
    CLASS_WEIGHTS_TRAIN = None
    PRINT_FREQ = 100
    NO_VIS = True
    CAL_LOSS = True
    SAVE_BEST = False
    INFERENCE = False

    def parse(self, kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribute {0}".format(k))
            setattr(self, k, v)

        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k, ':', getattr(self, k))
