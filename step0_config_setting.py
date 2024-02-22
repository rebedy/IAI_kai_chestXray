# #### SYSTEM #####
# [SYSTEM]
SUBJECT_NAME = 'chestXray'
GPU_NUMBER = 0


# #### CLASSIFICATION #####
# [TASK]
N_CLASSES = 8
RESHAPE = (256, 256)  # (512, 512)

NUMBER_OF_FOLD = 5
NUMBER_OF_PICKLE = 4
AUGMENTATION = False
MEAN_SUBTRACTION = True


# #### INFERENCE #####
# [INFER]
TESTING = True
AUC = True
BY_EPOCH = True
GRAD_CAM = True
""" ************************************* """

# #### DATASET_MODALITY #####
# [DATASET_MODALITY]
WORKSPACE = '/home/ubuntu/'
DATASET_DIR = '/home/ubuntu/DATASET_STORAGE/NIH_chestXray_data/'
LABEL_FILE = '/home/ubuntu/DATASET_STORAGE/NIH_chestXray_data/Data_Entry_2017.csv'

LOG_DIR = WORKSPACE + '/_LOGS'

# TODO | DY : '''Only for Training '''
PICKLE_DIR = '/home/ubuntu/DATASET_STORAGE/NIH_chestXray_data/'

# TODO |  DY :'''Only for Inference '''
# TESTSET_DIR
""" ************************************* """

# [NETWORK]
# #### NETWORK #####
# TODO | DY : For manual Grid Search. Fine-Tuning with loop.
EPOCHS = 30
BATCH_SIZE = 16

KERNEL_INITIALIZER = 'he_normal'  # ['he_normal', 'glorot_uniform']

# #### [Optimizer] 'SGD', 'RMSprop', 'Ada_delta', 'Adam', 'Adam_beta', 'Adamax', 'Nadam'
OPTIMIZER = 'Adam'    # ['Adam','RMSprop', 'Nadam']
LEARNING_RATE = 1e-4  # [1e-3, 1e-4]
LEARNING_DECAY = 0.   # [0., 0.001, 0.0001, 0.01]
INCLUDE_TOP = False
VERBOSE_OPTION = 2

CLASS_WEIGHT = True  # #### TODO | DY : Solved Class Imbalance
CLASS_WEIGHT_DICT = [None, 'balanced', {0: 1, 1: 10}]
""" ************************************* """
