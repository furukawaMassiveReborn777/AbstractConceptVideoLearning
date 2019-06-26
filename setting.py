FEATURE_SAVEPATH = "none"# "none" when not extracting feature, otherwise no training and save feature
LOAD_CHECKPOINT_PATH = "none"# "none" when loading nothing
LOAD_OPTIM = False

MODEL_SAVE_DIR = "./data/models/ckpt"
TRAIN_CSV = "./data/train_csv/hmdb.csv"

NUM_CLASSES = 51 # num_classes of pretrained model
NEW_NUM_CLASSES = 51 # if NUM_CLASSES != NEW_NUM_CLASSES final layer will be replaced using new classes num

NUM_EPOCHS = 2000 # train epoch total
BATCH_SIZE = 10
INPUT_SIZE = 224
FRAME_SAMPLE_NUM = 16

IMG_EXT = ".jpg"

MULTI_GPU = False

if MULTI_GPU:
    BATCH_SIZE *= 4
