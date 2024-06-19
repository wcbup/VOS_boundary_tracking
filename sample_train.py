from Loader_17 import DAVIS_Rawset
from BoundarySampler import sample_save_dataset

train_rawset = DAVIS_Rawset(is_train=True)
# sample the training rawset
sample_save_dataset(
    train_rawset.data_set,
    0,
    "train",
    256,
    True,
    100,
)
