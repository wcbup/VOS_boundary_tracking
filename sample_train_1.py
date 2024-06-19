from Loader_17 import DAVIS_Rawset
from BoundarySampler import sample_save_dataset

train_rawset = DAVIS_Rawset(is_train=True)
# sample the second half of the training dataset
sample_save_dataset(
    train_rawset.data_set[len(train_rawset.data_set) // 2 :],
    len(train_rawset.data_set) // 2,
    "train",
    256,
    True,
    5,
    100,
)
