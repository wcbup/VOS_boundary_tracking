from Loader_17 import DAVIS_Rawset
from BoundarySampler import sample_save_dataset

val_rawset = DAVIS_Rawset(is_train=False)
sample_save_dataset(
    val_rawset.data_set,
    0,
    "val",
    256,
    True,
    5,
    100,
)
