from Loader_17 import DAVIS_Rawset
from BoundarySampler import sample_save_dataset

val_rawset = DAVIS_Rawset(is_train=False)
# sample_save_dataset(
#     val_rawset.data_set,
#     0,
#     "val",
#     256,
#     True,
#     5,
#     100,
# )
sample_save_dataset(
    video_dataset=val_rawset.data_set,
    start_idx=0,
    save_title="val",
    max_point_num=256,
    use_std_loss=True,
    fir_epoch_multi=5,
    epoch_num=100,
    min_threshold=25,
)
