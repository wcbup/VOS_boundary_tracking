from Loader_17 import DAVIS_Rawset
from BoundarySampler import sample_save_dataset

train_rawset = DAVIS_Rawset(is_train=True)
# sample the first quarter of the training dataset

sample_save_dataset(
    video_dataset=train_rawset.data_set[: len(train_rawset.data_set) // 4],
    start_idx=0,
    save_title="train",
    max_point_num=256,
    use_std_loss=True,
    fir_epoch_multi=5,
    epoch_num=100,
    min_threshold=25,
)

# sample_save_dataset(
#     video_dataset=train_rawset.data_set[: len(train_rawset.data_set) // 2],
#     start_idx=0,
#     save_title="train",
#     max_point_num=256,
#     use_std_loss=True,
#     fir_epoch_multi=5,
#     epoch_num=100,
#     min_threshold=25,
# )

# sample_save_dataset(
#     train_rawset.data_set[: len(train_rawset.data_set) // 2],
#     0,
#     "train",
#     256,
#     True,
#     5,
#     100,
# )
