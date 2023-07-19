import os

root_data_dir = '../../'
dataset = 'Dataset/Hm-large'
behaviors = 'hm_50w_users.tsv'
images = 'hm_50w_items.tsv'
lmdb_data = 'hm_50w_items.lmdb'
logging_num = 4
testing_num = 1

CV_resize = 224
CV_model_load = 'None'
freeze_paras_before = 0


mode = 'train'
item_tower = 'id'

epoch = 50
load_ckpt_name = 'epoch-4.pt'

l2_weight = 0.1
drop_rate = 0.1
batch_size = 256
lr = 5e-5
embedding_dim = 2048

fine_tune_lr = 0
label_screen = '{}_bs{}_ed{}_lr{}_dp{}_wd{}_Flr{}'.format(
    item_tower, batch_size, embedding_dim, lr,
    drop_rate, l2_weight, fine_tune_lr)
run_py = "CUDA_VISIBLE_DEVICES='0' \
         /opt/anaconda3/bin/python  -m torch.distributed.launch --nproc_per_node 1 --master_port 1234\
         run_test.py --root_data_dir {}  --dataset {} --behaviors {} --images {}  --lmdb_data {}\
         --mode {} --item_tower {} --load_ckpt_name {} --label_screen {} --logging_num {} --testing_num {}\
         --l2_weight {} --drop_rate {} --batch_size {} --lr {} --embedding_dim {}\
         --CV_resize {} --CV_model_load {}  --epoch {} --freeze_paras_before {}  --fine_tune_lr {}".format(
    root_data_dir, dataset, behaviors, images, lmdb_data,
    mode, item_tower, load_ckpt_name, label_screen, logging_num, testing_num,
    l2_weight, drop_rate, batch_size, lr, embedding_dim,
    CV_resize, CV_model_load, epoch, freeze_paras_before, fine_tune_lr)
os.system(run_py)
