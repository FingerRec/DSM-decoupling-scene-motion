# Dataset Preparation


## Notice
For Kinetics, we decode on the fly, each row in the txt file include:
> video_path class

And we load the videos directly, please place the training set in SSD for fast IO.

Prepare dataset (UCF101/diving48/sth/hmdb51/actor-hmdb51), and each row of txt is as below:

> video_path class frames_num

These datasets saved in frames. We offer list for all these datasets.


## Download Datasets lists.
```bash
mkdir datasets
wget -c []
mkdir experiments && cd experiments
mkdir logs
```

## Kinetics
As some Youtube Link is lost, we use a copy of kinetics-400 from [K400](), the training set is 234643 videos now and the val set is 19761 now. 
All the videos in mpeg/avi format.

## UCF101/HMDB51
These two video datasets contain three train/test splits. 
Down UCF101 from [UCF101]()  and HMDB51 from [HMDB51]().


## Sth-v1
Down Sth-v1 from [UCF101]() 

## Frame Extract
refer utils/data_process/gen_hmdb51_dir.py and utils/data_process/gen_hmdb51_frames.py for details.

### Semi-supervised Subdataset
We also provide manuscript to get the sub-set of kinetics-400. Please refer to Contrastive/utils/data_provess/semi_data_split.py for details.
