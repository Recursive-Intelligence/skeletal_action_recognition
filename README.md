# skeletal_action_recognition

Using demo

```
python3 demo.py --device cuda --video 0 --method stgcn --action_checkpoint_name stgcn_ntu_cv_lw_openpose

```

skeleton data extraction using lightweight openpose
```
python3 skeleton_extraction.py --videos_path path_to_dataset --out_folder path_to_save_skeleton_data --device cuda --num_channels 2  

```