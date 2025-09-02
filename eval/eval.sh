test_img_path=path_to_test_img_folder/  # test set path, need to change
syn_img_path=path_to_synthesized_img_folder/    # results path, need to change (mask sure containing same images with test_img_path)
yolo_img_path=path_to_ultralytics_img_folder/   # ultralytics data path, need to change

# calculate clip score
python eval_clip_score.py --folder $syn_img_path --ann $syn_img_path/metadata.jsonl

# calculate fidelity
CUDA_VISIBLE_DEVICES=0 fidelity --input1 $test_img_path --input2 $syn_img_path -b 16 -g 0 -f

# calculate yoloscore
rm -rf $yolo_img_path/val
mkdir -p $yolo_img_path/val
cp $syn_img_path/* $yolo_img_path/val
python ultralytics/val.py