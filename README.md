conda activate nerfstream
cd /mnt/disk1/ftp/file/60397193/Unimed
python app.py --config_file config.json

cd /mnt/disk1/ftp/file/60397193/Unimed
bash start.sh



# 使用默认配置
python start_action_scanner.py

# 单次扫描
python start_action_scanner.py --once

# 指定扫描目录
python start_action_scanner.py --scan_dir my_action_videos
============
# 只扫描动作视频
python start_scanner.py --mode action

# 同时扫描头像和动作视频
python start_scanner.py --mode both

# 单次扫描动作视频
python start_scanner.py --mode action --once