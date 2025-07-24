# 兼容导入：支持两个版本的模型
# 384分辨率版本在 wav2lip.py 中
# 256分辨率版本在 wav2lip_v2.py 中
# 默认导入256版本以保持向后兼容
from .wav2lip_v2 import Wav2Lip, Wav2Lip_disc_qual
from .syncnet import SyncNet_color

# 两个版本都可以通过完整路径导入：
# from wav2lip.models.wav2lip import Wav2Lip as Wav2Lip384
# from wav2lip.models.wav2lip_v2 import Wav2Lip as Wav2Lip256