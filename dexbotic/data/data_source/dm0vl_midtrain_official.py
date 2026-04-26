# DM0 mid-train VL 50GB dataset register（upstream 风格）
# 通过 DM0_VL_ROOT 环境变量定位数据根目录
import os

from dexbotic.data.data_source.register import register_dataset


_ROOT = os.environ.get("DM0_VL_ROOT")
if _ROOT:
    _IMAGES = os.path.join(_ROOT, "images")
    _ANNOT = os.path.join(_ROOT, "annotations")
    DM0_VL_MIDTRAIN_50GB = {
        "cambrian737k": {
            "data_path_prefix": _IMAGES,
            "annotations": os.path.join(_ANNOT, "cambrian737k"),
            "frequency": 1,
        },
        "cambrian10m_filtered": {
            "data_path_prefix": _IMAGES,
            "annotations": os.path.join(_ANNOT, "cambrian10m_filtered"),
            "frequency": 1,
        },
        "llava_onevision": {
            "data_path_prefix": _IMAGES,
            "annotations": os.path.join(_ANNOT, "llava_onevision"),
            "frequency": 1,
        },
        "self_collected_proxy": {
            "data_path_prefix": _IMAGES,
            "annotations": os.path.join(_ANNOT, "self_collected_proxy"),
            "frequency": 1,
        },
    }
    register_dataset(DM0_VL_MIDTRAIN_50GB, meta_data={}, prefix="dm0vlmid")
