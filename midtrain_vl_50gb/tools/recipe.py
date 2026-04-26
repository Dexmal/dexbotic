# Single source of truth for collection targets and subset lists.
# Layout is rooted at ${DM0_VL_ROOT}; export it before running any tool.
import os
from pathlib import Path

ROOT_DIR = Path(os.environ.get("DM0_VL_ROOT", "data/dm0_vl_midtrain_50gb"))
ANNOTATIONS_DIR = ROOT_DIR / "annotations"
IMAGES_DIR = ROOT_DIR / "images"
RAW_DIR = ROOT_DIR / "raw"
PROGRESS_DIR = ROOT_DIR / ".progress"
DATA_SOURCE_DIR = ROOT_DIR / "data_source"

REGISTER_PREFIX = "dm0vlmid"

# Paper §3.2 VL split. cambrian10m is source-limited by COCO pool dedup;
# llava_onevision absorbs the deficit through a wider subset list.
RECIPE = {
    "cambrian10m_filtered": {
        "target_image_bytes": 20 * 1024**3,
        "estimated_avg_kb": 80,
    },
    "cambrian737k": {
        "target_image_bytes": 12 * 1024**3,
        "estimated_avg_kb": 80,
    },
    "llava_onevision": {
        "target_image_bytes": 22 * 1024**3,
        "estimated_avg_kb": 100,
        "subsets": [
            "chartqa(cauldron,llava_format)",
            "ai2d(cauldron,llava_format)",
            "sharegpt4v(coco)",
            "scienceqa(cauldron,llava_format)",
            "dvqa(cauldron,llava_format)",
            "sharegpt4v(llava)",
            "sharegpt4v(sam)",
            "aokvqa(cauldron,llava_format)",
            "vsr(cauldron,llava_format)",
            "tallyqa(cauldron,llava_format)",
            "iconqa(cauldron,llava_format)",
            "chart2text(cauldron)",
            "llavar_gpt4_20k",
            "rendered_text(cauldron)",
            "infographic_vqa_llava_format",
            "image_textualization(filtered)",
            "sharegpt4v(knowledge)",
            "visualmrc(cauldron)",
            "hateful_memes(cauldron,llava_format)",
            "intergps(cauldron,llava_format)",
            "robut_wtq(cauldron,llava_format)",
            "vistext(cauldron)",
            "visual7w(cauldron,llava_format)",
            "tqa(cauldron,llava_format)",
            "allava_instruct_vflan4v",
            "robut_wikisql(cauldron)",
            "mapqa(cauldron,llava_format)",
            "textcaps",
            "ureader_cap",
            "vision_flan(filtered)",
        ],
    },
    "self_collected_proxy": {
        "target_image_bytes": 8 * 1024**3,
        "estimated_avg_kb": 100,
        "subsets": ["uground", "embspatial", "synthdog_en", "cord_v2"],
    },
}

TOTAL_TARGET_BYTES = 50 * 1024**3
DISK_FLOOR_GB = 100
EPISODE_BATCH_SIZE = 10000
DISK_CHECK_EVERY = 200

COLLECTOR_ORDER = [
    "cambrian737k",
    "cambrian10m_filtered",
    "llava_onevision",
    "self_collected_proxy",
]
