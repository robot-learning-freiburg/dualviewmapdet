export PYTHONPATH="$(dirname $0)/..":$PYTHONPATH

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes_ssd \
    --canbus ./data/nuscenes_ssd \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes_ssd \
    --version v1.0-mini

python tools/data_converter/nuscenes_converter.py nuscenes \
    --root-path ./data/nuscenes_ssd \
    --canbus ./data/nuscenes_ssd \
    --out-dir ./data/infos/ \
    --extra-tag nuscenes_ssd \
    --version v1.0

