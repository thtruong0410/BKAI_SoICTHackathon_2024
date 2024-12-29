export PYTHONPATH=${PYTHONPATH}:/base
python3 tools/train.py blackbox-cfg/co_dino_5scale_swin_large_16e_o365tococo.py --auto-resume --work-dir blackbox
