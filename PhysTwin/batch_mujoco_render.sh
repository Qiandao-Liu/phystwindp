#!/bin/bash

SCENE_PATH=~/workspace/mujoco_gs/aloha_custom/scene_with_cloth.xml
RENDER_SCRIPT=~/workspace/mujoco_gs/aloha_custom/traj_in_mujoco_offscreen.py
PKL_DIR=~/workspace/PhysTwin/mpc_replay
FPS=20
SCALE=0.6

mkdir -p replay_videos

for i in $(seq -f "%04g" 0 99); do
    PKL_FILE=$PKL_DIR/concat_commands${i}.pkl
    if [ -f "$PKL_FILE" ]; then
        echo "🎬 Rendering $PKL_FILE"
        python $RENDER_SCRIPT \
            --scene $SCENE_PATH \
            --pkl $PKL_FILE \
            --fps $FPS \
            --cloth \
            --scale $SCALE
    else
        echo "❌ File not found: $PKL_FILE"
    fi
done
