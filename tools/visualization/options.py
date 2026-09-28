# Copyright (c) 2026 Robert Bosch GmbH
# SPDX-License-Identifier: AGPL-3.0

# Plotting options
plot_choices = dict(
    draw_pred = True, # True: draw gt and pred; False: only draw gt
    det = True,
    track = False, # True: draw history tracked boxes
    motion = False,
    map = False,
    planning = False,
)
RENDER_BEV_JOINT = True # Render gt and pred in one BEV image
DET_COLOR_TRACKID = False # Color boxes by track id instead of detection class
DET_GT_CAM = True # Draw gt boxes in camera views
DET_GT_GREY = True # Draw gt boxes in grey
DET_RENDER_VEL = False # Render velocity with arrow for each box
DET_RANGE = 50.0 # Range for gt and pred boxes
RENDER_CAM_NAME = False # Render camera name on top of each image
RENDER_BLACK_BEV = True # Black background for BEV
LINE_WIDTH = 3.0 # (2.0 for nuScenes)

START = 0
END = 500
INTERVAL = 1
FPS=8 # 12 for nuScenes, 8 for Argoverse2

SCORE_THRESH = 0.17 # detection threshold (0.3 or 0.17 for nuScenes, 0.13 for Argoverse 2 long-range)
MAP_SCORE_THRESH = 0.2 # 0.3
