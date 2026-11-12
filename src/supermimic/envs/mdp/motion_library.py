"""Compatibility exports for the motion library now owned by the data layer."""

from supermimic.data.discovery import discover_npz as discover_npz
from supermimic.data.motion_library import GOAL_FEAT_DIM as GOAL_FEAT_DIM
from supermimic.data.motion_library import REF_DIM as REF_DIM
from supermimic.data.motion_library import ClipInfo as ClipInfo
from supermimic.data.motion_library import MotionLibrary as MotionLibrary
from supermimic.data.motion_library import RefBodies as RefBodies
from supermimic.data.motion_library import RefState as RefState
from supermimic.data.motion_library import goal_features as goal_features
