from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"
 
ICON_PATHS = {
    0: ICONS_DIR / "person_top_down.png",
    2: ICONS_DIR / "car_facing.png",
    3: ICONS_DIR / "motorcycle_top_down.png",
    5: ICONS_DIR / "bus_facing.png", 
    7: ICONS_DIR / "truck_facing.png",  
}
 

ICONS_SIZES = {
    0: (18, 30),
    2: (28, 44),
    3: (24, 28),
    5: (26, 64),
    7: (34, 60),
}

DEFAULT_ICON_SIZE = (28, 44)
 

EGO_CAR_ICON_PATH = ICONS_DIR / "car_backward.png"
EGO_CAR_ICON_SIZE = (180, 180)