from pathlib import Path
 
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ICONS_DIR = PROJECT_ROOT / "assets" / "icons"
 
ICON_PATHS = {
    0: ICONS_DIR / "person_top_down.png",
    2: ICONS_DIR / "car_facing.png",
    3: ICONS_DIR / "motorcycle_top_down.png",
    5: ICONS_DIR / "bus_facing.png", 
    7: ICONS_DIR / "truck_facing.png",
    9: ICONS_DIR / "traffic_light.png",
    11: ICONS_DIR / "stop_sign.png",  
}
 

ICONS_SIZES = {
    0: (18, 30),
    2: (50, 66),
    3: (24, 28),
    5: (36, 82),
    7: (45, 86),
    9: (30, 50),
    11: (40, 60)
}

DEFAULT_ICON_SIZE = (50, 66)


WARNING_SIGN = ICONS_DIR / "warning_sign.png"
CAUTION_SIGN = ICONS_DIR / "caution_sign.png"
 

EGO_CAR_ICON_PATH = ICONS_DIR / "ego_car.png"
EGO_CAR_ICON_SIZE = (180, 180)