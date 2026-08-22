import pygame
from screeninfo import get_monitors

from visualization.icon_map import ICON_PATHS, ICON_SIZE, EGO_CAR_ICON_PATH, EGO_CAR_ICON_SIZE, DEFAULT_ICON_SIZE



def tint_surface(surface, color_rgb):
    tinted = surface.copy()

    tint = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
    tint.fill((*color_rgb, 255))
    tinted.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
    return tinted



class HUD:
    def __init__(self, bev_transform):
        pygame.init()
        try:
            resolution = get_monitors()[0]
            self.height = resolution.height
            self.width = resolution.width
        except Exception:
            print("Warning: could not detect a monitor, falling back to 800x480 "
                  "(that's also roughly your target Pi touchscreen size).")
            self.width, self.height = 800, 480
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Driver HUD")
        self.clock = pygame.time.Clock()
        self.bev = bev_transform

        self.colors = {
            "background": (25, 25, 30),
            "grid": (50, 50, 60),
            "your car": (30, 30, 90),
            "detected car": (60, 170, 190),
            "warning": (255, 50, 50),
        }

        self.font = pygame.font.SysFont("segoeui", 24)
        self.pixels_per_meter = 20
        self._scaled_icon_cache = {}

        self.icons = {}
        for class_id, path in ICON_PATHS.items():
            try:
                surf = pygame.image.load(path).convert_alpha()
                base_size = ICON_SIZE.get(class_id, DEFAULT_ICON_SIZE)
                self.icons[class_id] = pygame.transform.scale(surf, ICON_SIZE)
            except Exception as e:
                print(f"Warning: could not load icon for class {class_id} ({path}): {e}")


        self.ego_icon = None
        try:
            surf = pygame.image.load(str(EGO_CAR_ICON_PATH)).convert_alpha()
            self.ego_icon = pygame.transform.scale(surf, EGO_CAR_ICON_SIZE)
        except Exception as e:
            print(f"Cound not load the proper image on ego car on {e}")

            car_icon = self.icons.get(2)

            if car_icon is not None:
                tinted = tint_surface(car_icon, (90, 90, 230))

                self.ego_icon = pygame.transform.scale(tinted, EGO_CAR_ICON_SIZE)

                print(" Using a tinted copy of the ego car")


            
    def _get_scaled_icon(self, class_id, scale):
        bucket = round(scale, 1)
        key = (class_id, bucket)
        if key not in self._scaled_icon_cache:
            base = self.icons.get(class_id)
            if base is None:
                return None
            w, h = base.get_size()
            self._scaled_icon_cache[key] = pygame.transform.scale(base, (int(w * bucket), int(h * bucket)))
        return self._scaled_icon_cache[key]

    def draw_3d_grid(self):
        center_x = self.width * 0.5
        horizon_y = int(self.height * 0.3)

        for i in range(-500, 500, 100):
            pygame.draw.line(self.screen, self.colors["grid"], (center_x, horizon_y), (center_x + i, self.height), 2)

        for y in range(horizon_y + 50, self.height, 40):
            pygame.draw.line(self.screen, self.colors["grid"], (0, y), (self.width, y), 1)

    def world_to_screen(self, x_lateral, z_forward):
        origin_x = self.width // 2
        origin_y = self.height - 100  
        screen_x = origin_x + int(x_lateral * self.pixels_per_meter)
        screen_y = origin_y - int(z_forward * self.pixels_per_meter)  
        return screen_x, screen_y

    def render(self, detections, speed_kmh=0):
        self.screen.fill(self.colors["background"])
        self.draw_3d_grid()

    
        ego_x, ego_y = self.width // 2, self.height - 60
        if self.ego_icon is not None:
            rect = self.ego_icon.get_rect(center=(ego_x, ego_y))
            self.screen.blit(self.ego_icon, rect)
        else:
            pygame.draw.rect(
                self.screen, self.colors["your car"],
                (self.width // 2 - 20, self.height - 100, 40, 80),
                border_radius=10,
            )

        projected = []
        for det in detections:
            x_lat, z_fwd = self.bev.to_bev(det["bbox"])
            if z_fwd <= 0:
                continue
            projected.append((z_fwd, x_lat, det))
        projected.sort(key=lambda p: p[0], reverse=True)  

        for z_fwd, x_lat, det in projected:
            sx, sy = self.world_to_screen(x_lat, z_fwd)
            if not (0 <= sx <= self.width and 0 <= sy <= self.height):
                continue

            scale = max(0.5, min(1.5, 15 / max(z_fwd, 5)))

            icon = self._get_scaled_icon(det["class_id"], scale)
            if icon is not None:
                rect = icon.get_rect(center=(sx, sy))
                self.screen.blit(icon, rect)
            else:
                color = self.colors["warning"] if z_fwd < 15 else self.colors["detected car"]
                pygame.draw.circle(self.screen, color, (sx, sy), int(8 * scale))

        speed_surface = self.font.render(f"Speed: {speed_kmh} km/h", True, (255, 255, 255))
        self.screen.blit(speed_surface, (20, 20))

        pygame.display.flip()
        self.clock.tick(30)

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def quit(self):
        pygame.quit()