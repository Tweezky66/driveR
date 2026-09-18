import pygame
from screeninfo import get_monitors

from visualization.icon_map import ICON_PATHS, ICONS_SIZES, EGO_CAR_ICON_PATH, EGO_CAR_ICON_SIZE, DEFAULT_ICON_SIZE, WARNING_SIGN, CAUTION_SIGN



def tint_surface(surface, color_rgb):
    tinted = surface.copy()

    tint = pygame.Surface(tinted.get_size(), pygame.SRCALPHA)
    tint.fill((*color_rgb, 255))
    tinted.blit(tint, (0,0), special_flags=pygame.BLEND_RGBA_MULT)
    return tinted



class HUD:
    def __init__(self, bev_transform, mode="standalone", panel_width_ratio=0.32, panel_side="right"):
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

        self.mode = mode
        self.panel_side = panel_side

        if self.mode == "panel":
            self.panel_y0 = 0
            self.panel_w = max(240, int(self.width * panel_width_ratio)) # scaling HUD width to be side pannel
            self.panel_x0 = (self.width - self.panel_w) if panel_side == "right" else 0
            self.panel_h = self.height
            self._geo_scale = self.panel_w / 1920
            horizon_y = self.panel_y0 + int(self.panel_h * 0.3)
            origin_y = self.panel_y0 + self.panel_h - int(100 * self._geo_scale)
            usable_vertical_px = max(1, origin_y - horizon_y)
            PANEL_MAX_RANGE_M = 50
            self.pixels_per_meter = max(4, usable_vertical_px / PANEL_MAX_RANGE_M)
        else:
            self.panel_x0 = 0
            self.panel_y0 = 0
            self.panel_w = self.width
            self.panel_h = self.height
            self._geo_scale = 1.0
            self.pixels_per_meter = 20

        self.colors = {
            "background": (25, 25, 30),
            "grid": (50, 50, 60),
            "your car": (30, 30, 90),
            "detected car": (60, 170, 190),
            "caution": (230, 180, 60),
            "warning": (255, 50, 50),
            "panel_bg": (18, 18, 22, 235), # 235 for alpha chanel to be semi-transperent
            "panel_border": (90, 90, 100),
            "accent": (80, 210, 220),
        }

        self.font = pygame.font.SysFont("segoeui", 24)
        self.font_small = pygame.font.SysFont("segoeui", 15)
        self.font_speed = pygame.font.SysFont("segoeui", 42, bold=True)
        self._scaled_icon_cache = {}
        self._scaled_risk_cache = {}
        self._tinted_icon_cache = {}
        self._bg_gradient = self._build_gradient_surface(self.panel_w, self.panel_h)

        self.icons = {}
        for class_id, path in ICON_PATHS.items():
            try:
                surf = pygame.image.load(path).convert_alpha()
                base_size = ICONS_SIZES.get(class_id, DEFAULT_ICON_SIZE)
                self.icons[class_id] = pygame.transform.scale(surf, base_size)
            except Exception as e:
                print(f"Warning: could not load icon for class {class_id} ({path}): {e}")


        self.risk_icons = {}

        self._load_risk_icon(risk_level=1, path=CAUTION_SIGN)

        self._load_risk_icon(risk_level=2, path=WARNING_SIGN)


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


    def _load_risk_icon(self, risk_level, path):
        try:
            surface = pygame.image.load(path).convert_alpha()
            base_size = (50, 50)
            if risk_level == 2:
                surface = tint_surface(surface, self.colors["warning"])

            self.risk_icons[risk_level] = pygame.transform.smoothscale(surface, base_size)
        except Exception as e:
            print(f"Could not load {risk_level}: {e}")

    def _get_scaled_risk_icon(self, risk_level, scale):


        bucket = round(scale, 1)
        key = (risk_level, bucket)

        if key not in self._scaled_risk_cache:

            base = self.risk_icons.get(risk_level)

            if base is None:
                return None

            width, height = base.get_size()

            MIN_BADGE_PX = 22

            new_size = (
                max(MIN_BADGE_PX, int(width * bucket)),
                max(MIN_BADGE_PX, int(height * bucket)),
            )

            self._scaled_risk_cache[key] = (
                pygame.transform.smoothscale(
                    base,
                    new_size,
                )
            )

        return self._scaled_risk_cache[key]

            
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

    def _dectlutter_rect(self, rect, placed_rect, max_shift=40, step=6):
        if not any(rect.colliderect(r) for r in placed_rect):
            return rect # basicly no collision detected

        for attempt in range(1, max_shift // step + 1):
            for direction in (1, -1):
                candidate = rect.move(direction * attempt * step, 0)
                if not any(candidate.colliderect(r) for r in placed_rect):
                    return candidate # another fallback but for calibrated rectangle
        return rect 


    def _get_tinted_class_icon(self, class_id, risk_level):
        key = (class_id, risk_level)
        if key not in self._tinted_icon_cache:
            base = self.icons.get(class_id)
            if base is None:
                return None
            if risk_level == 1:
                tinted = tint_surface(base, self.colors["caution"])
            elif risk_level == 2:
                tinted = tint_surface(base, self.colors["warning"])
            else:
                tinted = base
            self._tinted_icon_cache[key] = tinted
        return self._tinted_icon_cache[key] 


    
    def _get_risk_icon(self, class_id, scale, risk_level):
        if risk_level not in (0, 1, 2):
            raise ValueError(f"Uncorrect risk level: {risk_level}")

        if risk_level == 0:
            return self._get_scaled_icon(class_id, scale)

        tinted = self._get_tinted_class_icon(class_id, risk_level)

        if tinted is None:
            return self._get_scaled_risk_icon(risk_level, scale)

        bucket = round(scale, 1)
        cache_key = (class_id, risk_level, bucket)
        if cache_key not in self._scaled_icon_cache:
            w, h = tinted.get_size()
            self._scaled_icon_cache[cache_key] = pygame.transform.scale(
                tinted, 
                (
                    max(1, int(w * bucket)),
                    max(1, int(h * bucket))
                )
            )
        return self._scaled_icon_cache[cache_key]



    def _build_gradient_surface(self, w, h):
        surf = pygame.Surface((w, max(1, h)))
        top_color = (14, 16, 26)
        bottom_color = (35, 38, 48)
        for i in range(h):
            t = i / max(1, h) # gradient cooficient
            r = int(top_color[0] + (bottom_color[0] - top_color[0]) * t)
            g = int(top_color[1] + (bottom_color[1] - top_color[1]) * t)
            b = int(top_color[2] + (bottom_color[2] - top_color[2]) * t)
            pygame.draw.line(surf, (r, g, b), (0, i), (w, i))
        return surf


    def draw_3d_grid(self):
        center_x = self.panel_x0 + self.panel_w * 0.5
        horizon_y = self.panel_y0 + int(self.panel_h * 0.3) # use panel variable for starting pt if panel gets as mode
        bottom_y = self.panel_y0 + self.panel_h


        if self.mode == "panel":
            top_road_w = max(6, int(self.panel_w * 0.035))
            bot_road_w = max(60, int(self.panel_w * 0.62))

            top_shoulder_w = max(10, int(self.panel_w * 0.05))
            bot_shoulder_w = max(80, int(self.panel_w) * 0.96)
        else:
            top_road_w = max(12, int(40 * self._geo_scale))
            bot_road_w = max(80, int(750 * self._geo_scale))
    
            top_shoulder_w = max(18, int(60 * self._geo_scale))
            bot_shoulder_w = max(100, int(900 * self._geo_scale))
    
        outer_poly = [
            (center_x - top_shoulder_w // 2, horizon_y),
            (center_x + top_shoulder_w // 2, horizon_y),
            (center_x + bot_shoulder_w // 2, bottom_y),
            (center_x - bot_shoulder_w // 2, bottom_y)
        ]
        pygame.draw.polygon(self.screen, (60, 60, 65), outer_poly)
    
        road_poly = [
            (center_x - top_road_w // 2, horizon_y),
            (center_x + top_road_w // 2, horizon_y),
            (center_x + bot_road_w // 2, bottom_y),
            (center_x - bot_road_w // 2, bottom_y)
        ]
        pygame.draw.polygon(self.screen, (25, 25, 30), road_poly)



        # Draw a road centered line
        dash_color = (150, 150, 90)
        n_dashes = 10
        for i in range(n_dashes):
            t0 = i / n_dashes
            t1 = (i + 0.5) / n_dashes
            if t0 < 0.03:
                continue
            y0 = horizon_y + (bottom_y - horizon_y) * t0
            y1 = horizon_y + (bottom_y - horizon_y) * t1
            width_at_t0 = top_road_w + (bot_road_w - top_road_w) * t0
            width_at_t1 = top_road_w + (bot_road_w - top_road_w) * t1
            line_w = max(1, int(2 * self._geo_scale * (0.4 + t0)))
            pygame.draw.line(
                self.screen,
                dash_color,
                (center_x, y0),
                (center_x, y1),
                width=line_w
            )


    def _draw_shadow(self, center_x, bottom_y, width):
        shadow_w = max(4, int(width * 0.8))
        shadow_h = max(2, int(shadow_w * 0.28))
        shadow_surf = pygame.Surface((shadow_w, shadow_h), pygame.SRCALPHA)
        pygame.draw.ellipse(shadow_surf, (0, 0, 0, 90), (0, 0, shadow_w, shadow_h))
        rect = shadow_surf.get_rect(center=(center_x, bottom_y))
        self.screen.blit(shadow_surf, rect)



    
        

    def world_to_screen(self, x_lateral, z_forward):
        origin_x = self.panel_x0 + self.panel_w // 2
        origin_y = self.panel_y0 + self.panel_h - int(100 * self._geo_scale)  
        screen_x = origin_x + int(x_lateral * self.pixels_per_meter)
        screen_y = origin_y - int(z_forward * self.pixels_per_meter)  
        return screen_x, screen_y


    def _draw_panel_frame(self):
        panel_surf = pygame.Surface((self.panel_w, self.panel_h), pygame.SRCALPHA)
        panel_surf.fill(self.colors["panel_bg"])
        self.screen.blit(panel_surf, (self.panel_x0, self.panel_y0))

        header_h = max(28, int(40 * self._geo_scale))
        pygame.draw.rect(
            self.screen, (28, 30, 36),
            (self.panel_x0, self.panel_y0, self.panel_w, header_h),
        )


        pygame.draw.line(
            self.screen,
            self.colors["accent"],
            (self.panel_x0, self.panel_y0 + header_h),
            (self.panel_x0 + self.panel_w, self.panel_y0 + header_h),
            width=max(2, int(2 * self._geo_scale))
        )

        title = self.font_small.render("COLLISION AVOIDANCE", True, self.colors["accent"])
        self.screen.blit(title, title.get_rect(
            midleft=(self.panel_x0 + 14, self.panel_y0 + header_h // 2)
        ))



        pygame.draw.rect(
            self.screen,
            self.colors["panel_border"],
            (self.panel_x0, self.panel_y0, self.panel_w, self.panel_h),
            width=2,
        )

        return header_h

    def _draw_speed_card(self, speed_km, x, y):
        card_w = int(96 * max(self._geo_scale, 0.5)) + 40
        card_h = 74
        card = pygame.Surface((card_w, card_h), pygame.SRCALPHA)
        pygame.draw.rect(card, (0, 0, 0, 140), (0, 0, card_w, card_h), border_radius=10)
        pygame.draw.rect(card, self.colors["panel_border"], (0, 0, card_w, card_h), width=1, border_radius=10)


        value = self.font_speed.render(str(int(speed_km)), True, (255, 255, 255))
        unit = self.font_small.render("km/h", True, (170, 170, 180))
        card.blit(value, value.get_rect(midtop=(card_w // 2, 6)))
        card.blit(unit, unit.get_rect(midtop=(card_w // 2, 6 + value.get_height())))
 
        self.screen.blit(card, (x, y))

    def render(self, detections, speed_kmh=0, camera_frame=None):
        content_top = self.panel_y0
        if self.mode == "panel" and camera_frame is not None:
            bg = self._frame_to_surface(camera_frame)
            self.screen.blit(bg, (0, 0))
            header_h = self._draw_panel_frame()
            content_top = self.panel_y0 + header_h
        else:
            self.screen.blit(self._bg_gradient, (self.panel_x0, self.panel_y0))

        self.draw_3d_grid()

    
        ego_x = self.panel_x0 + self.panel_w // 2
        ego_y = self.panel_y0 + self.panel_h - int(60 * self._geo_scale)
        if self.ego_icon is not None:
            self._draw_shadow(ego_x, ego_y + self.ego_icon.get_height() // 2 - 4, self.ego_icon.get_width())
            rect = self.ego_icon.get_rect(center=(ego_x, ego_y))
            self.screen.blit(self.ego_icon, rect)
        else:
            pygame.draw.rect(
                self.screen, self.colors["your car"],
                (ego_x - 20, ego_y - 20, 40, 80),
                border_radius=10,
            )

        projected = []
        for det in detections:
            x_lat, z_fwd = self.bev.to_bev(det["bbox"])
            if z_fwd <= 0:
                continue
            projected.append((z_fwd, x_lat, det))
        projected.sort(key=lambda p: p[0], reverse=True) 

        placed_rect = []  # for _decluter_rect 

        panel_bounds = pygame.Rect(self.panel_x0, self.panel_y0, self.panel_w, self.panel_h)

        for z_fwd, x_lat, det in projected:
            sx, sy = self.world_to_screen(x_lat, z_fwd)
            if not panel_bounds.collidepoint(sx, sy):
                continue

            

            risk_level = det.get("risk_level", 0)

            if risk_level > 0:
                scale = max(0.7, min(1.5, 18 / max(z_fwd, 5)))
            else:
                scale = max(0.5, min(1.5, 15 / max(z_fwd, 5)))
            scale *= self._geo_scale if self._geo_scale > 0 else 1.0
            scale = max(0.25, scale)

            icon = self._get_risk_icon(det["class_id"], scale, risk_level)
            if icon is not None:
                rect = icon.get_rect(center=(sx, sy))
                rect = self._dectlutter_rect(rect, placed_rect)
                self._draw_shadow(rect.centerx, rect.bottom - 4, rect.width)

                self.screen.blit(icon, rect)

                badge = self._get_scaled_risk_icon(risk_level, scale) if risk_level > 0 else None
                if badge is not None:
                    badge_rect = badge.get_rect(midbottom=(rect.centerx, rect.top + 6))

                    self.screen.blit(badge, badge_rect)

                placed_rect.append(rect) 

        self._draw_speed_card(speed_kmh, self.panel_x0 + 14, content_top + 14)

        pygame.display.flip()
        self.clock.tick(30)

    def _frame_to_surface(self, frame_gbr):
        import cv2

        frame_rgb = cv2.cvtColor(frame_gbr, cv2.COLOR_BGR2RGB)
        h, w = frame_gbr.shape[:2]
        surf = pygame.image.frombuffer(frame_rgb.tobytes(), (w, h), "RGB")
        if (w, h) != (self.width, self.height):
            surf = pygame.transform.smoothscale(surf, (self.width, self.height))
        return surf

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def quit(self):
        pygame.quit()