import cv2
import numpy as np 
import pygame
from screeninfo import get_monitors

DEFAULT_RESOLUTION = (480, 800)

class HUD:

    def __init__(self, bev_transform):
        pygame.init()
        try:
            resolution = get_monitors()[0]
            self.height = resolution.height
            self.width = resolution.width
        except Exception:
            print("Warn: could not detect a monitor, setting standart resolution")
            self.height = DEFAULT_RESOLUTION[0]
            self.width = DEFAULT_RESOLUTION[1]

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("driveR HUD")
        self.clock = pygame.time.Clock()
        self.bev = bev_transform

        self.colors = {
            "background": (25, 25, 30),
            "grid": (50, 50, 60),
            "your car": (30, 30, 90),
            "detected car": (60, 170, 190),
            "warning": (255, 50, 50)
        }

        self.font = pygame.font.SysFont("segoeui", 24)
        self.pixels_per_meter = 20 # hardcoded value, need to change in future

    def draw_3d_grid(self):
        center_x = self.width * 0.5
        horizon_y = int(self.height * 0.3)

        for i in range(-500, 500, 100):
            pygame.draw.line(self.screen, self.colors["grid"], (center_x, horizon_y), (center_x + i, self.height), 2)
            
        for y in range(horizon_y + 50, int(self.height), 40):
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
        pygame.draw.rect(
            self.screen, 
            self.colors["your car"],
            (self.width // 2 - 20, self.height - 100, 40, 80), 
            border_radius=10,
        )

        for det in detections:
            x_lat, z_fwd = self.bev.to_bev(det["bbox"])
            if z_fwd <= 0:         
                continue
            sx, sy = self.world_to_screen(x_lat, z_fwd)
            color = self.colors["warning"] if z_fwd < 15 else self.colors["detected car"]
            pygame.draw.circle(self.screen, color, (sx, sy), 8)

        speed_surface = self.font.render(f"Speed {speed_kmh}", True, (255, 255, 255))
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