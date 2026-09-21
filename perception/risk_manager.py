import time
import risk_engine_cpp as _risk
from collections import deque
import numpy as np


class RiskManager:

    def __init__(self, bev, stale_after=1.0, window_size=6, min_window_dt=0.15, debug=True, warning_distance_m=4.0, caution_distance_m=8.0):
        self.bev = bev
        self.stale_after = stale_after
        self.min_window_dt = min_window_dt
        self._history = {} # Caching (z_forward, timestamp)
        self.debug = debug
        self._last_debug_print = 0.0
        self.warning_distance_m = warning_distance_m
        self.caution_distance_m = caution_distance_m
        self._filters = {}
        self._last_seen = {}



    def _proximity_risk_level(self, z_fwd):
        # Helper function to fallback and confirm risk_level using only distance
        if z_fwd <= 0:
            return 0
        if z_fwd < self.warning_distance_m:
            return 2
        if z_fwd < self.caution_distance_m:
            return 1
        return 0



    def _compute_ttc(self, det, track_id, now):
        NO_DATA = (0.0, -1.0, 0)

        z_fwd = det["z_fwd"]
        if z_fwd <= 0:
            return NO_DATA

        prev_t = self._last_seen.get(track_id)
        self._last_seen[track_id] = now  # cache  all
        if prev_t is None:
            return NO_DATA # first sight on object

        dt = now = prev_t
        if dt <= 0:
            return NO_DATA # dublicate detect, perf decrease case prevention

        kf = self._filters.get(track_id)
        if kf is None:
            kf = _risk.KalmanFilter()
            self._filters[track_id] = kf

        kf.predict(dt)
        r_x = 1.0
        r_z = r_x + 0.05 * z_fwd ** 2
        x_smooth, z_smooth = kf.update(det["x_lateral"], z_fwd, r_x, r_z)
        closing_speed_est = -kf.velocity_z()
        prev_z_reconstructed = z_smooth + closing_speed_est * dt

        result = _risk.evaluate_risk(prev_z=prev_z_reconstructed, curr_z=z_smooth, dt=dt)
        return result.closing_speed, result.ttc, result.risk_level



    def update(self, tracked, timestamp=None):
        now = timestamp if timestamp is not None else time.perf_counter()

        seen_ids = set()

        for det in tracked:
            x_lat, z_fwd = self.bev.to_bev(det.get("raw_bbox", det["bbox"])) # Make sure to add bith raw bbox and bbox
            det["x_lateral"] = x_lat
            det["z_fwd"] = z_fwd

            track_id = det["track_id"]
            seen_ids.add(track_id)

            closing_speed, ttc, ttc_risk_level = self._compute_ttc(det, track_id, now)

            det["closing_speed"] = closing_speed
            det["ttc"] = ttc

            proximity_risk_level = self._proximity_risk_level(z_fwd)
            det["risk_level"] = max(ttc_risk_level, proximity_risk_level)


        stale = [tid for tid in self._last_seen if tid not in seen_ids and now - self._last_seen[tid] > self.stale_after]
        for tid in stale:
            del self._last_seen[tid]
            self._filters.pop(tid, None)

        if self.debug and  now - self._last_debug_print > 1.0:
            self._last_debug_print = now
            levels = {det["track_id"]: (det["risk_level"], round(det["ttc"], 2)) for det in tracked}
            print(f"[risk_manager] -> track_id, (risk_level, ttc) : {levels}")

        return tracked

        
