import time
import risk_engine_cpp as _risk
from collections import deque


class RiskManager:

    def __init__(self, bev, stale_after=1.0, window_size=6, min_window_dt=0.15, debug=True, warning_distance_m=4, caution_distance_m=8):
        self.bev = bev
        self.stale_after = stale_after
        self.window_size = window_size
        self.min_window_dt = min_window_dt
        self._history = {} # Caching (z_forward, timestamp)
        self.debug = debug
        self._last_debug_print = 0.0
        self.warning_distance_m = warning_distance_m
        self.caution_distance_m = caution_distance_m



    def _proximity_risk_level(self, z_fwd):
        # Helper function to fallback and confirm risk_level using only distance
        if z_fwd <= 0:
            return 0
        if z_fwd < self.caution_distance_m:
            return 1
        if z_fwd < self.warning_distance_m:
            return 2
        return 0

    def update(self, tracked, timestamp=None):
        now = timestamp if timestamp is not None else time.perf_counter()

        seen_ids = set()

        for det in tracked:
            x_lat, z_fwd = self.bev.to_bev(det.get("raw_bbox", det["bbox"])) # Make sure to add bith raw bbox and bbox
            det["x_lateral"] = x_lat
            det["z_fwd"] = z_fwd

            track_id = det["track_id"]
            seen_ids.add(track_id)
            window = self._history.setdefault(track_id, deque(maxlen=self.window_size))

            ttc_risk_level = 0

            if z_fwd > 0:
                oldest = window[0] if window else None
                if oldest is not None and (now - oldest[1]) >= self.min_window_dt:
                    prev_z, prev_t = oldest # keep track of frames position and time
                    dt = now - prev_t
                    result = _risk.evaluate_risk(prev_z=prev_z, curr_z=z_fwd, dt=dt)


                    det["closing_speed"] = result.closing_speed
                    det["ttc"] = result.ttc
                    ttc_risk_level = result.risk_level
                else:
                    det["closing_speed"] = 0.0
                    det["ttc"] = -1
                    ttc_risk_level = 0

                window.append((z_fwd, now))
            else:
                det["closing_speed"] = 0.0
                det["ttc"] = -1
                ttc_risk_level = 0

            proximity_risk_level = self._proximity_risk_level(z_fwd)
            det["risk_level"] = max(ttc_risk_level, proximity_risk_level)


        stale = [tid for tid, window in self._history.items() if tid not in seen_ids and window and now - window[-1][1] > self.stale_after]
        for tid in stale:
            del self._history[tid]

        if self.debug and  now - self._last_debug_print > 1.0:
            self._last_debug_print = now
            levels = {det["track_id"]: (det["risk_level"], round(det["ttc"], 2)) for det in tracked}
            print(f"[risk_manager] -> track_id, (risk_level, ttc) : {levels}")

        return tracked

        
