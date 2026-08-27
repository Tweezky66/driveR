import time
import risk_engine_src as _risk


class RiskManager:

    def __init__(self, bev, stale_after=1.0):
        self.bev = bev
        self.stale_after = stale_after
        self._history = {} # Caching (z_forward, timestamp)


    def update(self, tracked):
        now = time.perf_counter()

        seen_ids = set()

        for det in tracked:
            x_lat, z_fwd = self.bev.to_bev(det["bbox"])
            det["x_lateral"] = x_lat
            det["z_fwd"] = z_fwd

            track_id = det["track_id"]
            seen_ids.add(track_id)

            prev = self._history.get(track_id)
            if prev is not None and  z_fwd > 0:
                prev_z, prev_t = prev
                dt = now - prev_t
                result = _risk.evaluate_risk(prev_z=prev_z, curr_z=z_fwd, dt=dt)

                det["closing_speed"] = result.closing_speed
                det["ttc"] = result.ttc
                det["risk_level"] = result.risk_level

            else:
                det["closing_speed"] = 0.0
                det["ttc"] = -1.0
                det["risk_level"] = 0

            self._history[track_id] = (z_fwd, now)

        stale = [tid for tid, (_, t) in self._history.items() if tid not in seen_ids and now - t > self.stale_after]
        for tid in stale:
            del tid

        return tracked
