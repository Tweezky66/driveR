#include <pybind11/pybind11.h>

namespace py = pybind11;

struct RiskResult {
    double closing_speed;
    double ttc;
    int risk_level;
};

RiskResult evaluate_risk(double prev_z, double curr_z, double dt, double caution_ttc=4.0, double warning_ttc=2.0) {

    RiskResult result{0.0, -1.0, 0};
    
    if (dt <= 0 || curr_z <= 0) {
        return result;  // negative distance or change in time, return
    }



    double closing_speed = (prev_z - curr_z) / dt; 
    result.closing_speed = closing_speed;
    

    if (closing_speed <= 0.0) {
        return result; 
    }

    double ttc = curr_z / closing_speed;
    result.ttc = ttc;

    if (ttc <= warning_ttc) {
        result.risk_level = 2;
    } else if (ttc <= caution_ttc) {
        result.risk_level = 1;
    }
    return result;

}

PYBIND11_MODULE(risk_engine_cpp, m) {
    m.doc() = "Deterministic TTC/risk evaluation - bounded worst-case execution "
               "time, no GC, no dynamic allocation. Consumes track-to-track "
               "position deltas from tracker.py; does not do tracking itself)";
 
    py::class_<RiskResult>(m, "RiskResult")
        .def_readonly("closing_speed", &RiskResult::closing_speed)
        .def_readonly("ttc", &RiskResult::ttc)
        .def_readonly("risk_level", &RiskResult::risk_level);
 
    m.def("evaluate_risk", &evaluate_risk,
          py::arg("prev_z"), py::arg("curr_z"), py::arg("dt"),
          py::arg("caution_ttc") = 4.0, py::arg("warning_ttc") = 2.0,
          "Compute closing speed, TTC, and risk level (0=safe, 1=caution, "
          "2=warning) from two ground-plane forward-distance samples of the "
          "SAME tracked object and the time delta between them.");
}
