#include <pybind11/pybind11.h>
#include <Eigen/Dense>

namespace py = pybind11;

class KalmanFilter {
public:
    KalmanFilter() {
        state = Eigen::Vector4d::Zero();
        P = Eigen::Matrix4d::Identity() * 10.0;
        Q = Eigen::Matrix4d::Identity() * 0.1;

        H = Eigen::Matrix<double, 2, 4>::Zero();
        H(0, 0) = 1.0;  // for x_lat
        H(1, 1) = 1.0; // for z_fwd

    }



    std::pair<double, double> predict(double dt) {
        Eigen::Matrix4d F = Eigen::Matrix4d::Identity();
        F(0, 2) = dt;
        F(1, 3) = dt;


        state = F * state;
        P = F * P * F.transpose() + Q;
        return {state(0), state(1)};
    }


    std::pair<double, double> update(double x_meas, double z_meas, double r_x, double r_z) {
        Eigen::Vector2d z(x_meas, z_meas);
        Eigen::Matrix2d R = Eigen::Matrix2d::Zero();
        R(0, 0) = r_x;
        R(1, 1) = r_z;

        Eigen::Vector2d y = z - H * state;
        Eigen::Matrix2d S = H * P * H.transpose() + R;
        Eigen::Matrix<double, 4, 2> K = P * H.transpose() * S.inverse();
        
        state = state + K *  y;
        P = P - K * H * P;
        return {state(0), state(1)};

    }

    double velocity_z() const { return state(3); }
    double velocity_x() const { return state(2); }


private:
    Eigen::Vector4d state;
    Eigen::Matrix4d P;
    Eigen::Matrix4d Q;
    Eigen::Matrix<double, 2, 4> H;

};


void init_kalman_filter(py::module_ &m) {
    py::class_<KalmanFilter>(m, "KalmanFilter")
        .def(py::init<>())
        .def("predict", &KalmanFilter::predict, py::arg("dt"))
        .def("update", &KalmanFilter::update, py::arg("x_meas"), py::arg("z_meas"), py::arg("r_x"), py::arg("r_z"))
        .def("velocity_z", &KalmanFilter::velocity_z)
        .def("velocity_x", &KalmanFilter::velocity_x);
} 

