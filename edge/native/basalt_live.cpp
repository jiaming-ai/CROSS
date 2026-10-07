// basalt_live: Basalt's stereo-inertial VIO on a live stream (cross-edge, cross_edge/basalt.py).
//
//   basalt_live --cam-calib calib.json --config-path config.json [--num-threads 4] [--use-double 0]
//
// The same estimator, calibration and configuration as Basalt's basalt_vio (src/vio.cpp), fed from stdin instead of
// a dataset folder, one record after another (little endian):
//   'I' int64 t_ns, double gx gy gz ax ay az        an IMU sample (rad/s, m/s^2; IMU frame)
//   'F' int64 t_ns, uint32 w, uint32 h, w*h bytes left, w*h bytes right
//                                                    a rectified stereo frame (8-bit grey; Basalt reads 16-bit: v << 8)
//   'E'                                              end of the stream
// Every state Basalt estimates is written to stdout as one line (as basalt_vio's --save-trajectory, one per frame):
//   S t_ns px py pz qx qy qz qw vx vy vz             T_w_i of the IMU and its velocity in the world frame
// A frame is processed once an IMU sample after its time has arrived (Basalt integrates the IMU up to the frame).
// Messages go to stderr.  Built against the Basalt commit of scripts/vio/install_basalt.sh (install_basalt_live.sh).

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include <Eigen/Core>
#include <cereal/archives/json.hpp>
#include <sophus/se3.hpp>
#include <tbb/concurrent_queue.h>
#include <tbb/global_control.h>

#include <basalt/calibration/calibration.hpp>
#include <basalt/io/dataset_io.h>
#include <basalt/optical_flow/optical_flow.h>
#include <basalt/serialization/headers_serialization.h>
#include <basalt/utils/vio_config.h>
#include <basalt/vi_estimator/vio_estimator.h>

namespace {

bool read_exact(void* dst, size_t n) { return std::fread(dst, 1, n, stdin) == n; }

std::string arg(int argc, char** argv, const std::string& name, const std::string& fallback) {
  for (int i = 1; i + 1 < argc; i++)
    if (name == argv[i]) return argv[i + 1];
  return fallback;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string calib_path = arg(argc, argv, "--cam-calib", "");
  const std::string config_path = arg(argc, argv, "--config-path", "");
  const int num_threads = std::stoi(arg(argc, argv, "--num-threads", "0"));
  const bool use_double = arg(argc, argv, "--use-double", "0") != "0";
  if (calib_path.empty()) {
    std::cerr << "usage: basalt_live --cam-calib calib.json [--config-path config.json] [--num-threads N]" << std::endl;
    return 2;
  }
  std::unique_ptr<tbb::global_control> tbb_control;
  if (num_threads > 0)
    tbb_control = std::make_unique<tbb::global_control>(tbb::global_control::max_allowed_parallelism, num_threads);

  basalt::VioConfig vio_config;
  if (!config_path.empty()) vio_config.load(config_path);
  basalt::Calibration<double> calib;
  {
    std::ifstream is(calib_path, std::ios::binary);
    if (!is.is_open()) {
      std::cerr << "could not load camera calibration " << calib_path << std::endl;
      return 2;
    }
    cereal::JSONInputArchive archive(is);
    archive(calib);
  }
  if (calib.intrinsics.size() != 2) {
    std::cerr << "basalt_live expects a stereo calibration (2 cameras), got " << calib.intrinsics.size() << std::endl;
    return 2;
  }

  basalt::OpticalFlowBase::Ptr flow = basalt::OpticalFlowFactory::getOpticalFlow(vio_config, calib);
  basalt::VioEstimatorBase::Ptr vio =
      basalt::VioEstimatorFactory::getVioEstimator(vio_config, calib, basalt::constants::g, true, use_double);
  vio->initialize(Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero());
  tbb::concurrent_bounded_queue<basalt::PoseVelBiasState<double>::Ptr> states;
  flow->output_queue = &vio->vision_data_queue;
  vio->out_state_queue = &states;
  std::cerr << "basalt_live: ready (" << calib.intrinsics.size() << " cameras)" << std::endl;

  std::thread writer([&]() {
    basalt::PoseVelBiasState<double>::Ptr s;
    while (true) {
      states.pop(s);
      if (!s) break;
      const Eigen::Vector3d p = s->T_w_i.translation();
      const Eigen::Quaterniond q = s->T_w_i.unit_quaternion();
      std::printf("S %lld %.9f %.9f %.9f %.9f %.9f %.9f %.9f %.6f %.6f %.6f\n", static_cast<long long>(s->t_ns), p.x(),
                  p.y(), p.z(), q.x(), q.y(), q.z(), q.w(), s->vel_w_i.x(), s->vel_w_i.y(), s->vel_w_i.z());
      std::fflush(stdout);
    }
  });

  std::vector<uint8_t> buf;
  while (true) {
    char kind = 0;
    if (!read_exact(&kind, 1) || kind == 'E') break;
    int64_t t_ns = 0;
    if (!read_exact(&t_ns, sizeof(t_ns))) break;
    if (kind == 'I') {
      double v[6];
      if (!read_exact(v, sizeof(v))) break;
      basalt::ImuData<double>::Ptr d(new basalt::ImuData<double>);
      d->t_ns = t_ns;
      d->gyro = Eigen::Vector3d(v[0], v[1], v[2]);
      d->accel = Eigen::Vector3d(v[3], v[4], v[5]);
      vio->imu_data_queue.push(d);
    } else if (kind == 'F') {
      uint32_t wh[2];
      if (!read_exact(wh, sizeof(wh))) break;
      const size_t n = size_t(wh[0]) * wh[1];
      basalt::OpticalFlowInput::Ptr in(new basalt::OpticalFlowInput);
      in->t_ns = t_ns;
      in->img_data.resize(2);
      buf.resize(n);
      bool ok = true;
      for (int c = 0; c < 2 && ok; c++) {
        ok = read_exact(buf.data(), n);
        in->img_data[c].img.reset(new basalt::ManagedImage<uint16_t>(wh[0], wh[1]));
        uint16_t* out = in->img_data[c].img->ptr;
        for (size_t i = 0; i < n; i++) out[i] = static_cast<uint16_t>(buf[i]) << 8;
      }
      if (!ok) break;
      flow->input_queue.push(in);
    } else {
      std::cerr << "basalt_live: unknown record '" << kind << "'" << std::endl;
      break;
    }
  }
  flow->input_queue.push(nullptr);
  vio->imu_data_queue.push(nullptr);
  vio->maybe_join();
  vio->drain_input_queues();
  states.push(nullptr);
  writer.join();
  std::cerr << "basalt_live: finished" << std::endl;
  return 0;
}
