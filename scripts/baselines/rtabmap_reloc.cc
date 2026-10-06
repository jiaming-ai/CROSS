// RTAB-Map RGB-D mapping / localization driver for SimChange-style sequences.
//
//   rtabmap_reloc <sequence_dir> <odom.txt> <database.db> <out_poses.txt> [--localization] [--vo] [--fps F]
//                 [--imu imu.txt --times frame_times.txt --T-cam-imu T.txt] [--Param value ...]
//
// sequence_dir: left/*.png, depth_mm/*.png (16-bit millimetres), calib.json (fx fy cx cy read from calib_pinhole.txt)
// odom.txt    : per-frame odometry camera-to-world (OpenCV convention), 16 values per row (noisy, as given to CROSS)
// Mapping run : creates database.db (Mem/IncrementalMemory=true).
// Localization: opens database.db read-only (Mem/IncrementalMemory=false, Mem/InitWMWithAllNodes=true).
// --vo        : RTAB-Map's own visual odometry (Odom/Strategy, default frame-to-map) instead of odom.txt (still read for
//               the frame count); after a frame without odometry it resets to its latest pose (Odom/ResetCountdown 1)
//               and frames without odometry are written with state 0.
// --imu       : the camera's IMU (rows "t wx wy wz ax ay az" in the IMU frame, '#' comments; --times: one timestamp per
//               image on the IMU clock; --T-cam-imu: 16 values, x_cam = T x_imu).  RTAB-Map's complementary filter turns
//               the samples up to each image into an orientation, attached to the image's SensorData with the IMU's pose
//               in the base frame: the visual odometry uses it for its motion guess and gravity, the map graph for its
//               gravity constraints (Optimizer/GravitySigma).
// Every frame writes:  idx  state  t00 ... t33  (camera-to-world in the map frame; state 3 once localized)
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/Odometry.h>
#include <rtabmap/core/OdometryInfo.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/core/IMU.h>
#include <rtabmap/core/IMUFilter.h>
#include <rtabmap/utilite/ULogger.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace rtabmap;

static std::vector<std::string> listPng(const std::string& dir) {
    std::vector<std::string> files;
    DIR* d = opendir(dir.c_str());
    if (!d) return files;
    while (dirent* e = readdir(d)) {
        std::string n = e->d_name;
        if (n.size() > 4 && n.substr(n.size() - 4) == ".png") files.push_back(dir + "/" + n);
    }
    closedir(d);
    std::sort(files.begin(), files.end());
    return files;
}

static Transform fromRow(const std::vector<double>& v) {
    return Transform((float)v[0], (float)v[1], (float)v[2], (float)v[3], (float)v[4], (float)v[5], (float)v[6], (float)v[7],
                     (float)v[8], (float)v[9], (float)v[10], (float)v[11]);
}

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: rtabmap_reloc <sequence_dir> <odom.txt> <database.db> <out_poses.txt> [--localization] [--fps F] [--Param value]\n";
        return 1;
    }
    ULogger::setType(ULogger::kTypeConsole);
    ULogger::setLevel(ULogger::kWarning);
    std::string seq = argv[1], odomFile = argv[2], db = argv[3], out = argv[4];
    bool localization = false, visualOdometry = false;
    double fps = 10.0;
    std::string stereo_dir;          // stereo mode: directory with the right images (rectified, same intrinsics)
    std::string imu_file, times_file, tci_file;
    double baseline = 0.0;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--localization") localization = true;
        else if (a == "--vo") visualOdometry = true;
        else if (a == "--fps" && i + 1 < argc) fps = atof(argv[++i]);
        else if (a == "--stereo" && i + 2 < argc) { stereo_dir = argv[++i]; baseline = atof(argv[++i]); }
        else if (a == "--imu" && i + 1 < argc) imu_file = argv[++i];
        else if (a == "--times" && i + 1 < argc) times_file = argv[++i];
        else if (a == "--T-cam-imu" && i + 1 < argc) tci_file = argv[++i];
    }
    ParametersMap params = Parameters::parseArguments(argc, argv, true);
    params.insert(ParametersPair(Parameters::kRGBDEnabled(), "true"));
    params.insert(ParametersPair(Parameters::kOdomResetCountdown(), "1"));   // (used by --vo only)
    params.insert(ParametersPair(Parameters::kMemIncrementalMemory(), localization ? "false" : "true"));
    if (localization) {
        params.insert(ParametersPair(Parameters::kMemInitWMWithAllNodes(), "true"));
        params.insert(ParametersPair(Parameters::kRGBDStartAtOrigin(), "true"));
    }

    // calibration
    double fx, fy, cx, cy;
    int width, height;
    {
        std::ifstream c(seq + "/calib_pinhole.txt");
        if (!(c >> fx >> fy >> cx >> cy >> width >> height)) {
            std::cerr << "cannot read " << seq << "/calib_pinhole.txt\n";
            return 1;
        }
    }
    CameraModel model("sim", fx, fy, cx, cy, CameraModel::opticalRotation(), 0, cv::Size(width, height));
    StereoCameraModel stereoModel(fx, fy, cx, cy, baseline, CameraModel::opticalRotation(), cv::Size(width, height));
    const bool stereo = !stereo_dir.empty();

    auto left = listPng(seq + "/left");
    auto depth = stereo ? listPng(seq + "/" + stereo_dir) : listPng(seq + "/depth_mm");
    if (left.empty() || left.size() != depth.size()) {
        std::cerr << "bad sequence (" << left.size() << " rgb / " << depth.size() << (stereo ? " right)" : " depth)") << "\n";
        return 1;
    }
    std::vector<Transform> odom;
    {
        std::ifstream f(odomFile);
        std::string line;
        while (std::getline(f, line)) {
            std::istringstream ss(line);
            std::vector<double> v;
            double x;
            while (ss >> x) v.push_back(x);
            if (v.size() == 16) odom.push_back(fromRow(v));
        }
    }
    if (odom.size() != left.size()) {
        std::cerr << "odometry rows " << odom.size() << " != frames " << left.size() << "\n";
        return 1;
    }
    // IMU stream (optional)
    auto readRows = [](const std::string& path) {
        std::vector<std::vector<double>> rows;
        std::ifstream f(path);
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::istringstream ss(line);
            std::vector<double> v;
            double x;
            while (ss >> x) v.push_back(x);
            if (!v.empty()) rows.push_back(v);
        }
        return rows;
    };
    std::vector<std::vector<double>> imu;
    std::vector<double> times;
    Transform imuLocal;            // the IMU in the base frame
    std::unique_ptr<IMUFilter> imuFilter;
    if (!imu_file.empty()) {
        for (auto& r : readRows(imu_file)) if (r.size() >= 7) imu.push_back(r);
        for (auto& r : readRows(times_file)) times.push_back(r[0]);
        std::vector<double> T;
        for (auto& r : readRows(tci_file)) T.insert(T.end(), r.begin(), r.end());
        if (imu.empty() || times.size() < left.size() || T.size() < 12) {
            std::cerr << "--imu needs IMU rows, one time per image (--times) and --T-cam-imu (" << imu.size() << " / "
                      << times.size() << " / " << T.size() << ")\n";
            return 1;
        }
        Transform Tci((float)T[0], (float)T[1], (float)T[2], (float)T[3], (float)T[4], (float)T[5], (float)T[6], (float)T[7],
                      (float)T[8], (float)T[9], (float)T[10], (float)T[11]);
        imuLocal = CameraModel::opticalRotation() * Tci;
        imuFilter.reset(IMUFilter::create(IMUFilter::kComplementaryFilter, params));
        std::cerr << imu.size() << " IMU samples, IMU in base frame: " << imuLocal.prettyPrint() << "\n";
    }
    size_t nextImu = 0;
    while (!imu.empty() && nextImu < imu.size() && imu[nextImu][0] < times[0] - 1.0) ++nextImu;   // 1 s of filter warm-up
    if (!localization) remove(db.c_str());
    Rtabmap rtabmap;
    rtabmap.init(params, db);

    Transform optical = CameraModel::opticalRotation();       // base -> camera
    std::ofstream fo(out);
    fo << std::setprecision(9);
    int nLoc = 0, nLost = 0;
    std::unique_ptr<Odometry> vo(visualOdometry ? Odometry::create(params) : nullptr);
    Transform lastOdom = Transform::getIdentity();
    for (size_t i = 0; i < left.size(); ++i) {
        cv::Mat rgb = cv::imread(left[i], cv::IMREAD_COLOR);
        SensorData data;
        if (stereo) {
            cv::Mat l8, r8;
            cv::cvtColor(rgb, l8, cv::COLOR_BGR2GRAY);
            cv::cvtColor(cv::imread(depth[i], cv::IMREAD_COLOR), r8, cv::COLOR_BGR2GRAY);
            data = SensorData(l8, r8, stereoModel, (int)i + 1, imuFilter ? times[i] : i / fps);
        } else {
            cv::Mat d16 = cv::imread(depth[i], cv::IMREAD_UNCHANGED);
            data = SensorData(rgb, d16, model, (int)i + 1, imuFilter ? times[i] : i / fps);
        }
        if (imuFilter) {            // samples up to this image -> orientation of the IMU
            const std::vector<double>* last = nullptr;
            while (nextImu < imu.size() && imu[nextImu][0] <= times[i]) {
                const auto& r = imu[nextImu++];
                imuFilter->update(r[1], r[2], r[3], r[4], r[5], r[6], r[0]);
                last = &r;
            }
            if (last) {
                double qx, qy, qz, qw;
                imuFilter->getOrientation(qx, qy, qz, qw);
                cv::Mat cov = cv::Mat::eye(3, 3, CV_64FC1) * 0.01;
                data.setIMU(IMU(cv::Vec4d(qx, qy, qz, qw), cov, cv::Vec3d((*last)[1], (*last)[2], (*last)[3]), cov,
                                cv::Vec3d((*last)[4], (*last)[5], (*last)[6]), cov, imuLocal));
            }
        }
        Transform odomBase = odom[i] * optical.inverse();        // camera c2w -> base c2w
        int lc = 0;
        bool tracked = true;
        if (vo) {
            OdometryInfo info;
            Transform p = vo->process(data, &info);
            tracked = !p.isNull();
            if (tracked) {
                odomBase = lastOdom = p;
                rtabmap.process(data, odomBase, info.reg.covariance);
            } else {
                odomBase = lastOdom;
                nLost++;
            }
        } else {
            rtabmap.process(data, odomBase);
        }
        if (tracked) {
            const Statistics& st = rtabmap.getStatistics();
            lc = st.loopClosureId() > 0 ? st.loopClosureId() : st.proximityDetectionId();
            if (lc > 0) nLoc++;
        }
        bool localized = tracked && (localization ? (nLoc > 0) : true);
        Transform mapPose = rtabmap.getMapCorrection() * odomBase * optical;   // camera c2w in map frame
        Eigen::Matrix4f M = mapPose.toEigen4f();
        fo << i << " " << (localized ? 2 : tracked ? 1 : 0);   // 2 = localized (same code as ORB-SLAM3 OK), 0 = no odometry
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) fo << " " << M(r, c);
        fo << " " << lc << "\n";
        fo.flush();
    }
    std::cerr << "frames with a loop/proximity detection: " << nLoc << "/" << left.size() << "\n";
    if (vo) std::cerr << "frames without visual odometry: " << nLost << "/" << left.size() << "\n";
    // final optimized graph (node id = frame index + 1): the mapping trajectory after all loop closures
    {
        std::map<int, Transform> poses;
        std::multimap<int, Link> links;
        rtabmap.getGraph(poses, links, true, true);
        std::ofstream ff(out + ".final");
        ff << std::setprecision(9);
        for (const auto& kv : poses) {
            if (kv.first <= 0 || kv.second.isNull()) continue;
            Eigen::Matrix4f M = (kv.second * optical).toEigen4f();
            ff << kv.first - 1 << " 2";
            for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) ff << " " << M(r, c);
            ff << "\n";
        }
    }
    rtabmap.close(true);
    return 0;
}
