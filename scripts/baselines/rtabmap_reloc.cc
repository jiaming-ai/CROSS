// RTAB-Map RGB-D mapping / localization driver for SimChange-style sequences.
//
//   rtabmap_reloc <sequence_dir> <odom.txt> <database.db> <out_poses.txt> [--localization] [--fps F] [--Param value ...]
//
// sequence_dir: left/*.png, depth_mm/*.png (16-bit millimetres), calib.json (fx fy cx cy read from calib_pinhole.txt)
// odom.txt    : per-frame odometry camera-to-world (OpenCV convention), 16 values per row (noisy, as given to CROSS)
// Mapping run : creates database.db (Mem/IncrementalMemory=true).
// Localization: opens database.db read-only (Mem/IncrementalMemory=false, Mem/InitWMWithAllNodes=true).
// Every frame writes:  idx  state  t00 ... t33  (camera-to-world in the map frame; state 3 once localized)
#include <rtabmap/core/Rtabmap.h>
#include <rtabmap/core/SensorData.h>
#include <rtabmap/core/CameraModel.h>
#include <rtabmap/core/StereoCameraModel.h>
#include <rtabmap/core/Parameters.h>
#include <rtabmap/core/Transform.h>
#include <rtabmap/utilite/ULogger.h>
#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
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
    bool localization = false;
    double fps = 10.0;
    std::string stereo_dir;          // stereo mode: directory with the right images (rectified, same intrinsics)
    double baseline = 0.0;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--localization") localization = true;
        else if (a == "--fps" && i + 1 < argc) fps = atof(argv[++i]);
        else if (a == "--stereo" && i + 2 < argc) { stereo_dir = argv[++i]; baseline = atof(argv[++i]); }
    }
    ParametersMap params = Parameters::parseArguments(argc, argv, true);
    params.insert(ParametersPair(Parameters::kRGBDEnabled(), "true"));
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
    if (!localization) remove(db.c_str());
    Rtabmap rtabmap;
    rtabmap.init(params, db);

    Transform optical = CameraModel::opticalRotation();       // base -> camera
    std::ofstream fo(out);
    fo << std::setprecision(9);
    int nLoc = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        cv::Mat rgb = cv::imread(left[i], cv::IMREAD_COLOR);
        SensorData data;
        if (stereo) {
            cv::Mat l8, r8;
            cv::cvtColor(rgb, l8, cv::COLOR_BGR2GRAY);
            cv::cvtColor(cv::imread(depth[i], cv::IMREAD_COLOR), r8, cv::COLOR_BGR2GRAY);
            data = SensorData(l8, r8, stereoModel, (int)i + 1, i / fps);
        } else {
            cv::Mat d16 = cv::imread(depth[i], cv::IMREAD_UNCHANGED);
            data = SensorData(rgb, d16, model, (int)i + 1, i / fps);
        }
        Transform odomBase = odom[i] * optical.inverse();        // camera c2w -> base c2w
        rtabmap.process(data, odomBase);
        const Statistics& st = rtabmap.getStatistics();
        int lc = st.loopClosureId() > 0 ? st.loopClosureId() : st.proximityDetectionId();
        if (lc > 0) nLoc++;
        bool localized = localization ? (nLoc > 0) : true;
        Transform mapPose = rtabmap.getMapCorrection() * odomBase * optical;   // camera c2w in map frame
        Eigen::Matrix4f M = mapPose.toEigen4f();
        fo << i << " " << (localized ? 2 : 1);   // 2 = localized (same code as ORB-SLAM3 OK)
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) fo << " " << M(r, c);
        fo << " " << lc << "\n";
        fo.flush();
    }
    std::cerr << "frames with a loop/proximity detection: " << nLoc << "/" << left.size() << "\n";
    rtabmap.close(true);
    return 0;
}
