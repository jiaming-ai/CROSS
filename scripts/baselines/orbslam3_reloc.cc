// ORB-SLAM3 driver for image-folder sequences (SimChange / benchmark layout: left/, right/, depth/ PNG folders).
//
//   orbslam3_reloc <voc> <settings.yaml> <sequence_dir> <out_poses.txt> [--localization] [--fps F]
//                  [--sensor stereo|rgbd|mono|imu_mono] [--left-dir left] [--right-dir right] [--depth-dir depth]
//                  [--times frame_times.txt] [--imu imu.txt] [--imu-time-offset S]
//
// imu_mono: monocular-inertial; needs --times (one camera timestamp per image, seconds) and --imu (rows
// "t wx wy wz ax ay az" in the IMU frame, '#' comments allowed; the settings carry the IMU.* block).  The IMU
// samples in (t_{i-1}, t_i] go with image i; --imu-time-offset S is added to the IMU timestamps (camera clock =
// IMU clock + S).  --times also replaces the i / fps timestamps of the other sensors.
//
// rgbd: depth PNGs are uint16 (scale RGBD.DepthMapFactor of the settings, 1000 = millimetres).
// The settings file controls atlas loading/saving (System.LoadAtlasFromFile / SaveAtlasToFile).
// With --localization the tracking runs in localization-only mode against the loaded atlas.
// Every frame writes:  idx  state  t00 t01 ... t33   (camera-to-world, OpenCV convention)
// and, with --times, three more: the number of maps in the atlas, whether the active map's IMU is initialized (0/1;
// always 0 without an IMU), and the active map's id
// state (Tracking::eTrackingState): -1=SYSTEM_NOT_READY, 0=NO_IMAGES_YET, 1=NOT_INITIALIZED, 2=OK, 3=RECENTLY_LOST, 4=LOST
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <dirent.h>
#include <algorithm>
// the atlas (IMU state, active map id) is a private member of System
#define private public
#include <System.h>
#undef private

static std::vector<std::vector<double>> readRows(const std::string& path) {
    std::vector<std::vector<double>> rows;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        std::vector<double> r;
        double v;
        while (ss >> v) r.push_back(v);
        if (!r.empty()) rows.push_back(r);
    }
    return rows;
}

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

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "usage: orbslam3_reloc <voc> <settings.yaml> <sequence_dir> <out_poses.txt> [--localization] [--fps F]\n";
        return 1;
    }
    std::string voc = argv[1], settings = argv[2], seq = argv[3], out = argv[4];
    bool localization = false, atlas_loaded = false;
    double fps = 10.0;
    std::string right_dir = "right", left_dir = "left", depth_dir = "depth", sensor = "stereo";
    std::string times_file, imu_file;
    double imu_time_offset = 0.0;
    int pace_ms = 5;
    for (int i = 5; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--localization") localization = true;
        else if (a == "--fps" && i + 1 < argc) fps = atof(argv[++i]);
        else if (a == "--right-dir" && i + 1 < argc) right_dir = argv[++i];
        else if (a == "--left-dir" && i + 1 < argc) left_dir = argv[++i];
        else if (a == "--depth-dir" && i + 1 < argc) depth_dir = argv[++i];
        else if (a == "--sensor" && i + 1 < argc) sensor = argv[++i];
        else if (a == "--pace-ms" && i + 1 < argc) pace_ms = atoi(argv[++i]);
        else if (a == "--times" && i + 1 < argc) times_file = argv[++i];
        else if (a == "--imu" && i + 1 < argc) imu_file = argv[++i];
        else if (a == "--imu-time-offset" && i + 1 < argc) imu_time_offset = atof(argv[++i]);
    }
    const bool inertial = sensor == "imu_mono";
    auto left = listPng(seq + "/" + left_dir);
    std::vector<std::string> right;
    if (sensor == "stereo") right = listPng(seq + "/" + right_dir);
    else if (sensor == "rgbd") right = listPng(seq + "/" + depth_dir);
    else right = left;
    if (left.empty() || left.size() != right.size()) {
        std::cerr << "bad sequence " << seq << " (" << left.size() << "/" << right.size() << ")\n";
        return 1;
    }
    std::vector<double> times;
    if (!times_file.empty())
        for (auto& r : readRows(times_file)) times.push_back(r[0]);
    if (!times.empty() && times.size() < left.size()) {
        std::cerr << "bad times " << times_file << " (" << times.size() << " < " << left.size() << " images)\n";
        return 1;
    }
    std::vector<ORB_SLAM3::IMU::Point> imu;
    if (inertial) {
        if (times.empty() || imu_file.empty()) {
            std::cerr << "imu_mono needs --times and --imu\n";
            return 1;
        }
        for (auto& r : readRows(imu_file))
            if (r.size() >= 7) imu.emplace_back(r[4], r[5], r[6], r[1], r[2], r[3], r[0] + imu_time_offset);
        std::cerr << imu.size() << " IMU samples, " << times.size() << " frame times\n";
    }
    size_t next_imu = 0;
    if (inertial)       // samples up to the first image are not used (ORB-SLAM3 examples)
        while (next_imu < imu.size() && imu[next_imu].t <= times[0]) ++next_imu;
    {
        cv::FileStorage fs(settings, cv::FileStorage::READ);
        atlas_loaded = !fs["System.LoadAtlasFromFile"].empty();
    }
    const auto type = sensor == "rgbd" ? ORB_SLAM3::System::RGBD
                    : sensor == "mono" ? ORB_SLAM3::System::MONOCULAR
                    : inertial ? ORB_SLAM3::System::IMU_MONOCULAR : ORB_SLAM3::System::STEREO;
    ORB_SLAM3::System SLAM(voc, settings, type, false);
    if (localization) SLAM.ActivateLocalizationMode();
    std::ofstream f(out);
    f << std::setprecision(9);
    double t_total = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        cv::Mat imL = cv::imread(left[i], cv::IMREAD_COLOR);   // 8-bit: SimChange-Long renders are 16-bit PNGs
        double ts = times.empty() ? i / fps : times[i];
        std::vector<ORB_SLAM3::IMU::Point> meas;
        if (inertial && i > 0)
            while (next_imu < imu.size() && imu[next_imu].t <= ts) meas.push_back(imu[next_imu++]);
        auto t0 = std::chrono::steady_clock::now();
        Sophus::SE3f Tcw;
        if (sensor == "rgbd") Tcw = SLAM.TrackRGBD(imL, cv::imread(right[i], cv::IMREAD_UNCHANGED), ts);
        else if (sensor == "mono") Tcw = SLAM.TrackMonocular(imL, ts);
        else if (inertial) Tcw = SLAM.TrackMonocular(imL, ts, meas);
        else Tcw = SLAM.TrackStereo(imL, cv::imread(right[i], cv::IMREAD_COLOR), ts);
        t_total += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        int state = SLAM.GetTrackingState();
        // in a multi-session run the new session lives in its own map until it is merged into the loaded
        // atlas; only after the merge (atlas back to one map) are its poses expressed in the map frame
        int nmaps = SLAM.GetNumberOfMaps();
        if (atlas_loaded && !localization && nmaps != 1 && state == 2) state = 6;   // 6 = tracking in an unmerged session
        Sophus::SE3f Twc = Tcw.inverse();
        Eigen::Matrix4f M = Twc.matrix();
        f << i << " " << state;
        for (int r = 0; r < 4; ++r) for (int c = 0; c < 4; ++c) f << " " << M(r, c);
        if (!times.empty()) {
            ORB_SLAM3::Map* active = SLAM.mpAtlas->GetCurrentMap();
            f << " " << nmaps << " " << int(inertial && SLAM.mpAtlas->isImuInitialized()) << " "
              << (active ? long(active->GetId()) : -1L);
        }
        f << "\n";
        f.flush();
        // pace slightly so that the local mapping thread keeps up (no real-time constraint in offline eval)
        std::this_thread::sleep_for(std::chrono::milliseconds(pace_ms));
    }
    std::cerr << "tracking time per frame: " << 1000.0 * t_total / left.size() << " ms\n";
    // final (post-merge / post-BA) trajectory of every non-lost frame, in the frame of the largest map: a session
    // that is merged into the atlas late in a trial still gets its earlier frames expressed in the map frame
    int nmaps_final = SLAM.GetNumberOfMaps();
    SLAM.Shutdown();
    SLAM.SaveTrajectoryWithMapId(out + ".final");
    { std::ofstream fn(out + ".final_nmaps"); fn << nmaps_final << "\n"; }
    return 0;
}
