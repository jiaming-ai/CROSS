// ORB-SLAM3 driver for image-folder sequences (SimChange / benchmark layout: left/, right/, depth/ PNG folders).
//
//   orbslam3_reloc <voc> <settings.yaml> <sequence_dir> <out_poses.txt> [--localization] [--fps F]
//                  [--sensor stereo|rgbd|mono] [--left-dir left] [--right-dir right] [--depth-dir depth]
//
// rgbd: depth PNGs are uint16 (scale RGBD.DepthMapFactor of the settings, 1000 = millimetres).
// The settings file controls atlas loading/saving (System.LoadAtlasFromFile / SaveAtlasToFile).
// With --localization the tracking runs in localization-only mode against the loaded atlas.
// Every frame writes:  idx  state  t00 t01 ... t33   (camera-to-world, OpenCV convention)
// state (Tracking::eTrackingState): -1=SYSTEM_NOT_READY, 0=NO_IMAGES_YET, 1=NOT_INITIALIZED, 2=OK, 3=RECENTLY_LOST, 4=LOST
#include <System.h>
#include <opencv2/opencv.hpp>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <thread>
#include <dirent.h>
#include <algorithm>

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
    }
    auto left = listPng(seq + "/" + left_dir);
    std::vector<std::string> right;
    if (sensor == "stereo") right = listPng(seq + "/" + right_dir);
    else if (sensor == "rgbd") right = listPng(seq + "/" + depth_dir);
    else right = left;
    if (left.empty() || left.size() != right.size()) {
        std::cerr << "bad sequence " << seq << " (" << left.size() << "/" << right.size() << ")\n";
        return 1;
    }
    {
        cv::FileStorage fs(settings, cv::FileStorage::READ);
        atlas_loaded = !fs["System.LoadAtlasFromFile"].empty();
    }
    const auto type = sensor == "rgbd" ? ORB_SLAM3::System::RGBD
                    : sensor == "mono" ? ORB_SLAM3::System::MONOCULAR : ORB_SLAM3::System::STEREO;
    ORB_SLAM3::System SLAM(voc, settings, type, false);
    if (localization) SLAM.ActivateLocalizationMode();
    std::ofstream f(out);
    f << std::setprecision(9);
    double t_total = 0;
    for (size_t i = 0; i < left.size(); ++i) {
        cv::Mat imL = cv::imread(left[i], cv::IMREAD_COLOR);   // 8-bit: SimChange-Long renders are 16-bit PNGs
        double ts = i / fps;
        auto t0 = std::chrono::steady_clock::now();
        Sophus::SE3f Tcw;
        if (sensor == "rgbd") Tcw = SLAM.TrackRGBD(imL, cv::imread(right[i], cv::IMREAD_UNCHANGED), ts);
        else if (sensor == "mono") Tcw = SLAM.TrackMonocular(imL, ts);
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
