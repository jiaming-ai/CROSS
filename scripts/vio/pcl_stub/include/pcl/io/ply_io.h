#pragma once
#include <fstream>
#include <string>
#include <pcl/point_types.h>
namespace pcl { namespace io {
template <typename P> int savePLYFileASCII(const std::string& name, const PointCloud<P>& c) {
  std::ofstream f(name);
  f << "ply\nformat ascii 1.0\nelement vertex " << c.points.size()
    << "\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
  for (const auto& p : c.points) f << p.x << " " << p.y << " " << p.z << "\n";
  return f ? 0 : -1;
}
}}  // namespace pcl::io
