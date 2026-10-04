// Minimal stand-in for PCL (OKVIS2-X uses it only to write ASCII PLY point clouds).
#pragma once
#include <cstdint>
#include <vector>
namespace pcl {
struct PointXYZ { float x = 0, y = 0, z = 0; };
template <typename P> struct PointCloud {
  uint32_t width = 0, height = 1; bool is_dense = true; std::vector<P> points;
  void resize(size_t n) { points.resize(n); }
  size_t size() const { return points.size(); }
  typename std::vector<P>::iterator begin() { return points.begin(); }
  typename std::vector<P>::iterator end() { return points.end(); }
};
}  // namespace pcl
