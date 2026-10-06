# Header-only stand-in for PCL (see include/pcl): enough for OKVIS2-X's PLY output.
get_filename_component(_pcl_root "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)
set(PCL_FOUND TRUE)
set(PCL_INCLUDE_DIRS "${_pcl_root}/include")
set(PCL_LIBRARIES "")
set(PCL_LIBRARY_DIRS "")
set(PCL_DEFINITIONS "")
set(PCL_VERSION 1.14.0)
include_directories("${_pcl_root}/include")
