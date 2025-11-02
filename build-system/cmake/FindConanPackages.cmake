################################################################################
# Copyright 1998-2025 by Authors (see AUTHORS.txt)
#
#   This file is part of LuxCoreRender.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
################################################################################

# This script finds all packages required for the build, from Conan source.
# This is the nominal and STRONGLY RECOMMENDED WAY to provide packages to the
# build process.
#
# (for distro package managers, however:
# see LUX_USE_DISTRO_PACKAGES and FindDistroPackages.cmake for an alternative)

find_package(TBB REQUIRED)
find_package(minizip REQUIRED)
find_package(spdlog REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenColorIO REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(embree REQUIRED)
find_package(c-blosc REQUIRED)
find_package(oidn REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(Imath REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenSubdiv REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(robin_hood REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenJPEG REQUIRED)
find_package(openjph REQUIRED)
if (WIN32 OR LINUX)
  find_package(nvrtc REQUIRED)
endif()

# Bison/Flex
find_package(BISON REQUIRED)
find_package(FLEX REQUIRED)

# Additional dependencies for pyluxcore
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Additional dependencies for luxcoreui
find_package(imgui)
find_package(glfw3)
find_package(imguifiledialog)

# Documentation
find_package(Doxygen)
