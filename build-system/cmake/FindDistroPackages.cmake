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

# This script is aimed at finding packages for the build from local sources.
# This script responds to the request of package managers who wish to build
# LuxCore against their own version of dependencies.
#
# However, please note that:
# - we DO NOT RECOMMEND this build method
# - we will therefore not provide any support for this build method
# - this method is provided as-is and you use it at your own risk
#
# The right method is implemented in FindConanPackages.cmake
# If you persist in using this method, however, you can activate it by
# setting LUX_USE_DISTRO_PACKAGES to ON. This script will be executed
# instead. You can keep it as-is or replace all or part of its content
# according to your needs.

find_package(TBB REQUIRED)
#find_package(minizip REQUIRED)
find_package(spdlog REQUIRED)
find_package(OpenImageIO REQUIRED)
find_package(PNG REQUIRED)
find_package(OpenColorIO REQUIRED)
find_package(OpenEXR REQUIRED)
find_package(embree REQUIRED)
#find_package(blosc2 REQUIRED)
find_package(OpenImageDenoise REQUIRED)
find_package(Boost REQUIRED)
find_package(OpenVDB REQUIRED)
find_package(Imath REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenSubdiv REQUIRED)
find_package(nlohmann_json REQUIRED)
#find_package(robin_hood REQUIRED)
find_package(OpenMP REQUIRED)
find_package(OpenJPEG REQUIRED)
find_package(openjph REQUIRED)
if (WIN32 OR LINUX)
  find_package(CUDA REQUIRED)
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

MESSAGE(STATUS "OIDN=${OpenImageDenoise_LIBRARIES}")
set(LUX_OIDN_DEVICE_CPU ${OpenImageDenoise_LIBRARIES} CACHE STRING "")
