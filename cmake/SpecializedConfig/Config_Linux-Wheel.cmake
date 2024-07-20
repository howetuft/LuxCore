###########################################################################
#
# Configuration
#
###########################################################################

# Specialization for wheel manylinux_2_28

MESSAGE(STATUS "Using Linux wheel settings")

set(BOOST_LIBRARYDIR "/usr/lib64/boost1.78;/usr/lib64" CACHE PATH "" FORCE)
set(BOOST_INCLUDEDIR "/usr/include/boost1.78;/usr/lib64" CACHE PATH "" FORCE)
set(Boost_NO_SYSTEM_PATHS ON CACHE BOOL "" FORCE)

set(PYTHON_V "36")
set(EMBREE_SEARCH_PATH "/usr/lib64")

SET(CMAKE_INCLUDE_PATH "${LuxRays_SOURCE_DIR}/../target-64-sse2/include;${LuxRays_SOURCE_DIR}/../target-64-sse2")
SET(CMAKE_LIBRARY_PATH "${LuxRays_SOURCE_DIR}/../target-64-sse2/lib;${LuxRays_SOURCE_DIR}/../target-64-sse2")
SET(Blosc_USE_STATIC_LIBS   "ON")

#SET(BUILD_LUXCORE_DLL TRUE)

#SET(CMAKE_BUILD_TYPE "Debug")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
SET(CMAKE_BUILD_TYPE "Release")
