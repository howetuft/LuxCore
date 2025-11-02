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

# This script ensures the expected and the actual versions of dependencies
# match.
#
# Version number is to be set by external caller (typically, toolchain)
# by setting LUXCOREDEPS_VERSION.
# Otherwise (like in a plain make build), we'll fallback to build-settings
# default version

if (${LUX_USE_DISTRO_PACKAGES})
  MESSAGE(STATUS "Building against distro packages (not recommended)")
  return()
endif()

# Check whether LUXCOREDEPS_VERSION has been provided
if("${LUXCOREDEPS_VERSION}" STREQUAL "")
  message(
    FATAL_ERROR
    "Error: LUXCOREDEPS_VERSION (aka LuxCore dependencies version) is not set.
Are you building LuxCore in the recommended way (make deps && make)?"
  )

endif()
FILE(READ "${PROJECT_SOURCE_DIR}/build-system/build-settings.json" settings_json)
STRING(JSON _DEFAULT_VERSION_MAJOR GET "${settings_json}" DefaultVersion major)
STRING(JSON _DEFAULT_VERSION_MINOR GET "${settings_json}" DefaultVersion minor)
STRING(JSON _DEFAULT_VERSION_PATCH GET "${settings_json}" DefaultVersion patch)
SET(
  _DEFAULT_VERSION_BASE
  "${_DEFAULT_VERSION_MAJOR}.${_DEFAULT_VERSION_MINOR}.${_DEFAULT_VERSION_PATCH}"
)
STRING(JSON _DEFAULT_VERSION_PRERELEASE GET "${settings_json}" DefaultVersion prerelease)
if (NOT "${_DEFAULT_VERSION_PRERELEASE}" STREQUAL "")
  SET(_DEFAULT_LUXCORE_VERSION "${_DEFAULT_VERSION_BASE}-${_DEFAULT_VERSION_PRERELEASE}")
else()
  SET(_DEFAULT_LUXCORE_VERSION "${_DEFAULT_VERSION_BASE}")
endif()

SET(LUXCORE_VERSION "${_DEFAULT_LUXCORE_VERSION}" CACHE STRING "LuxCore version")


# Dependencies check
STRING(JSON REQUIRED_LUXCOREDEPS_VERSION GET "${settings_json}" Dependencies release)
MESSAGE(STATUS "LuxCore Dependencies - Required version: \"${REQUIRED_LUXCOREDEPS_VERSION}\"")
IF(NOT ${REQUIRED_LUXCOREDEPS_VERSION} VERSION_EQUAL "${LUXCOREDEPS_VERSION}")
  MESSAGE(SEND_ERROR "LuxCore Dependencies - Provided version: \"${LUXCOREDEPS_VERSION}\"")
  MESSAGE(FATAL_ERROR "\
LuxCore Dependencies - Provided version does not match required version!\n\
This may result in build errors or undefined behavior.\n\
You should try to update dependencies ('make deps').\
  ")
ELSE()
  MESSAGE(STATUS "LuxCore Dependencies - Provided version: ${LUXCOREDEPS_VERSION} - OK")
ENDIF()

