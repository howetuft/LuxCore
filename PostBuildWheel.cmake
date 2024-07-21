# SPDX-FileCopyrightText: 2024 Howetuft
#
# SPDX-License-Identifier: Apache-2.0

# Stats
MESSAGE(STATUS "Add stat command")
add_custom_target(stats ALL COMMAND "echo 'Stats'")
add_dependencies(stats pyluxcore)
add_custom_command(TARGET stats POST_BUILD COMMAND sccache --show-stats)

# Install
install(TARGETS pyluxcore LIBRARY DESTINATION lib)
