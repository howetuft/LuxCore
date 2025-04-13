# SPDX-FileCopyrightText: 2024-2025 Authors (see AUTHORS.txt)
#
# SPDX-License-Identifier: Apache-2.0



if (NOT TARGET doc)
  add_custom_target(doc)
endif()

macro(generate_doc target)
  if(DOXYGEN_FOUND)

    # Generate doxygen.template
    set(DOXYGEN_${target}_TEMPLATE ${PROJECT_SOURCE_DIR}/doxygen/${target}.template)
    configure_file("${DOXYGEN_${target}_TEMPLATE}.in" "${DOXYGEN_${target}_TEMPLATE}")

    set(DOXYGEN_${target}_INPUT ${CMAKE_BINARY_DIR}/doc/doxygen-${target}.conf)
    set(DOXYGEN_${target}_OUTPUT_DIR ${CMAKE_BINARY_DIR}/doc/${target})
    set(DOXYGEN_${target}_OUTPUT ${DOXYGEN_${target}_OUTPUT_DIR}/html/index.html)

    message(STATUS "Doxygen ${target} output: " ${DOXYGEN_${target}_OUTPUT})

    if(DOXYGEN_DOT_FOUND)
        message(STATUS "Found dot")
        set(DOXYGEN_DOT_CONF "HAVE_DOT = YES")
    endif(DOXYGEN_DOT_FOUND)

    add_custom_command(
      OUTPUT ${DOXYGEN_${target}_OUTPUT}
      # Creating custom doxygen-xxx.conf
      COMMAND mkdir -p ${DOXYGEN_${target}_OUTPUT_DIR}
      COMMAND cp ${DOXYGEN_${target}_TEMPLATE} ${DOXYGEN_${target}_INPUT}
      COMMAND echo "INPUT = " ${PROJECT_SOURCE_DIR}/include/${target}  ${PROJECT_SOURCE_DIR}/src/${target} >> ${DOXYGEN_${target}_INPUT}
      COMMAND echo "OUTPUT_DIRECTORY = " ${DOXYGEN_${target}_OUTPUT_DIR} >> ${DOXYGEN_${target}_INPUT}
      COMMAND echo ${DOXYGEN_DOT_CONF} >> ${DOXYGEN_${target}_INPUT}
      # Launch doxygen
      COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_${target}_INPUT}
      DEPENDS ${DOXYGEN_${target}_TEMPLATE}
      WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    )


    add_custom_target(doc-${target} DEPENDS ${DOXYGEN_${target}_OUTPUT})
    add_dependencies(doc doc-${target})
    message(STATUS "Target 'doc-${target}' declared for ${target} documentation")
  else()
    message(AUTHOR_WARNING "Doxygen not found, could not create ${target} documentation")
  endif(DOXYGEN_FOUND)

endmacro()
