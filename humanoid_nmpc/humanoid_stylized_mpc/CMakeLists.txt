cmake_minimum_required(VERSION 3.14)
project(humanoid_stylized_mpc)

if (CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror=return-type)
endif ()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)

find_package(ament_cmake REQUIRED)
# ocs2_core's exported targets reference Boost:: and pinocchio; find them first
# (same as humanoid_common_mpc).
find_package(Boost REQUIRED COMPONENTS system filesystem log log_setup)
find_package(PkgConfig REQUIRED)
pkg_check_modules(pinocchio REQUIRED pinocchio)
find_package(ocs2_core REQUIRED)
find_package(ocs2_oc REQUIRED)
find_package(ocs2_pinocchio_interface REQUIRED)
find_package(humanoid_common_mpc REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(dependencies
        ocs2_core
        ocs2_oc
        ocs2_pinocchio_interface
        humanoid_common_mpc
)

include_directories(
        include
        ${EIGEN3_INCLUDE_DIRS}
        ${pinocchio_INCLUDE_DIRS}
)

link_directories(
        ${pinocchio_LIBRARY_DIRS}
)

add_definitions("-DBOOST_MPL_LIMIT_LIST_SIZE=30")

add_library(${PROJECT_NAME}
        src/StylizedReference.cpp
        src/StylizedReferenceBuilder.cpp
        src/StylizedReferenceBuffer.cpp
        src/StylizedMotionManager.cpp
        src/cost/StylizedTaskSpaceCost.cpp
        src/cost/FootstepPlacementCost.cpp
)
ament_target_dependencies(${PROJECT_NAME} ${dependencies})
target_link_libraries(${PROJECT_NAME} ${pinocchio_LIBRARIES})
target_compile_options(${PROJECT_NAME} PUBLIC ${OCS2_CXX_FLAGS})

#############
## Testing ##
#############

if (BUILD_TESTING)
    find_package(ament_cmake_gtest REQUIRED)
    find_package(humanoid_wb_mpc REQUIRED)
    find_package(ament_index_cpp REQUIRED)

    ament_add_gtest(test_stylized_reference
            test/testStylizedReference.cpp
    )
    target_link_libraries(test_stylized_reference ${PROJECT_NAME})
    ament_target_dependencies(test_stylized_reference ${dependencies})

    ament_add_gtest(test_stylized_reference_buffer
            test/testStylizedReferenceBuffer.cpp
    )
    target_link_libraries(test_stylized_reference_buffer ${PROJECT_NAME})
    ament_target_dependencies(test_stylized_reference_buffer ${dependencies})

    ament_add_gtest(test_stylized_reference_builder
            test/testStylizedReferenceBuilder.cpp
    )
    target_link_libraries(test_stylized_reference_builder ${PROJECT_NAME})
    ament_target_dependencies(test_stylized_reference_builder ${dependencies} humanoid_wb_mpc ament_index_cpp)
endif ()

#############
## Install ##
#############

install(TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION lib
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin
)

install(DIRECTORY include/${PROJECT_NAME}
        DESTINATION include/
)

ament_export_include_directories(include)
ament_export_libraries(${PROJECT_NAME})
ament_export_dependencies(${dependencies})

ament_package()
