cmake_minimum_required(VERSION 3.10)
project(OpenCVHalconCanny)

set(CMAKE_CXX_STANDARD 17)

# Define path to dependencies (2 levels up from tools/OpenCVHalconCanny)
# This assumes the project is located at E:/VSProgram/YuxinVisionMeasure/tools/OpenCVHalconCanny
get_filename_component(ROOT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../" ABSOLUTE)
set(DEP_DIR "${ROOT_DIR}/dependency")

message(STATUS "Dependency Directory: ${DEP_DIR}")

# Qt Configuration
set(CMAKE_PREFIX_PATH "D:/Qt/6.8.3/msvc2022_64")
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 COMPONENTS Core Gui Widgets REQUIRED)

add_executable(OpenCVHalconCanny 
    main.cpp
    MainWindow.h
    MainWindow.cpp
)

# Halcon Include Paths
# Adapting from original project paths
target_include_directories(OpenCVHalconCanny PRIVATE
    "D:/Halcon/include"
    "D:/Halcon/include/halconcpp"
)

# Link Halcon
target_link_libraries(OpenCVHalconCanny PRIVATE
    "${DEP_DIR}/halcon.lib"
    "${DEP_DIR}/halconcpp.lib"
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
)

# OpenCV Configuration
find_package(OpenCV QUIET)

if(OpenCV_FOUND)
    message(STATUS "OpenCV found via find_package: ${OpenCV_VERSION}")
    target_include_directories(OpenCVHalconCanny PRIVATE ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(OpenCVHalconCanny PRIVATE ${OpenCV_LIBS})
    target_compile_definitions(OpenCVHalconCanny PRIVATE HAVE_OPENCV=1)
else()
    message(STATUS "OpenCV not found via find_package, attempting to use local dependencies")
    
    # Add include paths matching the original project structure
    # Note: If these directories do not contain headers, compilation of OpenCV code will fail.
    # It seems headers might be in dependency/Includes or missing. 
    # Since the user has OpenCV installed elsewhere or inside dependency, we need to find where opencv2 folder is.
    # Assuming for now the user has headers in dependency/opencv/include or similar if not in dependency directly.
    # But based on LS results, they are NOT in dependency root or Includes.
    # We will add a fallback to a common install path if possible or rely on user to provide it.
    # HOWEVER, the screenshot shows HAVE_OPENCV not defined, meaning check failed.
    # We will force it if we can find the headers.
    
    # Attempt to find in standard locations or assume a path if known. 
    # If the user provided libs exist (opencv_world4120.lib), headers MUST be somewhere.
    # Let's try to look in the parent directory of dependency or common paths.
    
    # Since we cannot find headers in dependency/, and find_package failed, 
    # we will disable OpenCV backend unless we find them. 
    # BUT the user wants to fix the bug.
    # We will hardcode a common path or ask. 
    # Actually, let's assume they might be in dependency/include/opencv2 if it existed?
    # The previous ls showed only Includes folder with MvCameraControl stuff.
    # It implies OpenCV headers are MISSING from dependency folder!
    
    # We will TRY to find_package again with more hints or use the one from D:/opencv if it exists.
    # Let's add D:/opencv/build to CMAKE_PREFIX_PATH as a common location.
    
    list(APPEND CMAKE_PREFIX_PATH "D:/opencv/build")
    find_package(OpenCV QUIET)
    
    if(OpenCV_FOUND)
         message(STATUS "OpenCV found via second attempt: ${OpenCV_VERSION}")
         target_include_directories(OpenCVHalconCanny PRIVATE ${OpenCV_INCLUDE_DIRS})
         target_link_libraries(OpenCVHalconCanny PRIVATE ${OpenCV_LIBS})
         target_compile_definitions(OpenCVHalconCanny PRIVATE HAVE_OPENCV=1)
    else()
        # If still not found, we check if there is an "include" folder in dependency that we missed?
        # LS showed "Includes" but it had Hikrobot stuff.
        # It seems the repo is missing OpenCV headers in the dependency folder.
        # We will define HAVE_OPENCV=0 to avoid the error message being "undefined", 
        # but the UI will show "unavailable".
        # To FIX it, we need to point to a valid OpenCV install.
        # I will assume D:/opencv/build/include exists and add it.
        
        # Manual fallback for known common paths
         set(OPENCV_ROOT "D:/openCV/opencv/build")
         if(EXISTS "${OPENCV_ROOT}/include/opencv2/opencv.hpp")
              target_include_directories(OpenCVHalconCanny PRIVATE "${OPENCV_ROOT}/include")
              target_link_libraries(OpenCVHalconCanny PRIVATE 
                 "${OPENCV_ROOT}/x64/vc16/lib/opencv_world4120.lib"
              )
              message(STATUS "Forcing OpenCV headers from ${OPENCV_ROOT}/include")
              target_compile_definitions(OpenCVHalconCanny PRIVATE HAVE_OPENCV=1)
         elseif(EXISTS "C:/opencv/build/include/opencv2/opencv.hpp")
             target_include_directories(OpenCVHalconCanny PRIVATE "C:/opencv/build/include")
             target_link_libraries(OpenCVHalconCanny PRIVATE 
                "C:/opencv/build/x64/vc15/lib/opencv_world4120.lib"
             )
             message(STATUS "Forcing OpenCV headers from C:/opencv/build/include")
             target_compile_definitions(OpenCVHalconCanny PRIVATE HAVE_OPENCV=1)
        else()
             # If we really can't find headers, we might need to assume they are in the dep dir 
             # even if we didn't see them (maybe ls was incomplete or I missed something).
             # But I can't compile without headers.
             # Let's try one more aggressive search or just fail gracefully?
             # No, user said "Solve the problem".
             # I will assume the user has headers in the same folder as libs but maybe named differently?
             # No, that's unlikely.
             
             # Last resort: If the user has installed OpenCV via pip or other means, it's hard to find.
             # I'll assume D:/opencv is the place.
             message(WARNING "Could NOT find OpenCV headers. OpenCV backend will be unavailable.")
        endif()
    endif()
endif()
