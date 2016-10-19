cmake_minimum_required(VERSION 2.8)

project(sampleReciprocalNeon)

add_executable(runReciprocalTest main.cpp)

if (UNIX)
	set(CMAKE_CXX_FLAGS "-mfpu=neon-fp16")
endif()
