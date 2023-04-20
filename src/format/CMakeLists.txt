set(FORMAT_SOURCES
	format.cpp
	format.hpp
	format_cxx.hpp
	format_cxx.cpp
	format_csr.cpp
	format_csr.hpp
	format_cvs.cpp
	format_cvs.hpp
	format_sr_nm.hpp
	format_sr_nm.cpp)

set_source_files_properties(${FORMAT_SOURCES} PROPERTIES LANGUAGE CUDA)

add_library(formats STATIC
	${FORMAT_SOURCES})


#target_link_libraries(formats
#	PUBLIC Python::Python
#)

#target_include_directories(formats
#	PRIVATE ${PYTHON_INCLUDE_DIRS})

#pybind11_add_module(format "format_sr_nm.cpp" "format.cpp")

