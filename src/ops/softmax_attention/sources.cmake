target_sources(ninfer_ops PRIVATE
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/causal_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_fp8.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_sm70.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_k8v4.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/packed_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/packed/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/context_softmax_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/dense/context/launch.cu"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/sliding_window_attention.cpp"
  "${CMAKE_CURRENT_LIST_DIR}/sliding_window/launch.cu"
)

# The fast NVFP4 prompt kernel runs QK on block-scaled FP4 Tensor Cores (kind::mxf4nvf4), which
# only Blackwell has.
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "120a")
  target_sources(ninfer_ops PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/prompt_nvfp4_fast.cu")
endif()

# The INT8-family small-T partial launch, one generated unit per query width, head geometry, cache
# input and storage: the long ptxas runs stay parallel, and each storage's kernels form their own
# module, so a process loads only its cache's kernels on a width's first use (lazy loading parses
# a whole module).
set(_small_t_i8_storages
  Int8Group64
  RotatedInt8KeyInt4ValueGroup64
  RotatedLloyd4KeyInt4Value
  RotatedInt4KeyInt4ValueE8
  RotatedE8RootKeyInt4Value)
foreach(_geometry IN ITEMS H24 H16)
  if(_geometry STREQUAL "H24")
    set(_geometry_type CausalD256H24Kv4)
    set(_widths 1 2 3 4 5 6 7 8)
  else()
    set(_geometry_type CausalD256H16Kv2)
    set(_widths 1 2 3 4 5 6)
  endif()
  foreach(WIDTH IN LISTS _widths)
    foreach(_input IN ITEMS append cached)
      if(_input STREQUAL "append")
        set(INPUT CausalAppendInput)
      else()
        set(INPUT CausalCachedInput)
      endif()
      foreach(STORAGE IN LISTS _small_t_i8_storages)
        set(GEOMETRY ${_geometry_type})
        set(_unit "${CMAKE_CURRENT_BINARY_DIR}/softmax_attention/small_t_i8/")
        string(APPEND _unit "w${WIDTH}_${_geometry}_${_input}_${STORAGE}.cu")
        configure_file("${CMAKE_CURRENT_LIST_DIR}/dense/causal_cache/small_t_i8_unit.cu.in"
                       "${_unit}" @ONLY)
        target_sources(ninfer_ops PRIVATE "${_unit}")
      endforeach()
    endforeach()
  endforeach()
endforeach()
