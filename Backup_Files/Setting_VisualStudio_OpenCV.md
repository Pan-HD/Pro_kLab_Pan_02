- P1 - 環境整備

  - Visual Studio 2022

    - VisualStudioSetup.exe
      - 修改 -> 单个组件
        - MSVC v143 - VS 2022 C++ x64/x86 生成工具 (v14.38-17.8)(不受支持)

  - CMake-4.3.2

  - CUDA-12.3.0

  - opencv-4.10.0

  - opencv_contrib-4.10.0

  - 新建 build 目录

    - ```
      D:\Program_Files\opencv
      ```

    - ↑ opencv-4.10.0・opencv_contrib-4.10.0 同层级

- P2

  - [参考方案-CSDN](https://blog.csdn.net/m0_71790606/article/details/145847363?ops_request_misc=&request_id=&biz_id=102&utm_term=Visual%20Studio%20opencv%20CUDA&utm_medium=distribute.pc_search_result.none-task-blog-2~all~sobaiduweb~default-1-145847363.142%5Ev102%5Epc_search_result_base9&spm=1018.2226.3001.4187)

  - 01 - Configure
  - 02 - Configure
    - BUILD_opencv_world
      - on
    - OPENCV_EXTRA_MODULES_PATH
      - D:/Program_Files/opencv/opencv_contrib-4.10.0/modules
    - BUILD_CUDA_STUBS
      - on
    - WITH_CUDA
      - on
    - ENABLE_FAST_MATH
      - on
    - OPENCV_ENABLE_NONFREE
      - on
    - OpenCV_GENERATE_SETUPVARS
      - off
    - BUILD_opencv_python3, BUILD_opencv_python_bindings_generator, BUILD_opencv_python_tests
      - off
    - BUILD_JAVA, BUILD_opencv_java_bindings_generator
      - off
    - BUILD_opencv_js_bindings_generator
      - off
    - BUILD_PERF_TESTS, BUILD_TESTS
      - off
  - 03 - Configure
    - BUILD_opencv_xfeatures2d
      - off
    - CUDA_FAST_MATH
      - on
    - CUDA_ARCH_BIN
      - 7.5
  - 04 - Generate
  - VISUAL STUDIO 
    - CMake Targets
      - ALL_BUILD　→　生成
      - INSTALL　→　生成

- P3

  - 01 - 视图 - 其他窗口 - 属性管理器

  - 02 - release x64 - 添加新项目属性表 (已存在即直接编辑)

    - `Microsoft.Cpp.x64.user`

  - 03 - Microsoft.Cpp.x64.user - 属性

    - VC++

      - 包含目录

        - ```
          D:\Program_Files\opencv\build\install\include
          D:\Program_Files\opencv\build\install\include\opencv2
          C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include
          ```
    
      - 库目录
    
        - ```
          D:\Program_Files\opencv\build\install\x64\vc17\lib
          ```
    
    - 链接器 - 输入 - 附加依赖项
    
      - ```
        opencv_world4100.lib
        nppc.lib
        nppial.lib
        nppicc.lib
        nppif.lib
        nppig.lib
        nppim.lib
        nppist.lib
        nppisu.lib
        nppitc.lib
        cudart.lib
        ```
        
      - ```
        依赖项确认地址：
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\include\
        C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3\lib\x64\
        ```
    
  - 04 - OpenMP 支持
  
    - C/C++　→　语言　→　OpenMP支持
    
      - ```
        是（/openmp）
        ```
    
  - 05 ( .dll文件相关报错解决 )
  
    - ```
      filename: opencv_world4100.dll
      address: D:\Program_Files\opencv\build\install\x64\vc17\bin
      ```
  
    - ```
      把 opencv_world4100.dll 复制到 Pro_Graph_kLab.exe 所在的目录: 
      F:\ComputerScience\OpenCV_VS_Pro\Pro_Graph_kLab\x64\Release
      ```
    
  - 06 - .cu 文件配置
  
    - ```
      解决资源管理器
        Pro_KLab_Pan_Main 
          (right click) 生成依赖项 -> 生成自定义
            CUDA xx -> √
          源文件
            (right click) 添加 -> 现有项
              cuda_cc_filter_fixed_01.cu
                (right click) 属性 -> 常规 -> 项类型 -> CUDA C/C++ (To Confirm)
      ```
  
  
