- 确认 NPP 相关文件

  - ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\
    ```

  - ```
    include\npp.h
    include\nppi.h
    
    lib\x64\nppc.lib
    lib\x64\nppial.lib
    lib\x64\nppicc.lib
    lib\x64\nppif.lib
    lib\x64\nppig.lib
    lib\x64\nppim.lib
    lib\x64\nppist.lib
    lib\x64\nppisu.lib
    ```

- Visual Studio 配置

  - ```
    项目
    → 属性
    → 配置属性
    → 链接器
    → 常规
    → 附加库目录
    ```

  - ```
    C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\lib\x64
    ```

  - ```
    nppiLabelMarkersUF_8u32u_C1R(src.ptr<Npp8u>(), src.step, (Npp32u*)labels.ptr<int>(), labels.step, roi, nppiNormInf, pBuffer);
    ```

  - ```
    ratioMin
    <= ratio
    <= ratioMax
    ```

  - 