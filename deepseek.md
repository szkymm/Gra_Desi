# 图像识别与栅格数据匹配

嘿，你好deepseek！我正在进行一些工作。

我会与你介绍一下我的项目内容，请你根据我介绍的内容来安排我的两个文件并建立第三个文件作为`main.py`。

## 一、文件与文件夹结构

现在已经有的文件和文件夹结构如下：

```bash
./
|--meta_data
  |--1722
    |--capture
    |--metadata
    |-results
    |-REFLECTANCE_1722.dat
    |-……其他文件
  |--1723
    |--capture
    |--metadata
    |-results
    |-REFLECTANCE_1723.dat
    |-……其他文件
  |--...其他类似文件夹
|--points
|--images
|--image-tag.py
└--obtain-reflectance.py
```

## 二、需要你做的事情
1. 整理两个文件（已经发给你了）`image-tag.py`和`obtain-reflectance.py`，使其得出的结果可以直接合并到新的main.py当中去直接使用。
2. 格式化两个文件的输出内容，修改两个文件输出的目录，使得更加可读可视化。

## 三、第一个回答需要你做的事情
1. 需要你讲image-tag.py文件修改成在这个文件目录下的批量化处理。