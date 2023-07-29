# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     makefile.py
   Description :
   Author :       ASUS
   date：          2023-07-28
-------------------------------------------------
   Change Activity:
                   2023-07-28:
-------------------------------------------------
"""
__author__ = 'ASUS'
import os

# input directory and output directory.
srcDir = "./src" # source code
outDir = "./build/classes" # output classes

# java files to be compiled, note that:
# "hao" corresponds to *.java files stored in ./src/hao
# "hao.utils" corresponds to *.java file stored in ./src/hao/utils
srcs = ["hao", "hao.utils"]

# Format inputs. For example,
# hao -> ./src/hao/*.java
# hao.utils -> ./src/hao/*.java
classpath = srcDir
for i in range(len(srcs)):
    srcs[i] = os.path.join(os.path.join(srcDir, srcs[i].replace(".", "/"), "*.java"))

# create the compiling command
compileCmd = "javac -d {0} -cp {1} {2}".format(outDir, classpath, " ".join(srcs))

# compile
print(compileCmd)
os.system(compileCmd)

jarCmd = "jar cvf ./jars/hao.jar -C ./build/classes ."
os.system(jarCmd)
print("Compiled.")
