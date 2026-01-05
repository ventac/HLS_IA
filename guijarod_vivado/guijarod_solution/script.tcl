############################################################
## This file is generated automatically by Vivado HLS.
## Please DO NOT edit it.
## Copyright (C) 1986-2018 Xilinx, Inc. All Rights Reserved.
############################################################
open_project guijarod_vivado
set_top lenet_cnn_fixed
add_files FIXED/FIXED/conv_fixed.c
add_files FIXED/FIXED/fc_fixed.c
add_files FIXED/FIXED/fixed_point.h
add_files FIXED/FIXED/lenet_cnn_fixed
add_files FIXED/FIXED/lenet_cnn_fixed.c
add_files FIXED/FIXED/lenet_cnn_fixed.h
add_files FIXED/FIXED/pool_fixed.c
add_files FIXED/FIXED/utils.c
add_files FIXED/FIXED/weights.h
add_files FIXED/FIXED/conv_fixed.c
add_files FIXED/FIXED/fc_fixed.c
add_files FIXED/FIXED/fixed_point.h
add_files FIXED/FIXED/lenet_cnn_fixed
add_files FIXED/FIXED/lenet_cnn_fixed.c
add_files FIXED/FIXED/lenet_cnn_fixed.h
add_files FIXED/FIXED/pool_fixed.c
add_files FIXED/FIXED/utils.c
add_files FIXED/FIXED/weights.h
open_solution "guijarod_solution"
set_part {xc7z020clg484-1} -tool vivado
create_clock -period 10 -name default
#source "./guijarod_vivado/guijarod_solution/directives.tcl"
#csim_design
csynth_design
#cosim_design
export_design -format ip_catalog
