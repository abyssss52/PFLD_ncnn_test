// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2017 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include <android/asset_manager_jni.h>
#include <android/bitmap.h>
#include <android/log.h>

#include <jni.h>

#include <string>
#include <vector>

// ncnn
#include "net.h"
#include "benchmark.h"

// calculate euler angle
//#include <opencv.h>
#include <opencv2/core/types.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc/imgproc.hpp>


static ncnn::UnlockedPoolAllocator g_blob_pool_allocator;
static ncnn::PoolAllocator g_workspace_pool_allocator;


static ncnn::Net pfld;

static std::vector<std::string> split_string(const std::string& str, const std::string& delimiter)
{
    std::vector<std::string> strings;

    std::string::size_type pos = 0;
    std::string::size_type prev = 0;
    while ((pos = str.find(delimiter, prev)) != std::string::npos)
    {
        strings.push_back(str.substr(prev, pos - prev));
        prev = pos + 1;
    }

    // To get the last substring (or only, if delimiter is not found)
    strings.push_back(str.substr(prev));

    return strings;
}

extern "C" {

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "PFLDNcnn", "JNI_OnLoad");

    ncnn::create_gpu_instance();

    return JNI_VERSION_1_4;
}

JNIEXPORT void JNI_OnUnload(JavaVM* vm, void* reserved)
{
    __android_log_print(ANDROID_LOG_DEBUG, "PFLDNcnn", "JNI_OnUnload");

    ncnn::destroy_gpu_instance();
}

// public native boolean Init(AssetManager mgr);
JNIEXPORT jboolean JNICALL Java_com_tencent_squeezencnn_PFLDNcnn_Init(JNIEnv* env, jobject thiz, jobject assetManager)
{
    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &g_blob_pool_allocator;
    opt.workspace_allocator = &g_workspace_pool_allocator;

    // use vulkan compute
    if (ncnn::get_gpu_count() != 0)
        opt.use_vulkan_compute = true;

    AAssetManager* mgr = AAssetManager_fromJava(env, assetManager);

    pfld.opt = opt;

    // init param
    {
        int ret = pfld.load_param(mgr, "pfld-sim.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "PFLDNcnn", "load_param_bin failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = pfld.load_model(mgr, "pfld-sim.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_DEBUG, "PFLDNcnn", "load_model failed");
            return JNI_FALSE;
        }
    }

    return JNI_TRUE;
}

// public native String Detect(Bitmap bitmap, boolean use_gpu);
JNIEXPORT jstring JNICALL Java_com_tencent_squeezencnn_PFLDNcnn_Detect(JNIEnv* env, jobject thiz, jobject bitmap, jboolean use_gpu)
{
    if (use_gpu == JNI_TRUE && ncnn::get_gpu_count() == 0)
    {
        return env->NewStringUTF("no vulkan capable gpu");
    }

    double start_time = ncnn::get_current_time();

    AndroidBitmapInfo info;
    AndroidBitmap_getInfo(env, bitmap, &info);
    int width = info.width;
    int height = info.height;
    if (width != 112 || height != 112)
        return NULL;
    if (info.format != ANDROID_BITMAP_FORMAT_RGBA_8888)
        return NULL;

    // ncnn from bitmap
    ncnn::Mat in = ncnn::Mat::from_android_bitmap(env, bitmap, ncnn::Mat::PIXEL_BGR);

    // pfld
    std::vector<float> keypoints;
    {
//        const float mean_vals[3] = {103.f, 117.f, 123.f};
        const float norm_vals[3]  = {1/255.f,1/255.f,1/255.f};
        in.substract_mean_normalize(0, norm_vals);

        ncnn::Extractor ex = pfld.create_extractor();

        ex.set_vulkan_compute(use_gpu);

        ex.input("input_1", in);

        ncnn::Mat out;
        ex.extract("415", out);

        keypoints.resize(out.w);
        for (int j=0; j<out.w; j++)
        {
            keypoints[j] = out[j]*112;
        }
    }


//  Start calculate euler angle
    double K[9] = { 6.5308391993466671e+002, 0.0, 3.1950000000000000e+002, 0.0, 6.5308391993466671e+002, 2.3950000000000000e+002, 0.0, 0.0, 1.0 };
    double D[5] = { 7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000 };
    //fill in cam intrinsics and distortion coefficients
    cv::Mat cam_matrix = cv::Mat(3, 3, 4, K);
    cv::Mat camera_distortion = cv::Mat(5, 1, 4, D);

    //fill in 3D ref points(world coordinates), model referenced from http://aifi.isr.uc.pt/Downloads/OpenGL/glAnthropometric3DModel.cpp
    std::vector<cv::Point3d> object_pts;
    object_pts.push_back(cv::Point3d(6.825897, 6.760612, 4.402142));     //#33 left brow left corner
    object_pts.push_back(cv::Point3d(1.330353, 7.122144, 6.903745));     //#29 left brow right corner
    object_pts.push_back(cv::Point3d(-1.330353, 7.122144, 6.903745));    //#34 right brow left corner
    object_pts.push_back(cv::Point3d(-6.825897, 6.760612, 4.402142));    //#38 right brow right corner
    object_pts.push_back(cv::Point3d(5.311432, 5.485328, 3.987654));     //#13 left eye left corner
    object_pts.push_back(cv::Point3d(1.789930, 5.393625, 4.413414));     //#17 left eye right corner
    object_pts.push_back(cv::Point3d(-1.789930, 5.393625, 4.413414));    //#25 right eye left corner
    object_pts.push_back(cv::Point3d(-5.311432, 5.485328, 3.987654));    //#21 right eye right corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#55 nose left corner
    object_pts.push_back(cv::Point3d(-2.005628, 1.409845, 6.165652));    //#49 nose right corner
    object_pts.push_back(cv::Point3d(2.774015, -2.080775, 5.048531));    //#43 mouth left corner
    object_pts.push_back(cv::Point3d(-2.774015, -2.080775, 5.048531));   //#39 mouth right corner
    object_pts.push_back(cv::Point3d(0.000000, -3.116408, 6.097667));    //#45 mouth central bottom corner
    object_pts.push_back(cv::Point3d(0.000000, -7.415691, 4.070434));    //#6 chin corner



    //2D ref points(image coordinates), referenced from detected facial feature
    std::vector<cv::Point2d> image_pts;
    //fill in 2D ref points, annotations follow WFLW      [33, 38, 50, 46, 60, 64, 68, 72, 55, 59, 76, 82, 85, 16]
    image_pts.push_back(cv::Point2d(keypoints[33*2], keypoints[33*2+1])); //#33 left brow left corner
    image_pts.push_back(cv::Point2d(keypoints[38*2], keypoints[38*2+1])); //#38 left brow right corner
    image_pts.push_back(cv::Point2d(keypoints[50*2], keypoints[50*2+1])); //#50 right brow left corner
    image_pts.push_back(cv::Point2d(keypoints[46*2], keypoints[46*2+1])); //#46 right brow right corner
    image_pts.push_back(cv::Point2d(keypoints[60*2], keypoints[60*2+1])); //#60 left eye left corner
    image_pts.push_back(cv::Point2d(keypoints[64*2], keypoints[64*2+1])); //#64 left eye right corner
    image_pts.push_back(cv::Point2d(keypoints[68*2], keypoints[68*2+1])); //#68 right eye left corner
    image_pts.push_back(cv::Point2d(keypoints[72*2], keypoints[72*2+1])); //#72 right eye right corner
    image_pts.push_back(cv::Point2d(keypoints[55*2], keypoints[55*2+1])); //#55 nose left corner
    image_pts.push_back(cv::Point2d(keypoints[59*2], keypoints[59*2+1])); //#59 nose right corner
    image_pts.push_back(cv::Point2d(keypoints[76*2], keypoints[76*2+1])); //#76 mouth left corner
    image_pts.push_back(cv::Point2d(keypoints[82*2], keypoints[82*2+1])); //#82 mouth right corner
    image_pts.push_back(cv::Point2d(keypoints[85*2], keypoints[85*2+1])); //#85 mouth central bottom corner
    image_pts.push_back(cv::Point2d(keypoints[16*2], keypoints[16*2+1])); //#16 chin corner


    //result
    cv::Mat rotation_vec;                           //3 x 1
    cv::Mat rotation_mat;                           //3 x 3 R
    cv::Mat translation_vec;                        //3 x 1 T
    cv::Mat pose_mat = cv::Mat(3, 4, CV_64FC1);     //3 x 4 R | T
    cv::Mat euler_angle = cv::Mat(3, 1, CV_64FC1);

    //temp buf for decomposeProjectionMatrix()
    cv::Mat out_intrinsics = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_rotation = cv::Mat(3, 3, CV_64FC1);
    cv::Mat out_translation = cv::Mat(3, 1, CV_64FC1);


    //calc pose
    cv::solvePnP(object_pts, image_pts, cam_matrix, camera_distortion, rotation_vec, translation_vec);

    //calc euler angle
    cv::Rodrigues(rotation_vec, rotation_mat);
    cv::hconcat(rotation_mat, translation_vec, pose_mat);
    cv::decomposeProjectionMatrix(pose_mat, out_intrinsics, out_rotation, out_translation, cv::noArray(), cv::noArray(), cv::noArray(), euler_angle);


    // show result
    std::string result_str;
    char tmp[16];
    sprintf(tmp, "%.2f", euler_angle.at<double>(0));
    result_str = result_str + "X: " + tmp + ",";
    sprintf(tmp, "%.2f", euler_angle.at<double>(1));
    result_str = result_str + "Y: " + tmp + ",";
    sprintf(tmp, "%.2f", euler_angle.at<double>(2));
    result_str = result_str + "Z: " + tmp + ",";

    jstring result = env->NewStringUTF(result_str.c_str());

//    double elasped = ncnn::get_current_time() - start_time;
//    __android_log_print(ANDROID_LOG_DEBUG, "MobilenetNcnn", "%.2fms   detect", elasped);

    return result;
}

}
