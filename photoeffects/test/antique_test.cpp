#include <gtest/gtest.h>
#include "test_utils.hpp"
#include "photoeffects.hpp"

using namespace cv;

TEST(photoeffects, AntiqueTest) 
{
    Mat srcUCThreeChannels(10, 10, CV_8UC3);
    srcUCThreeChannels = Mat::zeros(10, 10, CV_8UC3);
    Mat textureUCThreeChannels(10, 10, CV_8UC3);
    textureUCThreeChannels = Mat::zeros(10, 10, CV_8UC3);
    Mat dst;
    EXPECT_EQ(0, antique(srcUCThreeChannels, dst, textureUCThreeChannels, 0.5f));
    Mat srcFCThreeChannels(10, 10, CV_32FC3);
    srcFCThreeChannels = Mat::zeros(10, 10, CV_32FC3);
    Mat textureFCThreeChannels(10, 10, CV_32FC3);
    textureFCThreeChannels = Mat::zeros(10, 10, CV_32FC3);
    EXPECT_EQ(0, antique(srcFCThreeChannels, dst, textureFCThreeChannels, 0.5f));
}

TEST(photoeffects, AntiqueInvalidImageFormat)
{
    Mat src(10, 10, CV_8UC1);
    Mat textureNormal(10, 10, CV_8UC3);
    Mat dst;
    EXPECT_ERROR(CV_StsAssert, antique(src, dst, textureNormal, 0.5f));
    Mat srcNormal(10, 10, CV_8UC3);
    EXPECT_ERROR(CV_StsAssert, antique(srcNormal, dst, textureNormal, 0.0f));
    Mat texture(10, 10, CV_8UC1);
    EXPECT_ERROR(CV_StsAssert, antique(srcNormal, dst, texture, 0.4f));
}

TEST(photoeffects, AntiqueRegressionTest)
{
    string input = "./testdata/antique_test.png";
    string texture = "./testdata/antique_texture_test.png";
    string expectedOut = "./testdata/antique_test_result.png";
    Mat src = imread(input, CV_LOAD_IMAGE_COLOR);
    if (src.empty())
    {
        FAIL() << "Can't read " + input + " image";
    }
    Mat txtre = imread(texture, CV_LOAD_IMAGE_COLOR);
    if (txtre.empty())
    {
        FAIL() << "Can't read " + texture + " image";
    }
    Mat expectedDst = imread(expectedOut, CV_LOAD_IMAGE_COLOR);
    if (expectedDst.empty())
    {
        FAIL() << "Can't read" + expectedOut + " image";
    }
    Mat dst;
    EXPECT_EQ(0, antique(src, dst, txtre, 0.9f));
    Mat diff = abs(expectedDst - dst);
    Mat mask = diff.reshape(1) > 1;
    EXPECT_EQ(0, countNonZero(mask));
}