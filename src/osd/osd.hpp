#ifndef OSD_HPP__
#define OSD_HPP__
#include "common/object.hpp"
#include <opencv2/opencv.hpp>
#include <unordered_map>
#include <vector>

void osd_detection(cv::Mat &image, const object::DetectionResultArray &detection_results);

void osd_pose(cv::Mat &image, const object::PoseResultArray &pose_results);

void osd_obb(cv::Mat &image, const object::DetectionObbResultArray &detection_obb_results);

void osd_segmentation(cv::Mat &image, const object::SegmentationResultArray &segmentation_results);
// namespace node

#endif // DRAWNODE_HPP__