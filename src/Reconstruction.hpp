#pragma once
#include <filesystem>
#include <vector>
#include <map>
#include <Eigen/Eigen>
#include "FeatureDatabase.hpp"
#include "CostMap.hpp"
#include "KRT.hpp"

class KRT;

/****************************************************************************************
 * @class Reconstruction
 * @brief Data structure to store the reconstruction result from COLMAP or PixelSfM.
 * 
 * This class stores the camera intrinsics, extrinsics, 2D keypoints, 3D points, and
 * 2D-3D correspondences. These data should be loaded from the COLMAP or PixelSfM result.
 * It also stores 2D keypoints' cost maps, which are computed in the optimization process.
 ****************************************************************************************/
class Reconstruction {

public:

	/**
	  * @brief Default constructor.
	  */
	Reconstruction(void) = default;

	/**
	  * @brief Construct the reconstruction from the given path.
	  * @param path 	The path to the reconstruction.
	  */
	Reconstruction(const std::filesystem::path& path) {
		this->load(path);
	}

	/**
	  * @brief Load the reconstruction from the given path.
	  *
	  * @param path 	The path to the reconstruction directory.
	  * @return True if the reconstruction is loaded successfully.
	  */
	bool load(const std::filesystem::path& path);

	/**
	  * @brief Save the reconstruction to the given path.
	  * @param path 	The path to the output directory.
	  * @param krt 		The KRT data. This is used to access the camera names.
	  * @return True if the reconstruction is saved successfully.
	  */
	bool save(const std::filesystem::path& path, const KRT& krt) const;

	/**
	  * @brief Update the 3D point errors.
	  */
	void updatePoint3DErrors(void);

	/**
	  * @brief Compute the cost map for each keypoint.
	  * @param featureDatabase 	The feature database.
	  */
	void computeCostMaps(FeatureDatabase& featureDatabase);

	/**
	  * @brief Get a summary string for the reconstruction result.
	  * @param krt 	The KRT data.
	  * @return The summary string.
	  * 
	  * The KRT data is used to count the number of all cameras and not registered cameras.
	  */
	std::string summary(const KRT& krt) const;

public:

	/**
	  * @struct Keypoint
	  * @brief 2D keypoint. Read from "images.txt".
	  */
	struct Keypoint {
		Eigen::Vector2d pixelPos = Eigen::Vector2d::Zero();
		std::int64_t point3DID = -1LL;
	};

	/**
	  * @struct Track
	  * @brief 2D-3D correspondence. Can be read from "points3D.txt" or computed from "images.txt".
	  * 
	  * In our implementation, we compute the tracks from "images.txt".
	  */
	struct Track {
		std::uint64_t cameraName = 0ULL;
		std::uint64_t keypointID = 0ULL;
	};

	/**
	  * @struct Point3D
	  * @brief 3D point. Read from "points3D.txt".
	  */
	struct Point3D {
		Eigen::Vector3d position = Eigen::Vector3d::Zero();
		Eigen::Vector<std::uint8_t, 3> color = Eigen::Vector<std::uint8_t, 3>::Zero();
		double error = 0.0; ///< Defined using COLMAP's formula.
		std::vector<Track> tracks{};
	};

	/**
	  * @struct CameraData
	  * @brief Camera data. Read from "cameras.txt" and "images.txt".
	  * 
	  * In each frame in multiface dataset, each image has a unique camera. We store both
	  * data from "cameras.txt" (intrinsics) and "images.txt" (extrinsics, 2D keypoints) here.
	  */
	struct CameraData {
		Eigen::Vector3d rotation = Eigen::Vector3d::Zero();
		Eigen::Vector3d translation = Eigen::Vector3d::Zero();
		Eigen::Vector4d intrinsics = Eigen::Vector4d::Zero(); // fx, fy, cx, cy
		std::vector<Keypoint> keypoints{};
	};

	/**
	  * @brief A map storing all camera data.
	  * 
	  * Key: Camera name.
	  * Value: Camera data.
	  */
	std::map<std::uint64_t, CameraData> cameraDataMap{};

	/**
	  * @brief A vector of 3D points.
	  */
	std::vector<Point3D> point3Ds{};

	/**
	  * @brief The cost maps.
	  */
	CostMaps costMaps{};

	/**
	  * @brief The name of the frame.
	  */
	std::uint64_t frameName = 0ULL;

	/**
	  * @brief The number of valid keypoints (keypoint that has a corresponding 3D point).
	  */
	std::size_t numValidKeypoints = 0ULL;

	/**
	  * @brief The number of valid 3D points (3D points that has at least one track).
	  */
	std::size_t numValidPoints3D = 0ULL;

};