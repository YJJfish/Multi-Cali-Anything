#pragma once
#include <vector>
#include <unordered_map>
#include <utility>
#include <cstdint>
#include <memory>
#include <Eigen/Eigen>
#include "utils.hpp"
#include "defines.hpp"

class CostMap;
class CostMaps;
class Reconstruction;
class FeatureDatabase;

/****************************************************************************************
 * @class CostMap
 * @brief Data structure to reference a cost map's data in a flattened cost maps storage.
 * 
 * The class is designed to be transient, and should not be instantiated directly.
 ****************************************************************************************/
class CostMap {

public:

	std::uint64_t cameraName(void) const { return this->_cameraName; }
	std::uint64_t keypointID(void) const { return this->_keypointID; }
	std::uint64_t cornerX(void) const { return this->_cornerX; }
	std::uint64_t cornerY(void) const { return this->_cornerY; }
	bool valid(void) const { return this->_pRawData != nullptr; }
	
	/**
	  * @brief Perform bicubic interpolation and return the 3-dimensional cost residual.
	  * @param x 	The x coordinate.
	  * @param y 	The y coordinate.
	  * @return The 3-dimensional cost residual.
	  * 
	  * This function is implemented as a template function to support ceres::AutoDiffCostFunction.
	  */
	template <class T>
	Eigen::Vector<T, 3> operator()(T x, T y) const {
		return bicubicInterpolation<T, double, COST_MAP_CHANNEL, true>(
			this->_pRawData,
			COST_MAP_WIDTH,
			COST_MAP_HEIGHT,
			x - static_cast<T>(this->_cornerX),
			y - static_cast<T>(this->_cornerY)
		);
	}

private:

	const double* _pRawData = nullptr;
	std::uint64_t _cameraName = 0ULL;
	std::uint64_t _keypointID = 0ULL;
	std::uint64_t _costMapID = 0ULL;
	std::uint64_t _cornerX = 0ULL;
	std::uint64_t _cornerY = 0ULL;

	/**
	  * @brief Constructor.
	  */
	CostMap(
		const double* pRawData,
		std::uint64_t cameraName,
		std::uint64_t keypointID,
		std::uint64_t costMapID,
		std::uint64_t cornerX,
		std::uint64_t cornerY
	) :
		_pRawData(pRawData),
		_cameraName(cameraName),
		_keypointID(keypointID),
		_costMapID(costMapID),
		_cornerX(cornerX),
		_cornerY(cornerY)
	{}

	friend class CostMaps;

};

/****************************************************************************************
 * @class CostMaps
 * @brief Data structure to store the cost maps of 2D keypoints for a given frame.
 * 
 * In bundle adjustment, we need to compute the Cauchy loss:
 * 			`|| denseFeature(project(3d point)) - referenceFeature ||_gamma`.
 * The reference feature is computed and fixed for each 3D point throughout the optimization.
 * denseFeature() involves bicubic interpolation of a 128-dimensional feature map, which is
 * memory-intensive.
 * Therefore, we compute `|| denseFeature(x) - referenceFeature ||_gamma` for every pixel x,
 * and store the results in a cost map.
 * For better precision, we also compute the gradient of the above loss.
 * The resulting cost map is three dimensional: loss, gradientX, gradientY.
 * We flatten all cost maps of a frame into a single data storage, so that we can process
 * them in CUDA kernels more efficiently.
 ****************************************************************************************/
class CostMaps {

public:

	/**
	  * @brief Default constructor.
	  */
	CostMaps(void) = default;

	/**
	  * @brief Get the cost map given the camera name and keypoint ID.
	  * @param cameraName 	The camera name.
	  * @param keypointID 	The keypoint ID.
	  * @return A CostMap object.
	  */
	CostMap getCostMap(std::uint64_t cameraName, std::uint64_t keypointID) const {
		auto iter = this->_cameraNameKeypointID2CostMapID.find(cameraName);
		if (iter == this->_cameraNameKeypointID2CostMapID.end())
			return CostMap(nullptr, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
		std::int64_t costMapID = iter->second[keypointID];
		if (costMapID == -1LL)
			return CostMap(nullptr, 0ULL, 0ULL, 0ULL, 0ULL, 0ULL);
		return CostMap(
			this->_costMaps.get() + costMapID * COST_MAP_HEIGHT * COST_MAP_WIDTH * COST_MAP_CHANNEL,
			cameraName,
			keypointID,
			costMapID,
			this->_cornerXs[costMapID],
			this->_cornerYs[costMapID]
		);
	}

	/**
	  * @brief Compute the cost maps.
	  * @param reconstruction 	The reconstruction from COLMAP or PixelSfM.
	  * @param featureDatabase 	The feature database.
	  */
	void computeCostMaps(const Reconstruction& reconstruction, const FeatureDatabase& featureDatabase);

private:

	/**
	  * @brief Camera name & keypoint ID to cost ID.
	  * 
	  * Not all keypoints have a corresponding 3D trangulated point.
	  * We only compute and store the cost maps for keypoints with
	  * a corresponding 3D point.
	  * This data structure maps the camera name and keypoint ID to
	  * the cost map ID in the flattened cost maps.
	  * The cost map ID is -1 if the keypoint does not have a cost map.
	  */
	std::unordered_map<std::uint64_t, std::vector<std::int64_t>> _cameraNameKeypointID2CostMapID{};

	/**
	  * @brief Cost map ID to camera name & keypoint ID.
	  * 
	  * This vector maps the cost map ID to the camera name and keypoint ID.
	  * This is the inverse of _cameraNameKeypointID2CostMapID.
	  */
	std::vector<std::pair<std::uint64_t, std::uint64_t>> _costMapID2CameraNameKeypointID{};

	/**
	  * @brief Flattened cost maps.
	  * 
	  * The cost maps are stored in a flattened format.
	  * NumCostMaps x Height x Width x 3.
	  */
	std::shared_ptr<double[]> _costMaps{};

	std::vector<std::uint64_t> _cornerXs{};
	std::vector<std::uint64_t> _cornerYs{};

	/**
	  * @brief Compute the keypoint features in CUDA.
	  * @param numValidKeypoints 				The number of valid keypoints. Also the number of cost maps.
	  * @param deviceFeaturePatches 			The feature patches. There are numValidKeypoints feature patches.
	  * @param deviceInterpolationCoordinates 	The interpolation coordinates. They should be keypoint coordinates - corner coordinates.
	  * @param deviceKeypointFeatures 			The output keypoint features. There are numValidKeypoints keypoint features.
	  */
	static void _computeKeypointFeaturesKernelDelegate(
		const std::uint64_t numValidKeypoints,
		const Eigen::half* deviceFeaturePatches,
		const Eigen::Vector2d* deviceInterpolationCoordinates,
		Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures
	);

	/**
	  * @brief Compute the reference features in CUDA.
	  * @param numPoint3Ds 					The number of 3D points.
	  * @param deviceTracks 				The tracks. Each track is a keypoint ID within 0 to numValidKeypoints-1.
	  * @param deviceTrackOffsets 			The track offsets. trackOffsets[i] is the index of the first track of 3D point i.
	  * @param deviceTrackLengths 			The track lengths. trackLengths[i] is the number of tracks of 3D point i.
	  * @param deviceKeypointFeatures 		The keypoint features. There are numValidKeypoints keypoint features.
	  * @param deviceReferenceFeatures 		The output reference features. There are numPoint3Ds reference features.
	  */
	static void _computeReferenceFeaturesKernelDelegate(
		const std::uint64_t numPoint3Ds,
		const std::uint64_t* deviceTracks,
		const std::uint64_t* deviceTrackOffsets,
		const std::uint64_t* deviceTrackLengths,
		const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures,
		Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures
	);

	/**
	  * @brief Compute the cost maps in CUDA.
	  * @param numCostMaps 					The number of cost maps. Also the number of valid keypoints.
	  * @param deviceFeaturePatches 		The feature patches. There are numCostMaps feature patches.
	  * @param deviceReferenceFeatures 		The reference features. There are #3D points reference features.
	  * @param deviceCostMapID2Point3DID 	The index array. Cost map ID to 3D point ID.
	  * @param deviceCostMaps 				The output cost maps. There are numCostMaps cost maps.
	  */
	static void _computeCostMapsKernelDelegate(
		std::uint64_t numCostMaps,
		const Eigen::half* deviceFeaturePatches,
		const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures,
		const std::uint64_t* deviceCostMapID2Point3DID,
		double* deviceCostMaps
	);

};