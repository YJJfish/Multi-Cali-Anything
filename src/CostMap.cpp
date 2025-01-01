#include "CostMap.hpp"
#include "Reconstruction.hpp"
#include "FeatureDatabase.hpp"
#include <ceres/ceres.h>
#include <cuda_runtime.h>

void CostMaps::computeCostMaps(const Reconstruction& reconstruction, const FeatureDatabase& featureDatabase) {
	// Clear the existing data.
	this->_cameraNameKeypointID2CostMapID.clear();
	this->_costMapID2CameraNameKeypointID.clear();
	this->_costMaps.reset();
	this->_cornerXs.clear();
	this->_cornerYs.clear();
	// Assign cost map IDs to keypoints.
	std::uint64_t costMapIDCounter = 0ULL;
	this->_costMapID2CameraNameKeypointID.resize(reconstruction.numValidKeypoints);
	for (const auto& cameraDataEntry : reconstruction.cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const auto& cameraData = cameraDataEntry.second;
		auto& keypointID2CostMapID = this->_cameraNameKeypointID2CostMapID.insert(
			std::make_pair(
				cameraName,
				std::vector<std::int64_t>(cameraData.keypoints.size(), -1LL)
			)
		).first->second;
		for (std::uint64_t keypointID = 0ULL; keypointID < cameraData.keypoints.size(); ++keypointID) {
			const auto& keypoint = cameraData.keypoints[keypointID];
			if (keypoint.point3DID == -1LL)
				continue;
			keypointID2CostMapID[keypointID] = static_cast<std::int64_t>(costMapIDCounter);
			this->_costMapID2CameraNameKeypointID[costMapIDCounter] = std::make_pair(cameraName, keypointID);
			++costMapIDCounter;
		}
	}
	// Allocate CUDA memory for feature patches and copy the data.
	// To avoid reloading the same feature patch multiple times, we will also store the corner coordinates, and
	// a llocate CUDA memory for interpolation coordinates and copy the data.
	// Since patch size is small, `cudaMalloc` is sufficient.
	Eigen::half* deviceFeaturePatches = nullptr;
	Eigen::Vector2d* deviceInterpolationCoordinates = nullptr;
	std::vector<Eigen::Vector2d> interpolationCoordinates(reconstruction.numValidKeypoints);
	cudaMalloc(&deviceFeaturePatches, reconstruction.numValidKeypoints * FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL * sizeof(Eigen::half));
	cudaMalloc(&deviceInterpolationCoordinates, reconstruction.numValidKeypoints * sizeof(Eigen::Vector2d));
	this->_cornerXs.resize(reconstruction.numValidKeypoints);
	this->_cornerYs.resize(reconstruction.numValidKeypoints);
	costMapIDCounter = 0ULL;
	for (const auto& cameraDataEntry : reconstruction.cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const auto& cameraData = cameraDataEntry.second;
		for (std::uint64_t keypointID = 0ULL; keypointID < cameraData.keypoints.size(); ++keypointID) {
			const auto& keypoint = cameraData.keypoints[keypointID];
			if (keypoint.point3DID == -1LL)
				continue;
			// The data in `featurePatchRef` is valid until the next call of `loadFeaturePatch`.
			FeaturePatchRef featurePatchRef;
			featureDatabase.loadFeaturePatch(reconstruction.frameName, cameraName, keypointID, featurePatchRef);
			// Store the corner coordinates.
			this->_cornerXs[costMapIDCounter] = featurePatchRef.cornerX;
			this->_cornerYs[costMapIDCounter] = featurePatchRef.cornerY;
			// Compute the interpolation coordinates.
			interpolationCoordinates[costMapIDCounter] = Eigen::Vector2d(
				keypoint.pixelPos(0) - static_cast<double>(featurePatchRef.cornerX),
				keypoint.pixelPos(1) - static_cast<double>(featurePatchRef.cornerY)
			);
			// Copy the feature patch to CUDA memory.
			cudaMemcpy(
				deviceFeaturePatches + costMapIDCounter * FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL,
				featurePatchRef.data.data(),
				FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL * sizeof(Eigen::half),
				cudaMemcpyHostToDevice
			);
			++costMapIDCounter;
		}
	}
	cudaMemcpy(deviceInterpolationCoordinates, interpolationCoordinates.data(), reconstruction.numValidKeypoints * sizeof(Eigen::Vector2d), cudaMemcpyHostToDevice);
	interpolationCoordinates.clear();
	// Allocate CUDA memory for keypoint features.
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures = nullptr;
	cudaMalloc(&deviceKeypointFeatures, reconstruction.numValidKeypoints * FEATURE_PATCH_CHANNEL * sizeof(double));
	// Allocate CUDA memory for reference features.
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures = nullptr;
	cudaMalloc(&deviceReferenceFeatures, reconstruction.point3Ds.size() * FEATURE_PATCH_CHANNEL * sizeof(double));
	// We also need to create index arrays to store the 2D-3D correspondences.
	// The key of deviceFeaturePatches, deviceInterpolationCoordinates, deviceKeypointFeatures is the cost map ID.
	// However, the key of deviceReferenceFeatures is the 3D point ID.
	// 2D to 3D correspondence: costMapID2Point3DID.
	// 3D to 2D correspondence: trackOffsets, trackLengths, tracks.
	std::vector<std::uint64_t> costMapID2Point3DID(reconstruction.numValidKeypoints);
	std::vector<std::uint64_t> trackOffsets(reconstruction.point3Ds.size());
	std::vector<std::uint64_t> trackLengths(reconstruction.point3Ds.size());
	std::vector<std::uint64_t> tracks; tracks.reserve(reconstruction.numValidKeypoints);
	for (std::uint64_t point3DID = 0ULL; point3DID < reconstruction.point3Ds.size(); ++point3DID) {
		const auto& point3D = reconstruction.point3Ds[point3DID];
		trackOffsets[point3DID] = tracks.size();
		trackLengths[point3DID] = point3D.tracks.size();
		for (const auto& track : point3D.tracks) {
			std::uint64_t costMapID = this->_cameraNameKeypointID2CostMapID.at(track.cameraName)[track.keypointID];
			tracks.push_back(costMapID);
			costMapID2Point3DID[costMapID] = point3DID;
		}
	}
	// Allocate CUDA memory for the index arrays and copy the data.
	std::uint64_t* deviceCostMapID2Point3DID = nullptr;
	std::uint64_t* deviceTrackOffsets = nullptr;
	std::uint64_t* deviceTrackLengths = nullptr;
	std::uint64_t* deviceTracks = nullptr;
	cudaMalloc(&deviceCostMapID2Point3DID, reconstruction.numValidKeypoints * sizeof(std::uint64_t));
	cudaMalloc(&deviceTrackOffsets, reconstruction.point3Ds.size() * sizeof(std::uint64_t));
	cudaMalloc(&deviceTrackLengths, reconstruction.point3Ds.size() * sizeof(std::uint64_t));
	cudaMalloc(&deviceTracks, reconstruction.numValidKeypoints * sizeof(std::uint64_t));
	cudaMemcpy(deviceCostMapID2Point3DID, costMapID2Point3DID.data(), reconstruction.numValidKeypoints * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceTrackOffsets, trackOffsets.data(), reconstruction.point3Ds.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceTrackLengths, trackLengths.data(), reconstruction.point3Ds.size() * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
	cudaMemcpy(deviceTracks, tracks.data(), reconstruction.numValidKeypoints * sizeof(std::uint64_t), cudaMemcpyHostToDevice);
	costMapID2Point3DID.clear();
	trackOffsets.clear();
	trackLengths.clear();
	tracks.clear();
	// Allocate CUDA memory for the cost maps.
	double* deviceCostMaps = nullptr;
	cudaMalloc(&deviceCostMaps, reconstruction.numValidKeypoints * COST_MAP_WIDTH * COST_MAP_HEIGHT * COST_MAP_CHANNEL * sizeof(double));
	// Compute the keypoint features.
	this->_computeKeypointFeaturesKernelDelegate(
		reconstruction.numValidKeypoints,
		deviceFeaturePatches,
		deviceInterpolationCoordinates,
		deviceKeypointFeatures
	);
	// Compute the reference features.
	this->_computeReferenceFeaturesKernelDelegate(
		reconstruction.point3Ds.size(),
		deviceTracks,
		deviceTrackOffsets,
		deviceTrackLengths,
		deviceKeypointFeatures,
		deviceReferenceFeatures
	);
	// Compute the cost maps.
	this->_computeCostMapsKernelDelegate(
		reconstruction.numValidKeypoints,
		deviceFeaturePatches,
		deviceReferenceFeatures,
		deviceCostMapID2Point3DID,
		deviceCostMaps
	);
	// Download the cost maps.
	this->_costMaps.reset(new double[reconstruction.numValidKeypoints * COST_MAP_WIDTH * COST_MAP_HEIGHT * COST_MAP_CHANNEL]);
	cudaMemcpy(this->_costMaps.get(), deviceCostMaps, reconstruction.numValidKeypoints * COST_MAP_WIDTH * COST_MAP_HEIGHT * COST_MAP_CHANNEL * sizeof(double), cudaMemcpyDeviceToHost);
	// Free CUDA memory.
	cudaFree(deviceFeaturePatches);
	cudaFree(deviceInterpolationCoordinates);
	cudaFree(deviceKeypointFeatures);
	cudaFree(deviceReferenceFeatures);
	cudaFree(deviceCostMapID2Point3DID);
	cudaFree(deviceTrackOffsets);
	cudaFree(deviceTrackLengths);
	cudaFree(deviceTracks);
	cudaFree(deviceCostMaps);
}
