#include "CostMap.hpp"
#include "Reconstruction.hpp"
#include <cuda_runtime.h>

/**
  * @brief CUDA kernel to compute the keypoint features.
  */
__global__ void computeKeypointFeaturesKernel(
	std::uint64_t numValidKeypoints,
	const Eigen::half* deviceFeaturePatches,
	const Eigen::Vector2d* deviceInterpolationCoordinates,
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures
) {
	std::uint32_t keypointID = blockIdx.x * blockDim.x + threadIdx.x; // This is the keypoint ID in CUDA threads. Not the keypoint ID in the reconstruction.
	// Handle out-of-bound cases.
	if (keypointID >= numValidKeypoints)
		return;
	// Get the base address of the feature patch.
	const Eigen::half* featurePatchBaseAddr = deviceFeaturePatches + keypointID * FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL;
	// Get the interpolation coordinates.
	const Eigen::Vector2d& interpolationCoordinate = deviceInterpolationCoordinates[keypointID];
	// Interpolate the feature vector.
	deviceKeypointFeatures[keypointID] = bicubicInterpolationDevice<double, Eigen::half, FEATURE_PATCH_CHANNEL, true>(
		featurePatchBaseAddr,
		FEATURE_PATCH_WIDTH,
		FEATURE_PATCH_HEIGHT,
		interpolationCoordinate.x(),
		interpolationCoordinate.y()
	).cast<double>();
}

void CostMaps::_computeKeypointFeaturesKernelDelegate(
	const std::uint64_t numValidKeypoints,
	const Eigen::half* deviceFeaturePatches,
	const Eigen::Vector2d* deviceInterpolationCoordinates,
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures
) {
	dim3 blockDim(256U);
	dim3 gridDim((numValidKeypoints + blockDim.x - 1U) / blockDim.x);
	computeKeypointFeaturesKernel<<<gridDim, blockDim>>>(
		numValidKeypoints,
		deviceFeaturePatches,
		deviceInterpolationCoordinates,
		deviceKeypointFeatures
	);
}

/**
  * @brief CUDA kernel to compute the reference features.
  */
__global__ void computeReferenceFeaturesKernel(
	std::uint64_t numPoint3Ds,
	const std::uint64_t* deviceTracks,
	const std::uint64_t* deviceTrackOffsets,
	const std::uint64_t* deviceTrackLengths,
	const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures,
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures
) {
	std::uint32_t point3DID = blockIdx.x * blockDim.x + threadIdx.x;
	// Handle out-of-bound cases.
	if (point3DID >= numPoint3Ds)
		return;
	// Get the tracks of the current 3D point.
	const std::uint64_t* trackBaseAddr = deviceTracks + deviceTrackOffsets[point3DID];
	const std::uint64_t trackLength = deviceTrackLengths[point3DID];
	if (trackLength == 0)
		return;
	// Initialize the mean feature under L2 norm.
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL> meanFeature = Eigen::Vector<double, FEATURE_PATCH_CHANNEL>::Zero();
	for (std::uint64_t i = 0; i < trackLength; i++) {
		meanFeature += deviceKeypointFeatures[trackBaseAddr[i]];
	}
	meanFeature /= static_cast<double>(trackLength);
	// Compute the robust mean under Cauchy loss.
	// Iteratively Reweighted Least Squares (IRLS) for Robust Mean.
	while (true) {
		Eigen::Vector<double, FEATURE_PATCH_CHANNEL> newMeanFeature = Eigen::Vector<double, FEATURE_PATCH_CHANNEL>::Zero();
		double sumWeight = 0.0;
		for (std::uint64_t i = 0; i < trackLength; i++) {
			Eigen::Vector<double, FEATURE_PATCH_CHANNEL> featureDiff = deviceKeypointFeatures[trackBaseAddr[i]] - meanFeature;
			double weight = 1.0 / (1.0 + featureDiff.squaredNorm() / (0.25 * 0.25));
			newMeanFeature += weight * deviceKeypointFeatures[trackBaseAddr[i]];
			sumWeight += weight;
		}
		newMeanFeature /= sumWeight;
		double diffNorm = (newMeanFeature - meanFeature).norm();
		meanFeature = newMeanFeature;
		// Terminate if the mean feature converges.
		if (diffNorm < 1e-3 * meanFeature.norm())
			break;
	}
	// Reference feature is the keypoint feature that is closest to the mean feature.
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL> referenceFeature = deviceKeypointFeatures[trackBaseAddr[0]];
	double minDistance = (referenceFeature - meanFeature).squaredNorm();
	for (std::uint64_t i = 1; i < trackLength; i++) {
		double distance = (deviceKeypointFeatures[trackBaseAddr[i]] - meanFeature).squaredNorm();
		if (distance < minDistance) {
			minDistance = distance;
			referenceFeature = deviceKeypointFeatures[trackBaseAddr[i]];
		}
	}
	// Store the reference feature.
	deviceReferenceFeatures[point3DID] = referenceFeature;
}

void CostMaps::_computeReferenceFeaturesKernelDelegate(
	const std::uint64_t numPoint3Ds,
	const std::uint64_t* deviceTracks,
	const std::uint64_t* deviceTrackOffsets,
	const std::uint64_t* deviceTrackLengths,
	const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceKeypointFeatures,
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures
) {
	dim3 blockDim(256U);
	dim3 gridDim((numPoint3Ds + blockDim.x - 1U) / blockDim.x);
	computeReferenceFeaturesKernel<<<gridDim, blockDim>>>(
		numPoint3Ds,
		deviceTracks,
		deviceTrackOffsets,
		deviceTrackLengths,
		deviceKeypointFeatures,
		deviceReferenceFeatures
	);
}

/**
  * @brief CUDA kernel to compute the cost maps.
  */
__global__ void computeCostMapsKernel(
	std::uint64_t numCostMaps,
	const Eigen::half* deviceFeaturePatches,
	const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures,
	const std::uint64_t* deviceCostMapID2Point3DID,
	double* deviceCostMaps
) {
	std::uint32_t costMapID = blockIdx.z;
	std::uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	std::uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
	// Handle out-of-bound cases.
	if (costMapID >= numCostMaps || x >= FEATURE_PATCH_WIDTH || y >= FEATURE_PATCH_HEIGHT)
		return;
	// Get the reference feature of the corresponding 3D point.
	const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>& referenceFeature = deviceReferenceFeatures[deviceCostMapID2Point3DID[costMapID]];
	// Get the base address of the feature patch.
	const Eigen::half* featurePatchBaseAddr = deviceFeaturePatches + costMapID * FEATURE_PATCH_WIDTH * FEATURE_PATCH_HEIGHT * FEATURE_PATCH_CHANNEL;
	// Fetch the feature vector at the current position.
	// Note that the feature patch is stored in row-major order.
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL> featureVector = reinterpret_cast<const Eigen::Vector<Eigen::half, FEATURE_PATCH_CHANNEL>*>(
		featurePatchBaseAddr + (y * FEATURE_PATCH_WIDTH + x) * FEATURE_PATCH_CHANNEL
	)->cast<double>();
	// Interpolate the feature vectors for the x and y derivatives.
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL> featureVectorX = bicubicInterpolationDevice<double, Eigen::half, FEATURE_PATCH_CHANNEL, true>(
		featurePatchBaseAddr,
		FEATURE_PATCH_WIDTH,
		FEATURE_PATCH_HEIGHT,
		static_cast<double>(x) + 0.1,
		static_cast<double>(y)
	);
	Eigen::Vector<double, FEATURE_PATCH_CHANNEL> featureVectorY = bicubicInterpolationDevice<double, Eigen::half, FEATURE_PATCH_CHANNEL, true>(
		featurePatchBaseAddr,
		FEATURE_PATCH_WIDTH,
		FEATURE_PATCH_HEIGHT,
		static_cast<double>(x),
		static_cast<double>(y) + 0.1
	);
	// Compute the cost residual.
	double costResidual = (featureVector - referenceFeature).norm();
	double dCostResidualDx = ((featureVectorX - referenceFeature).norm() - costResidual) / 0.1;
	double dCostResidualDy = ((featureVectorY - referenceFeature).norm() - costResidual) / 0.1;
	// Note that the cost map is stored in row-major order.
	double* costMapBaseAddr = deviceCostMaps + costMapID * COST_MAP_WIDTH * COST_MAP_HEIGHT * COST_MAP_CHANNEL;
	costMapBaseAddr[(y * COST_MAP_WIDTH + x) * COST_MAP_CHANNEL + 0] = costResidual;
	costMapBaseAddr[(y * COST_MAP_WIDTH + x) * COST_MAP_CHANNEL + 1] = dCostResidualDx;
	costMapBaseAddr[(y * COST_MAP_WIDTH + x) * COST_MAP_CHANNEL + 2] = dCostResidualDy;
}

void CostMaps::_computeCostMapsKernelDelegate(
	std::uint64_t numCostMaps,
	const Eigen::half* deviceFeaturePatches,
	const Eigen::Vector<double, FEATURE_PATCH_CHANNEL>* deviceReferenceFeatures,
	const std::uint64_t* deviceCostMapID2Point3DID,
	double* deviceCostMaps
) {
	dim3 blockDim(16U, 16U);
	dim3 gridDim(
		(FEATURE_PATCH_WIDTH + blockDim.x - 1U) / blockDim.x,
		(FEATURE_PATCH_HEIGHT + blockDim.y - 1U) / blockDim.y,
		static_cast<std::uint32_t>(numCostMaps)
	);
	computeCostMapsKernel<<<gridDim, blockDim>>>(
		numCostMaps,
		deviceFeaturePatches,
		deviceReferenceFeatures,
		deviceCostMapID2Point3DID,
		deviceCostMaps
	);
}