#include "BundleAdjuster.hpp"
#include <iostream>
#include <fstream>
#include <iomanip>
#include <memory>
#include <map>
#include <vector>
#include <Eigen/Eigen>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <ceres/solver.h>
#include <ceres/normal_prior.h>

bool BundleAdjusterParameters::load(const std::filesystem::path& path) {
	// TODO: Implement this function.
	return false;
}

bool BundleAdjusterParameters::save(const std::filesystem::path& path) const {
	// Check the path.
	if (std::filesystem::exists(path) && !std::filesystem::is_regular_file(path)) {
		std::cerr << "[BundleAdjusterParameters] Path \"" << path << "\" exists but is not a regular file." << std::endl;
		std::cerr << "[BundleAdjusterParameters] Please try to remove the file manually." << std::endl;
		return false;
	}
	if (std::filesystem::exists(path))
		std::filesystem::remove(path);
	// Write the parameters to the file.
	std::ofstream fout(path, std::ios::out);
	fout << std::setprecision(17);
	fout << "lambda0 = " << this->lambda0 << std::endl;
	fout << "lambda1 = " << this->lambda1 << std::endl;
	fout << "lambda2 = " << this->lambda2 << std::endl;
	fout << "lambda3 = " << this->lambda3 << std::endl;
	fout << "lambda4 = " << this->lambda4 << std::endl;
	fout << "lambda5 = " << this->lambda5 << std::endl;
	fout << "scale = " << this->scale << std::endl;
	fout << "terminationLambda1 = " << this->terminationLambda1 << std::endl;
	fout.close();
	return true;
}

/****************************************************************************************
 * @struct ReprojectionCost
 * @brief The reprojection cost.
 ****************************************************************************************/
class ReprojectionCost {

public:

	ReprojectionCost(
		const Eigen::Vector2d& keypointPosition
	)
		: keypointPosition(keypointPosition)
	{}

	template <class T>
	bool operator()(
		const T* focalLengths,
		const T* principalPoints,
		const T* rotation,
		const T* translation,
		const T* point3DPosition,
		T* residuals
	) const {

		// Rotate and translate the point.
		T p[3];
		ceres::AngleAxisRotatePoint(rotation, point3DPosition, p);
		p[0] += translation[0];
		p[1] += translation[1];
		p[2] += translation[2];

		// Perspective projection.
		T x = focalLengths[0] * p[0] / p[2] + principalPoints[0];
		T y = focalLengths[1] * p[1] / p[2] + principalPoints[1];

		// Compute residuals.
		residuals[0] = x - T(this->keypointPosition.x());
		residuals[1] = y - T(this->keypointPosition.y());

		return true;
	}

	Eigen::Vector2d keypointPosition;

};

/****************************************************************************************
 * @struct FeaturemetricReprojectionCost
 * @brief The featuremetric reprojection cost.
 ****************************************************************************************/
class FeaturemetricReprojectionCost {

public:

	FeaturemetricReprojectionCost(
		const CostMap& costMap
	)
		: costMap(costMap)
	{}

	template <class T>
	bool operator()(
		const T* focalLengths,
		const T* principalPoints,
		const T* rotation,
		const T* translation,
		const T* point3DPosition,
		T* residuals
	) const {

		// Rotate and translate the point.
		T p[3];
		ceres::AngleAxisRotatePoint(rotation, point3DPosition, p);
		p[0] += translation[0];
		p[1] += translation[1];
		p[2] += translation[2];

		// Perspective projection.
		T x = focalLengths[0] * p[0] / p[2] + principalPoints[0];
		T y = focalLengths[1] * p[1] / p[2] + principalPoints[1];

		// Interpolate the residuals
		Eigen::Vector<T, 3> residualVec = this->costMap(x, y);
		residuals[0] = residualVec(0);
		residuals[1] = residualVec(1);
		residuals[2] = residualVec(2);

		return true;
	}

	CostMap costMap;
	
};

/****************************************************************************************
 * @struct ReferenceVectorDiffCost
 * @brief The reference vector difference cost measures the difference between a variable
 *        and a reference vector.
 * @tparam N The length of the vector.
 ****************************************************************************************/
template <std::uint64_t N>
class ReferenceVectorDiffCost {

public:

	ReferenceVectorDiffCost(
		const Eigen::Vector<double, static_cast<int>(N)>& referenceVector
	)
		: referenceVector(referenceVector)
	{}

	template <class T>
	bool operator()(
		const T* vector,
		T* residuals
	) const {

		// Compute residuals.
		for (std::uint64_t i = 0; i < N; ++i)
			residuals[i] = vector[i] - this->referenceVector(i);

		return true;
	}

	Eigen::Vector<double, static_cast<int>(N)> referenceVector;

};

/****************************************************************************************
 * @struct VectorDiffCost
 * @brief The vector difference cost measures the difference between two variable vectors.
 * @tparam N The length of vectors.
 ****************************************************************************************/
template <std::uint64_t N>
class VectorDiffCost {

public:

	VectorDiffCost(void) = default;

	template <class T>
	bool operator()(
		const T* vector0,
		const T* vector1,
		T* residuals
	) const {

		// Compute residuals.
		for (std::uint64_t i = 0; i < N; ++i)
			residuals[i] = vector0[i] - vector1[i];

		return true;
	}

};

void BundleAdjuster::optimize(void) {
	// Clear records.
	this->reprojectionCostHistory.clear();
	this->featuremetricReprojectionCostHistory.clear();
	this->rotationRegularizationCostHistory.clear();
	this->translationRegularizationCostHistory.clear();
	this->focalLengthVarianceCostHistory.clear();
	this->principalPointVarianceCostHistory.clear();
	this->focalLengthErrorHistory.clear();
	this->principalPointErrorHistory.clear();
	// For each camera, find all frames that have this camera.
	std::cout << "[BundleAdjuster] Collect frame names for each camera." << std::endl;
	std::map<std::uint64_t, std::vector<std::uint64_t>> cameraName2FrameNames{};
	for (const auto& cameraParametersEntry : this->pKRT->cameraParametersMap) {
		std::uint64_t cameraName = cameraParametersEntry.first;
		cameraName2FrameNames.emplace(cameraName, std::vector<std::uint64_t>{});
	}
	for (const auto& reconstructionEntry : *this->pReconstructions) {
		std::uint64_t frameName = reconstructionEntry.first;
		const Reconstruction& reconstruction = reconstructionEntry.second;
		for (const auto& cameraDataEntry : reconstruction.cameraDataMap) {
			std::uint64_t cameraName = cameraDataEntry.first;
			cameraName2FrameNames.at(cameraName).push_back(reconstructionEntry.first);
		}
	}
	std::uint64_t numValidCameras = 0ULL;
	for (const auto& cameraName2FrameNamesEntry : cameraName2FrameNames) {
		if (cameraName2FrameNamesEntry.second.size() > 0ULL)
			++numValidCameras;
	}
	// Initialize the global intrinsics as the mean intrinsics under Cauchy loss.
	std::cout << "[BundleAdjuster] Initialize global intrinsics." << std::endl;
	for (const auto& cameraParametersEntry : this->pKRT->cameraParametersMap) {
		std::uint64_t cameraName = cameraParametersEntry.first;
		Eigen::Vector4d meanIntrinsics = Eigen::Vector4d::Zero();
		if (cameraName2FrameNames.at(cameraName).size() > 0ULL) {
			// Firstly initialize it as the mean intrinsics under L2 norm.
			for (std::uint64_t frameName : cameraName2FrameNames.at(cameraName)) {
				const Eigen::Vector4d& intrinsics = this->pReconstructions->at(frameName).cameraDataMap.at(cameraName).intrinsics;
				meanIntrinsics += intrinsics;
			}
			meanIntrinsics /= static_cast<double>(cameraName2FrameNames.at(cameraName).size());
			// Perform Iteratively Reweighted Least Squares (IRLS) to compute the mean under Cauchy loss.
			while (true) {
				Eigen::Vector4d newMeanIntrinsics = Eigen::Vector4d::Zero();
				double sumWeight = 0.0;
				for (std::uint64_t frameName : cameraName2FrameNames.at(cameraName)) {
					const Eigen::Vector4d& intrinsics = this->pReconstructions->at(frameName).cameraDataMap.at(cameraName).intrinsics;
					Eigen::Vector4d intrinsicsDiff = intrinsics - meanIntrinsics;
					double weight = 1.0 / (1.0 + intrinsicsDiff.squaredNorm() / (0.25 * 0.25));
					newMeanIntrinsics += weight * intrinsics;
					sumWeight += weight;
				}
				newMeanIntrinsics /= sumWeight;
				double diffNorm = (newMeanIntrinsics - meanIntrinsics).norm();
				meanIntrinsics = newMeanIntrinsics;
				// Terminate if the mean intrinsics converges.
				if (diffNorm < 1e-3 * meanIntrinsics.norm())
					break;
			}
		}
		this->globalIntrinsics.emplace(cameraName, meanIntrinsics);
	}
	// Construct the problem.
	std::cout << "[BundleAdjuster] Construct the problem." << std::endl;
	// Since we only change the coefficients and do not modify the topology of the problem,
	// we can reuse the problem object.
	ceres::Problem problem;
	// Avoid creating multiple loss functions - create once and reuse.
	std::unique_ptr<ceres::LossFunction> cauchyLoss(new ceres::CauchyLoss(0.25));
	// Also record loss functions and current coefficients for changing lambda1, lambda2, and lambda4, lambda5.
	std::vector<std::pair<ceres::LossFunctionWrapper*, double>> lambda1LossFunctionAndCoefficientPairs{};
	std::vector<std::pair<ceres::LossFunctionWrapper*, double>> lambda2LossFunctionAndCoefficientPairs{};
	std::vector<std::pair<ceres::LossFunctionWrapper*, double>> lambda4LossFunctionAndCoefficientPairs{};
	std::vector<std::pair<ceres::LossFunctionWrapper*, double>> lambda5LossFunctionAndCoefficientPairs{};
	// Also record the residual block IDs for loss evaluation.
	std::vector<ceres::ResidualBlockId> reprojectionCostResidualBlockIDs{};
	std::vector<ceres::ResidualBlockId> featuremetricReprojectionCostResidualBlockIDs{};
	std::vector<ceres::ResidualBlockId> rotationRegularizationCostResidualBlockIDs{};
	std::vector<ceres::ResidualBlockId> translationRegularizationCostResidualBlockIDs{};
	std::vector<ceres::ResidualBlockId> focalLengthVarianceCostResidualBlockIDs{};
	std::vector<ceres::ResidualBlockId> principalPointVarianceCostResidualBlockIDs{};
	// For each frame.
	for (auto& reconstructionEntry : *this->pReconstructions) {
		std::uint64_t frameName = reconstructionEntry.first;
		Reconstruction& reconstruction = reconstructionEntry.second;
		// Avoid creating multiple loss functions - create once and reuse.
		// Lambda0 and lambda3 are fixed during the optimization - We don't need to wrap them in `ceres::LossFunctionWrapper`s.
		double lambda0Coefficient = this->parameters.lambda0 / static_cast<double>(this->pReconstructions->size() * reconstruction.numValidKeypoints);
		double lambda3Coefficient = this->parameters.lambda3 / static_cast<double>(this->pReconstructions->size() * reconstruction.numValidKeypoints);
		ceres::LossFunction* lambda0ScaledLoss = new ceres::ScaledLoss(
			cauchyLoss.get(),
			lambda0Coefficient,
			ceres::DO_NOT_TAKE_OWNERSHIP
		);
		ceres::LossFunction* lambda3ScaledLoss = (this->parameters.lambda3 > 0.0) ?
			new ceres::ScaledLoss(
				cauchyLoss.get(),
				lambda3Coefficient,
				ceres::DO_NOT_TAKE_OWNERSHIP
			) : 
			nullptr;
		// Lambda1 and lambda2 will be changing during the optimization - We need to wrap them in `ceres::LossFunctionWrapper`s.
		double lambda1Coefficient = this->parameters.lambda1 / static_cast<double>(this->pReconstructions->size() * reconstruction.cameraDataMap.size());
		double lambda2Coefficient = this->parameters.lambda2 / static_cast<double>(this->pReconstructions->size() * reconstruction.cameraDataMap.size());
		ceres::LossFunctionWrapper* lambda1LossFunction = new ceres::LossFunctionWrapper(
			new ceres::ScaledLoss(
				cauchyLoss.get(),
				lambda1Coefficient,
				ceres::DO_NOT_TAKE_OWNERSHIP
			),
			ceres::TAKE_OWNERSHIP
		);
		ceres::LossFunctionWrapper* lambda2LossFunction = new ceres::LossFunctionWrapper(
			new ceres::ScaledLoss(
				cauchyLoss.get(),
				lambda2Coefficient,
				ceres::DO_NOT_TAKE_OWNERSHIP
			),
			ceres::TAKE_OWNERSHIP
		);
		lambda1LossFunctionAndCoefficientPairs.emplace_back(lambda1LossFunction, lambda1Coefficient);
		lambda2LossFunctionAndCoefficientPairs.emplace_back(lambda2LossFunction, lambda2Coefficient);
		// For each camera.
		for (auto& cameraDataEntry : reconstruction.cameraDataMap) {
			std::uint64_t cameraName = cameraDataEntry.first;
			Reconstruction::CameraData& cameraData = cameraDataEntry.second;
			const KRT::CameraParameters& groundTruthCameraParameters = this->pKRT->cameraParametersMap.at(cameraName);
			// For each keypoint.
			for (std::size_t keypointID = 0; keypointID < cameraData.keypoints.size(); ++keypointID) {
				const Reconstruction::Keypoint& keypoint = cameraData.keypoints[keypointID];
				if (keypoint.point3DID == -1) continue;
				// Reprojection cost.
				ceres::CostFunction* reprojectionCost = new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 2, 2, 3, 3, 3>(
					new ReprojectionCost(keypoint.pixelPos)
				);
				ceres::ResidualBlockId reprojectionCostResidualBlockID = problem.AddResidualBlock(
					reprojectionCost,
					lambda0ScaledLoss,
					cameraData.intrinsics.data(),
					cameraData.intrinsics.data() + 2,
					cameraData.rotation.data(),
					cameraData.translation.data(),
					reconstruction.point3Ds[keypoint.point3DID].position.data()
				);
				reprojectionCostResidualBlockIDs.push_back(reprojectionCostResidualBlockID);
				// Featuremetric reprojection cost.
				if (this->parameters.lambda3 > 0.0) {
					ceres::CostFunction* featuremetricReprojectionCost = new ceres::AutoDiffCostFunction<FeaturemetricReprojectionCost, 3, 2, 2, 3, 3, 3>(
						new FeaturemetricReprojectionCost(reconstruction.costMaps.getCostMap(cameraName, keypointID))
					);
					ceres::ResidualBlockId featuremetricReprojectionCostResidualBlockID = problem.AddResidualBlock(
						featuremetricReprojectionCost,
						lambda3ScaledLoss,
						cameraData.intrinsics.data(),
						cameraData.intrinsics.data() + 2,
						cameraData.rotation.data(),
						cameraData.translation.data(),
						reconstruction.point3Ds[keypoint.point3DID].position.data()
					);
					featuremetricReprojectionCostResidualBlockIDs.push_back(featuremetricReprojectionCostResidualBlockID);
				}
			}
			// Extrinsics regularization cost (rotation + translation).
			ceres::CostFunction* rotationRegularizationCost = new ceres::AutoDiffCostFunction<ReferenceVectorDiffCost<3>, 3, 3>(
				new ReferenceVectorDiffCost<3>(groundTruthCameraParameters.rotation)
			);
			ceres::ResidualBlockId rotationRegularizationCostResidualBlockID = problem.AddResidualBlock(
				rotationRegularizationCost,
				lambda1LossFunction,
				cameraData.rotation.data()
			);
			rotationRegularizationCostResidualBlockIDs.push_back(rotationRegularizationCostResidualBlockID);
			ceres::CostFunction* translationRegularizationCost = new ceres::AutoDiffCostFunction<ReferenceVectorDiffCost<3>, 3, 3>(
				new ReferenceVectorDiffCost<3>(groundTruthCameraParameters.translation)
			);
			ceres::ResidualBlockId translationRegularizationCostResidualBlockID = problem.AddResidualBlock(
				translationRegularizationCost,
				lambda2LossFunction,
				cameraData.translation.data()
			);
			translationRegularizationCostResidualBlockIDs.push_back(translationRegularizationCostResidualBlockID);
		}
	}
	// For each camera.
	for (const auto& cameraParametersEntry : this->pKRT->cameraParametersMap) {
		std::uint64_t cameraName = cameraParametersEntry.first;
		const std::vector<std::uint64_t>& frameNames = cameraName2FrameNames.at(cameraName);
		if (frameNames.size() == 0ULL) continue;
		// Lambda4 and lambda5 will be changing during the optimization - We need to wrap them in `ceres::LossFunctionWrapper`s.
		double lambda4Coefficient = this->parameters.lambda4 / static_cast<double>(this->pKRT->cameraParametersMap.size() * frameNames.size());
		double lambda5Coefficient = this->parameters.lambda5 / static_cast<double>(this->pKRT->cameraParametersMap.size() * frameNames.size());
		ceres::LossFunctionWrapper* lambda4LossFunction = new ceres::LossFunctionWrapper(
			new ceres::ScaledLoss(
				cauchyLoss.get(),
				lambda4Coefficient,
				ceres::DO_NOT_TAKE_OWNERSHIP
			),
			ceres::TAKE_OWNERSHIP
		);
		ceres::LossFunctionWrapper* lambda5LossFunction = new ceres::LossFunctionWrapper(
			new ceres::ScaledLoss(
				cauchyLoss.get(),
				lambda5Coefficient,
				ceres::DO_NOT_TAKE_OWNERSHIP
			),
			ceres::TAKE_OWNERSHIP
		);
		lambda4LossFunctionAndCoefficientPairs.emplace_back(lambda4LossFunction, lambda4Coefficient);
		lambda5LossFunctionAndCoefficientPairs.emplace_back(lambda5LossFunction, lambda5Coefficient);
		// Intrinsics variance cost (focal length + principal point).
		for (std::uint64_t frameName : frameNames) {
			Eigen::Vector4d& globalIntrinsics = this->globalIntrinsics.at(cameraName);
			Eigen::Vector4d& frameIntrinsics = this->pReconstructions->at(frameName).cameraDataMap.at(cameraName).intrinsics;
			ceres::CostFunction* focalLengthVarianceCost = new ceres::AutoDiffCostFunction<VectorDiffCost<2>, 2, 2, 2>(
				new VectorDiffCost<2>()
			);
			ceres::ResidualBlockId focalLengthVarianceCostResidualBlockID = problem.AddResidualBlock(
				focalLengthVarianceCost,
				lambda4LossFunction,
				globalIntrinsics.data(),
				frameIntrinsics.data()
			);
			focalLengthVarianceCostResidualBlockIDs.push_back(focalLengthVarianceCostResidualBlockID);
			ceres::CostFunction* principalPointVarianceCost = new ceres::AutoDiffCostFunction<VectorDiffCost<2>, 2, 2, 2>(
				new VectorDiffCost<2>()
			);
			ceres::ResidualBlockId principalPointVarianceCostResidualBlockID = problem.AddResidualBlock(
				principalPointVarianceCost,
				lambda5LossFunction,
				globalIntrinsics.data() + 2,
				frameIntrinsics.data() + 2
			);
			principalPointVarianceCostResidualBlockIDs.push_back(principalPointVarianceCostResidualBlockID);
		}
	}
	// Optimize.
	std::cout << "[BundleAdjuster] Optimize." << std::endl;
	double cumulativeScale = 1.0;
	std::int64_t iteration = -1LL; // Use iteration -1 to only evaluate costs before the first iteration.
	while (cumulativeScale * this->parameters.lambda1 <= this->parameters.terminationLambda1) {
		std::cout << "[BundleAdjuster] ----------------------------- Iteration " << iteration << " -----------------------------" << std::endl;
		std::cout << "[BundleAdjuster] lambda0 = " << this->parameters.lambda0 << std::endl;
		std::cout << "[BundleAdjuster] lambda1 = " << this->parameters.lambda1 * cumulativeScale << std::endl;
		std::cout << "[BundleAdjuster] lambda2 = " << this->parameters.lambda2 * cumulativeScale << std::endl;
		std::cout << "[BundleAdjuster] lambda3 = " << this->parameters.lambda3 << std::endl;
		std::cout << "[BundleAdjuster] lambda4 = " << this->parameters.lambda4 * cumulativeScale << std::endl;
		std::cout << "[BundleAdjuster] lambda5 = " << this->parameters.lambda5 * cumulativeScale << std::endl;
		// Configure the solver.
		if (iteration >= 0LL) {
			ceres::Solver::Options options;
			options.linear_solver_type = ceres::DENSE_SCHUR;
			options.minimizer_progress_to_stdout = true;
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.FullReport() << "\n";
		}
		// Evaluate the cost.
		double reprojectionCost = 0.0;
		for (ceres::ResidualBlockId residualBlockID : reprojectionCostResidualBlockIDs) {
			double cost{};
			problem.EvaluateResidualBlock(residualBlockID, true, &cost, nullptr, nullptr);
			reprojectionCost += cost;
		}
		if (this->parameters.lambda0 > 0.0)
			reprojectionCost /= this->parameters.lambda0;
		this->reprojectionCostHistory.push_back(reprojectionCost);
		double featuremetricReprojectionCost = 0.0;
		for (ceres::ResidualBlockId residualBlockID : featuremetricReprojectionCostResidualBlockIDs) {
			double cost{};
			problem.EvaluateResidualBlock(residualBlockID, true, &cost, nullptr, nullptr);
			featuremetricReprojectionCost += cost;
		}
		if (this->parameters.lambda3 > 0.0)
			featuremetricReprojectionCost /= this->parameters.lambda3;
		this->featuremetricReprojectionCostHistory.push_back(featuremetricReprojectionCost);
		double rotationRegularizationCost = 0.0;
		for (ceres::ResidualBlockId residualBlockID : rotationRegularizationCostResidualBlockIDs) {
			double cost{};
			problem.EvaluateResidualBlock(residualBlockID, true, &cost, nullptr, nullptr);
			rotationRegularizationCost += cost;
		}
		if (this->parameters.lambda1 > 0.0)
			rotationRegularizationCost /= this->parameters.lambda1 * cumulativeScale;
		this->rotationRegularizationCostHistory.push_back(rotationRegularizationCost);
		double translationRegularizationCost = 0.0;
		for (ceres::ResidualBlockId residualBlockID : translationRegularizationCostResidualBlockIDs) {
			double cost{};
			problem.EvaluateResidualBlock(residualBlockID, true, &cost, nullptr, nullptr);
			translationRegularizationCost += cost;
		}
		if (this->parameters.lambda2 > 0.0)
			translationRegularizationCost /= this->parameters.lambda2 * cumulativeScale;
		this->translationRegularizationCostHistory.push_back(translationRegularizationCost);
		double focalLengthVarianceCost = 0.0;
		for (ceres::ResidualBlockId residualBlockID : focalLengthVarianceCostResidualBlockIDs) {
			double cost{};
			problem.EvaluateResidualBlock(residualBlockID, true, &cost, nullptr, nullptr);
			focalLengthVarianceCost += cost;
		}
		if (this->parameters.lambda4 > 0.0)
			focalLengthVarianceCost /= this->parameters.lambda4 * cumulativeScale;
		this->focalLengthVarianceCostHistory.push_back(focalLengthVarianceCost);
		double principalPointVarianceCost = 0.0;
		for (ceres::ResidualBlockId residualBlockID : principalPointVarianceCostResidualBlockIDs) {
			double cost{};
			problem.EvaluateResidualBlock(residualBlockID, true, &cost, nullptr, nullptr);
			principalPointVarianceCost += cost;
		}
		if (this->parameters.lambda5 > 0.0)
			principalPointVarianceCost /= this->parameters.lambda5 * cumulativeScale;
		this->principalPointVarianceCostHistory.push_back(principalPointVarianceCost);
		// Evaluate the L1 error.
		double focalLengthError = 0.0;
		double principalPointError = 0.0;
		for (const auto& cameraParametersEntry : this->pKRT->cameraParametersMap) {
			std::uint64_t cameraName = cameraParametersEntry.first;
			const Eigen::Vector4d& globalIntrinsics = this->globalIntrinsics.at(cameraName);
			const Eigen::Vector4d& groundTruthIntrinsics = cameraParametersEntry.second.intrinsics;
			const std::vector<std::uint64_t>& frameNames = cameraName2FrameNames.at(cameraName);
			if (frameNames.size() == 0ULL) continue;
			focalLengthError += (globalIntrinsics.head<2>() - groundTruthIntrinsics.head<2>()).norm();
			principalPointError += (globalIntrinsics.tail<2>() - groundTruthIntrinsics.tail<2>()).norm();
		}
		focalLengthError /= static_cast<double>(numValidCameras);
		principalPointError /= static_cast<double>(numValidCameras);
		this->focalLengthErrorHistory.push_back(focalLengthError);
		this->principalPointErrorHistory.push_back(principalPointError);
		std::cout << "[BundleAdjuster] Reprojection cost = " << reprojectionCost << std::endl;
		std::cout << "[BundleAdjuster] Featuremetric reprojection cost = " << featuremetricReprojectionCost << std::endl;
		std::cout << "[BundleAdjuster] Rotation regularization cost = " << rotationRegularizationCost << std::endl;
		std::cout << "[BundleAdjuster] Translation regularization cost = " << translationRegularizationCost << std::endl;
		std::cout << "[BundleAdjuster] Focal length variance cost = " << focalLengthVarianceCost << std::endl;
		std::cout << "[BundleAdjuster] Principal point variance cost = " << principalPointVarianceCost << std::endl;
		std::cout << "[BundleAdjuster] Focal length L1 error = " << focalLengthError << std::endl;
		std::cout << "[BundleAdjuster] Principal point L1 error = " << principalPointError << std::endl;
		// Update coefficients.
		iteration += 1LL;
		if (iteration == 0LL) continue;
		cumulativeScale *= this->parameters.scale;
		for (auto& lossFunctionAndCoefficientPairsRef : {
			std::ref(lambda1LossFunctionAndCoefficientPairs),
			std::ref(lambda2LossFunctionAndCoefficientPairs),
			std::ref(lambda4LossFunctionAndCoefficientPairs),
			std::ref(lambda5LossFunctionAndCoefficientPairs)
		}) { 
			for (auto& lossFunctionAndCoefficientPair : lossFunctionAndCoefficientPairsRef.get()) {
				lossFunctionAndCoefficientPair.second *= this->parameters.scale;
				lossFunctionAndCoefficientPair.first->Reset(
					new ceres::ScaledLoss(
						cauchyLoss.get(),
						lossFunctionAndCoefficientPair.second,
						ceres::DO_NOT_TAKE_OWNERSHIP
					),
					ceres::TAKE_OWNERSHIP
				);
			}
		}
	}
	std::cout << "[BundleAdjuster] Done." << std::endl;
	// Update reconstruction 3D point errors.
	for (auto& reconstructionEntry : *this->pReconstructions)
		reconstructionEntry.second.updatePoint3DErrors();
	// Print history.
	std::cout << "[BundleAdjuster] Reprojection cost history:" << std::endl;
	for (double cost : this->reprojectionCostHistory)
		std::cout << cost << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Featuremetric reprojection cost history:" << std::endl;
	for (double cost : this->featuremetricReprojectionCostHistory)
		std::cout << cost << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Rotation regularization cost history:" << std::endl;
	for (double cost : this->rotationRegularizationCostHistory)
		std::cout << cost << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Translation regularization cost history:" << std::endl;
	for (double cost : this->translationRegularizationCostHistory)
		std::cout << cost << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Focal length variance cost history:" << std::endl;
	for (double cost : this->focalLengthVarianceCostHistory)
		std::cout << cost << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Principal point variance cost history:" << std::endl;
	for (double cost : this->principalPointVarianceCostHistory)
		std::cout << cost << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Focal length error history:" << std::endl;
	for (double focalLengthError : this->focalLengthErrorHistory)
		std::cout << focalLengthError << " ";
	std::cout << std::endl;
	std::cout << "[BundleAdjuster] Principal point error history:" << std::endl;
	for (double principalPointError : this->principalPointErrorHistory)
		std::cout << principalPointError << " ";
	std::cout << std::endl;
}

bool BundleAdjuster::saveHistory(const std::filesystem::path& path) const {
	// Check the path.
	if (std::filesystem::exists(path) && !std::filesystem::is_regular_file(path)) {
		std::cerr << "[BundleAdjuster] Path \"" << path << "\" exists but is not a regular file." << std::endl;
		std::cerr << "[BundleAdjuster] Please try to remove the file manually." << std::endl;
		return false;
	}
	if (std::filesystem::exists(path))
		std::filesystem::remove(path);
	// Write the history to the file.
	std::ofstream fout(path, std::ios::out);
	fout << std::setprecision(17);
	fout << "Name";
	for (std::uint64_t i = 0ULL; i < this->reprojectionCostHistory.size(); ++i)
		fout << ",Value" << i;
	fout << std::endl;
	fout << "ReprojectionCost";
	for (double cost : this->reprojectionCostHistory)
		fout << "," << cost;
	fout << std::endl;
	fout << "FeaturemetricReprojectionCost";
	for (double cost : this->featuremetricReprojectionCostHistory)
		fout << "," << cost;
	fout << std::endl;
	fout << "RotationRegularizationCost";
	for (double cost : this->rotationRegularizationCostHistory)
		fout << "," << cost;
	fout << std::endl;
	fout << "TranslationRegularizationCost";
	for (double cost : this->translationRegularizationCostHistory)
		fout << "," << cost;
	fout << std::endl;
	fout << "FocalLengthVarianceCost";
	for (double cost : this->focalLengthVarianceCostHistory)
		fout << "," << cost;
	fout << std::endl;
	fout << "PrincipalPointVarianceCost";
	for (double cost : this->principalPointVarianceCostHistory)
		fout << "," << cost;
	fout << std::endl;
	fout << "FocalLengthError";
	for (double error : this->focalLengthErrorHistory)
		fout << "," << error;
	fout << std::endl;
	fout << "PrincipalPointError";
	for (double error : this->principalPointErrorHistory)
		fout << "," << error;
	fout << std::endl;
	fout.close();

	return true;

}

bool BundleAdjuster::saveGlobalIntrinsics(const std::filesystem::path& path, const KRT& krt) const {
	// Check the path.
	if (std::filesystem	::exists(path) && !std::filesystem::is_regular_file(path)) {
		std::cerr << "[BundleAdjuster] Path \"" << path << "\" exists but is not a regular file." << std::endl;
		std::cerr << "[BundleAdjuster] Please try to remove the file manually." << std::endl;
		return false;
	}
	if (std::filesystem::exists(path))
		std::filesystem::remove(path);
	// Write the global intrinsics to the file.
	std::ofstream fout(path, std::ios::out);
	fout << std::setprecision(10);
	for (const auto& globalIntrinsicsEntry : this->globalIntrinsics) {
		std::uint64_t cameraName = globalIntrinsicsEntry.first;
		const Eigen::Vector4d& globalIntrinsics = globalIntrinsicsEntry.second;
		const KRT::CameraParameters& groundTruthCameraParameters = krt.cameraParametersMap.at(cameraName);
		fout << std::setw(6) << std::setfill('0') << cameraName << std::endl;
		fout << globalIntrinsics(0) << " 0.0 " << globalIntrinsics(2) << std::endl;
		fout << "0.0 " << globalIntrinsics(1) << " " << globalIntrinsics(3) << std::endl;
		fout << "0.0 0.0 1.0" << std::endl;
		fout << "0.0 0.0 0.0 0.0 0.0" << std::endl;
		Eigen::Matrix3d rotationMat{}; // Rotation matrix.
		Eigen::AngleAxisd rotationAngleAxis{}; // Rotation in axis-angle representation, stored as Eigen::AngleAxisd.
		rotationAngleAxis.axis() = groundTruthCameraParameters.rotation.normalized();
		rotationAngleAxis.angle() = groundTruthCameraParameters.rotation.norm();
		rotationMat = rotationAngleAxis.toRotationMatrix();
		fout << rotationMat(0, 0) << " " << rotationMat(0, 1) << " " << rotationMat(0, 2) << " " << groundTruthCameraParameters.translation(0) << std::endl;
		fout << rotationMat(1, 0) << " " << rotationMat(1, 1) << " " << rotationMat(1, 2) << " " << groundTruthCameraParameters.translation(1) << std::endl;
		fout << rotationMat(2, 0) << " " << rotationMat(2, 1) << " " << rotationMat(2, 2) << " " << groundTruthCameraParameters.translation(2) << std::endl;
		fout << std::endl;
	}
	fout.close();
	return true;
}