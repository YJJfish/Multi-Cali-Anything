#include <filesystem>
#include <vector>
#include <map>
#include <iomanip>
#include <argparse/argparse.hpp>
#include <Eigen/Eigen>

#include "Reconstruction.hpp"
#include "KRT.hpp"
#include "FeatureDatabase.hpp"
#include "BundleAdjuster.hpp"

/**
  * @brief Align the reconstruction's coordinate system to KRT's coordinate system.
  *
  * This function performs ICP algorithm to align the reconstruction's coordinate system to
  * KRT's coordinate system. The camera extrinsics and 3D points in the reconstruction will be
  * transformed.
  * 
  * @param reconstruction 	The reconstruction to be aligned.
  * @param krt 				The KRT data.
  * @return The mean squared distance of cameras after alignment.
  */
double alignCoordinateSystems(
	const KRT& krt,
	Reconstruction& reconstruction
) {
	// Convert extrinsics to point clouds.
	Eigen::Matrix3Xd P(3, reconstruction.cameraDataMap.size());
	Eigen::Matrix3Xd Q(3, reconstruction.cameraDataMap.size());
	std::size_t cnt = 0;
	for (const auto& cameraDataEntry : reconstruction.cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const Reconstruction::CameraData& reconCameraData = cameraDataEntry.second;
		const KRT::CameraParameters& krtCameraData = krt.cameraParametersMap.at(cameraName);
		P.col(cnt) = 
			-Eigen::Matrix3d(
				Eigen::AngleAxisd(reconCameraData.rotation.norm(), reconCameraData.rotation.normalized())
			).transpose() * reconCameraData.translation;
		Q.col(cnt) = 
			-Eigen::Matrix3d(
				Eigen::AngleAxisd(krtCameraData.rotation.norm(), krtCameraData.rotation.normalized())
			).transpose() * krtCameraData.translation;
		++cnt;
	}
	std::cout << "Mean squared distance of cameras before alignment: " << (Q - P).array().square().sum() / static_cast<double>(reconstruction.cameraDataMap.size()) << std::endl;
	// Translate the point clouds' centroids to the origin.
	Eigen::Vector3d centroidP = P.rowwise().mean();
	Eigen::Vector3d centroidQ = Q.rowwise().mean();
	Eigen::Matrix3Xd centeredP = P.colwise() - centroidP;
	Eigen::Matrix3Xd centeredQ = Q.colwise() - centroidQ;
	// Solve for the scale and rotation.
	Eigen::Matrix3d C = (centeredP * centeredQ.transpose()) / static_cast<double>(reconstruction.cameraDataMap.size());
	Eigen::JacobiSVD<Eigen::Matrix3d, Eigen::ComputeFullU | Eigen::ComputeFullV> svd;
	svd.compute(C, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::Matrix3d U = svd.matrixU();
	Eigen::Vector3d S = svd.singularValues();
	Eigen::Matrix3d V = svd.matrixV();
	// Make sure |V * U^T| is positive.
	if ((V * U.transpose()).determinant() < 0.0) {
		S[2] *= -1.0;
		U.col(2) *= -1.0;
	}
	Eigen::Matrix3d rotation = V * U.transpose();
	double scale = S.sum() / (centeredP.array().square().sum() / static_cast<double>(reconstruction.cameraDataMap.size()));
	// Solve for the translation.
	Eigen::Vector3d translation = centroidQ - scale * rotation * centroidP;
	Eigen::Matrix3Xd transformedP = (scale * rotation * P).colwise() + translation;
	// Compute the error.
	double meanSquaredDistance = (Q - transformedP).array().square().sum() / static_cast<double>(reconstruction.cameraDataMap.size());
	std::cout << "Mean squared distance of cameras after alignment: " << meanSquaredDistance << std::endl;
	// Transform camera extrinsics and 3D points in the reconstruction data.
	for (auto& cameraDataEntry : reconstruction.cameraDataMap) {
		Reconstruction::CameraData& cameraData = cameraDataEntry.second;
		Eigen::Matrix3d R0 = Eigen::AngleAxisd(cameraData.rotation.norm(), cameraData.rotation.normalized()).matrix();
		Eigen::AngleAxisd angleAxis = Eigen::AngleAxisd(R0 * rotation.transpose());
		cameraData.rotation = angleAxis.axis() * angleAxis.angle();
		cameraData.translation = scale * cameraData.translation - R0 * rotation.transpose() * translation;
	}
	for (auto& point3D : reconstruction.point3Ds) {
		point3D.position = scale * rotation * point3D.position + translation;
	}
	// Return the mean squared distance.
	return meanSquaredDistance;
}

int main(int argc, char** argv) {

	google::InitGoogleLogging(argv[0]);
	
	// Parse arguments
	BundleAdjusterParameters bundleAdjusterParameters{}; // Default parameters.
	argparse::ArgumentParser program("Camera Calibration", "1.0");
	program
		.add_argument("reconstruction")
		.help("Path to the COLMAP/PixelSfM reconstruction results.");
	program
		.add_argument("feature")
		.help("Path to the dense feature database. If lambda3 is 0, this argument is ignored.");
	program
		.add_argument("KRT")
		.help("Path to KRT.");
	program
		.add_argument("--output")
		.help("Path to the output directory. If not specified, the results won't be saved.")
		.nargs(1)
		.default_value("");
	program
		.add_argument("--mode")
		.help("0 for COLMAP, 1 for PixelSfM.")
		.nargs(1)
		.scan<'i', int>()
		.choices(0, 1)
		.default_value(0);
	program
		.add_argument("--lambda0")
		.help("The coefficient for reprojection loss.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.lambda0));
	program
		.add_argument("--lambda1")
		.help("The initial coefficient for rotation extrinsics regularization loss.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.lambda1));
	program
		.add_argument("--lambda2")
		.help("The initial coefficient for translation extrinsics regularization loss.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.lambda2));
	program
		.add_argument("--lambda3")
		.help("The coefficient for featuremetric reprojection loss.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.lambda3));
	program
		.add_argument("--lambda4")
		.help("The initial coefficient for focal length intrinsics variance loss.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.lambda4));
	program
		.add_argument("--lambda5")
		.help("The initial coefficient for principal point intrinsics variance loss.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.lambda5));
	program
		.add_argument("--scale")
		.help("The scale factor to be applied with lambda1, lambda2, and lambda4.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.scale));
	program
		.add_argument("--terminationLambda1")
		.help("The termination value for lambda1.")
		.nargs(1)
		.scan<'g', float>()
		.default_value(static_cast<float>(bundleAdjusterParameters.terminationLambda1));
	program.
		add_argument("--frameNames")
		.help("The list of frame names to be included in the optimization.")
		.nargs(argparse::nargs_pattern::any)
		.scan<'i', int>()
		.default_value(std::vector<int>{});
	try {
		program.parse_args(argc, argv);
	}
	catch (const std::exception& err) {
		std::cerr << err.what() << std::endl;
		std::cerr << program;
		return EXIT_FAILURE;
	}
	bundleAdjusterParameters.lambda0 = static_cast<double>(program.get<float>("--lambda0"));
	bundleAdjusterParameters.lambda1 = static_cast<double>(program.get<float>("--lambda1"));
	bundleAdjusterParameters.lambda2 = static_cast<double>(program.get<float>("--lambda2"));
	bundleAdjusterParameters.lambda3 = static_cast<double>(program.get<float>("--lambda3"));
	bundleAdjusterParameters.lambda4 = static_cast<double>(program.get<float>("--lambda4"));
	bundleAdjusterParameters.scale = static_cast<double>(program.get<float>("--scale"));
	bundleAdjusterParameters.terminationLambda1 = static_cast<double>(program.get<float>("--terminationLambda1"));

	// Load data.
	std::cout << "================================= Load data =================================" << std::endl;
	// Load KRT data.
	std::cout << "Loading KRT data." << std::endl;
	KRT krt;
	if (!krt.load(program.get<std::string>("KRT")))
		return EXIT_FAILURE;
	std::cout << "Loaded " << krt.cameraParametersMap.size() << " cameras from KRT." << std::endl;
	for (const auto& cameraParametersEntry : krt.cameraParametersMap)
		std::cout << cameraParametersEntry.first << " ";
	std::cout << std::endl;
	// Load COLMAP/PixelSfM reconstruction results.
	std::vector<int> frameNames = program.get<std::vector<int>>("--frameNames");
	if (program.get<int>("--mode") == 0)
		std::cout << "Loading COLMAP reconstruction results." << std::endl;
	else
		std::cout << "Loading PixelSfM reconstruction results." << std::endl;
	std::map<std::uint64_t, Reconstruction> reconstructions{};
	for (const auto& entry : std::filesystem::directory_iterator(program.get<std::string>("reconstruction"))) {
		std::cout << entry.path() << std::endl;
		Reconstruction reconstruction{};
		std::filesystem::path dataPath{};
		// Reconstructions by `colmap_batch.py` and `pixelsfm_batch.py` are stored in different directories.
		if (program.get<int>("--mode") == 0)
			dataPath = entry.path() / "sparse" / "0";
		else
			dataPath = entry.path() / "refined";
		if (!reconstruction.load(dataPath))
			continue;
		if (frameNames.size() > 0 && std::find(frameNames.begin(), frameNames.end(), static_cast<int>(reconstruction.frameName)) == frameNames.end())
			continue;
		reconstruction.updatePoint3DErrors();
		std::cout << reconstruction.summary(krt) << std::endl;
		reconstructions.emplace(reconstruction.frameName, std::move(reconstruction));
	}
	// Print frame names.
	std::cout << reconstructions.size() << " valid frames." << std::endl;
	for (const auto& reconstructionEntry : reconstructions) {
		std::cout << reconstructionEntry.first << " ";
	}
	std::cout << std::endl;

	// Align the coordinate systems.
	std::cout << "================================= Align coordinate systems =================================" << std::endl;
	for (auto& reconstructionEntry : reconstructions) {
		std::uint64_t frameName = reconstructionEntry.first;
		std::cout << "---- Frame " << frameName << " ----" << std::endl;
		Reconstruction& reconstruction = reconstructionEntry.second;
		double error = alignCoordinateSystems(krt, reconstruction);
	}

	// Connect to the feature database.
	std::cout << "================================= Connect to the feature database =================================" << std::endl;
	std::cout << "SQLite Version: " << sqlite3_libversion() << std::endl;
	FeatureDatabase featureDatabase;
	if (program.get<float>("--lambda3") > 0.0) {
		if (!featureDatabase.connect(program.get<std::string>("feature")))
			return EXIT_FAILURE;
		std::cout << "Connected to the feature database: " << featureDatabase.isConnected() << std::endl;
	} else {
		std::cout << "Featuremetric loss is disabled. Skipping." << std::endl;
	}

	// Compute the cost map for each keypoint.
	std::cout << "================================= Compute cost maps =================================" << std::endl;
	if (program.get<float>("--lambda3") > 0.0) {
		for (auto& reconstructionEntry : reconstructions) {
			std::uint64_t frameName = reconstructionEntry.first;
			Reconstruction& reconstruction = reconstructionEntry.second;
			std::cout << "---- Frame " << frameName << " ----" << std::endl;
			reconstruction.computeCostMaps(featureDatabase);
		}
	} else {
		std::cout << "Featuremetric loss is disabled. Skipping." << std::endl;
	}

	// Optimize.
	std::cout << "================================= Optimize =================================" << std::endl;
	BundleAdjuster bundleAdjuster;
	bundleAdjuster.parameters = bundleAdjusterParameters;
	bundleAdjuster.pReconstructions = &reconstructions;
	bundleAdjuster.pKRT = &krt;
	bundleAdjuster.pFeatureDatabase = &featureDatabase;
	bundleAdjuster.optimize();

	// Save the results.
	std::cout << "================================= Save results =================================" << std::endl;
	if (program.get<std::string>("--output").empty()) {
		std::cout << "Output directory is not specified. Skipping." << std::endl;
	} else {
		// Check the output directory.
		std::filesystem::path outputDirectory = program.get<std::string>("--output");
		if (std::filesystem::exists(outputDirectory) && !std::filesystem::is_directory(outputDirectory)) {
			std::cerr << "Output path is not a directory." << std::endl;
			return EXIT_FAILURE;
		}
		if (!std::filesystem::exists(outputDirectory))
			std::filesystem::create_directories(outputDirectory);
		// Save the reconstructions.
		std::cout << "Saving the reconstructions." << std::endl;
		for (const auto& reconstructionEntry : reconstructions) {
			std::uint64_t frameName = reconstructionEntry.first;
			const Reconstruction& reconstruction = reconstructionEntry.second;
			std::stringstream ss;
			ss << "frame" << std::setw(6) << std::setfill('0') << frameName;
			reconstruction.save(outputDirectory / ss.str(), krt);
		}
		// Save the parameters.
		std::cout << "Saving the parameters." << std::endl;
		bundleAdjusterParameters.save(outputDirectory / "parameters.txt");
		// Save the history.
		std::cout << "Saving the optimization history." << std::endl;
		bundleAdjuster.saveHistory(outputDirectory / "history.csv");
		// Save the global intrinsics.
		std::cout << "Saving the global intrinsics." << std::endl;
		bundleAdjuster.saveGlobalIntrinsics(outputDirectory / "optimized.txt", krt);
	}

	return EXIT_SUCCESS;
}