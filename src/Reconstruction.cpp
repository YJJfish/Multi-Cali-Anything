#include "Reconstruction.hpp"
#include "KRT.hpp"
#include <iostream>
#include <iomanip>
#include <cstring>
#include <fstream>
#include <sstream>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

bool Reconstruction::load(const std::filesystem::path& path) {
	if (!std::filesystem::is_regular_file(path / "cameras.txt") ||
		!std::filesystem::is_regular_file(path / "images.txt") ||
		!std::filesystem::is_regular_file(path / "points3D.txt")
	) {
		std::cerr << "[Reconstruction] No reconstruction data found at \"" << path << "\"." << std::endl;
		return false;
	}
	this->cameraDataMap.clear();
	this->point3Ds.clear();
	this->frameName = 0ULL;
	this->numValidKeypoints = 0ULL;
	this->numValidPoints3D = 0ULL;
	std::ifstream fin;

	// Read point3Ds.
	fin.open(path / "points3D.txt", std::ios::in);

	// The 3D points in "points3D.txt" are unordered. We store them and their IDs in temporary vectors first.
	std::vector<std::uint64_t> point3DIDs;
	std::vector<Point3D> unorderedPoint3Ds;
	std::uint64_t maxPoint3DID = 0ULL;

	while (true) {
		std::string line;
		std::stringstream stream;
		std::uint64_t point3DID{};
		Point3D point3D{};

		// Skip comments.
		do { std::getline(fin, line); } while (!line.empty() && line[0] == '#');
		if (line.empty()) break;

		// Point3D ID, x, y, z, r, g, b, error, track[] as (image ID, keypoint ID).
		// We ignore the error and track[] fields because we will compute them later.
		stream << line;
		stream >> point3DID >> point3D.position(0) >> point3D.position(1) >> point3D.position(2) >> point3D.color(0) >> point3D.color(1) >> point3D.color(2);
		line.clear(); stream.clear();

		point3DIDs.push_back(point3DID);
		unorderedPoint3Ds.push_back(point3D);
		maxPoint3DID = std::max(maxPoint3DID, point3DID);
	}
	this->numValidPoints3D = point3DIDs.size();
	fin.close();

	// Reorder the point3Ds according to their IDs.
	this->point3Ds.resize(maxPoint3DID + 1ULL);
	for (std::size_t i = 0; i < point3DIDs.size(); ++i)
		this->point3Ds[point3DIDs[i]] = unorderedPoint3Ds[i];

	// Read cameras.
	fin.open(path / "cameras.txt", std::ios::in);

	// Cameras in "cameras.txt" are identified by their IDs (assigned by COLMAP/PixelSfM).
	// However we will use a 6-digit camera name (assigned by multiface dataset) to identify them.
	// We first create a map from camera ID to camera intrinsics.
	std::map<std::uint64_t, Eigen::Vector4d> cameraID2IntrinsicsMap{};

	while (true) {
		std::string line;
		std::stringstream stream;
		std::string temp;
		std::uint64_t cameraID{};
		Eigen::Vector4d intrinsics{};

		// Skip comments.
		do { std::getline(fin, line); } while (!line.empty() && line[0] == '#');
		if (line.empty()) break;

		// Camera ID, model, w, h, fx, fy, cx, cy.
		stream << line;
		stream >> cameraID >> temp >> temp >> temp >> intrinsics(0) >> intrinsics(1) >> intrinsics(2) >> intrinsics(3);
		line.clear(); stream.clear();

		cameraID2IntrinsicsMap[cameraID] = intrinsics;
	}
	fin.close();

	// Read images.
	fin.open(path / "images.txt", std::ios::in);

	while (true) {
		std::string line;
		std::stringstream stream;
		std::string temp;
		std::uint64_t imageID{};
		Eigen::Quaterniond rotationQuat{}; // Rotation quaternion.
		Eigen::AngleAxisd rotationAngleAxis{}; // Rotation in angle-axis representation, stored as Eigen::AngleAxisd.
		Eigen::Vector3d rotation{}; // Rotation in angle-axis representation, stored as Eigen::Vector3d.
		Eigen::Vector3d translation{};
		std::uint64_t cameraID{};
		std::string imageFileName = "";
		std::uint64_t cameraName{};
		std::vector<Keypoint> keypoints{};

		// Skip comments.
		do { std::getline(fin, line); } while (!line.empty() && line[0] == '#');
		if (line.empty()) break;

		// Image ID, qw, qx, qy, qz, tx, ty, tz, camera ID, image file name.
		stream << line;
		stream >> imageID >> rotationQuat.w() >> rotationQuat.x() >> rotationQuat.y() >> rotationQuat.z()
			>> translation(0) >> translation(1) >> translation(2) >> cameraID >> imageFileName;
		line.clear(); stream.clear();
		
		// Convert the rotation quaternion to angle-axis representation.
		rotationAngleAxis = Eigen::AngleAxisd(rotationQuat);
		rotation = rotationAngleAxis.axis() * rotationAngleAxis.angle();

		// Extract the camera name and frame name from the image file name.
		// Image file name format: "camera[6-digit camera name]_frame[6-digit frame name].png".
		cameraName = std::stoull(imageFileName.substr(6ULL, 6ULL));
		this->frameName = std::stoull(imageFileName.substr(18ULL, 6ULL));
		
		// Skip comments.
		do { std::getline(fin, line); } while (!line.empty() && line[0] == '#');
		if (line.empty()) break;

		// Keypoint[] as (x, y, point3D ID).
		stream << line;
		while (true) {
			Keypoint Keypoint{};
			stream >> Keypoint.pixelPos(0) >> Keypoint.pixelPos(1) >> Keypoint.point3DID;
			if (!stream.good())
				break;
			keypoints.push_back(Keypoint);
			if (Keypoint.point3DID != -1)
				++this->numValidKeypoints;
		}
		line.clear(); stream.clear();

		// Store the data.
		this->cameraDataMap.insert(
			std::make_pair(
				cameraName,
				CameraData{
					.rotation = rotation,
					.translation = translation,
					.intrinsics = cameraID2IntrinsicsMap.find(cameraID)->second,
					.keypoints = std::move(keypoints)
				}
			)
		);
	}
	fin.close();

	// Compute the tracks of 3D points.
	for (const auto& cameraDataEntry : this->cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const CameraData& cameraData = cameraDataEntry.second;
		for (std::uint64_t keypointID = 0; keypointID < cameraData.keypoints.size(); ++keypointID) {
			const Keypoint& keypoint = cameraData.keypoints[keypointID];
			if (keypoint.point3DID == -1) continue;
			this->point3Ds[keypoint.point3DID].tracks.push_back(
				Track{
					.cameraName = cameraName,
					.keypointID = keypointID
				}
			);
		}
	}

	// Update the 3D point errors.
	this->updatePoint3DErrors();

	return true;

}

bool Reconstruction::save(const std::filesystem::path& path, const KRT& krt) const {
	// Check the path.
	if (std::filesystem::exists(path) && !std::filesystem::is_directory(path)) {
		std::cerr << "[Reconstruction] Path \"" << path << "\" exists but is not a directory." << std::endl;
		return false;
	}
	if (!std::filesystem::exists(path))
		std::filesystem::create_directories(path);
	for (const auto& fileName : { "cameras.txt", "images.txt", "points3D.txt" }) {
		if (std::filesystem::exists(path / fileName) && !std::filesystem::is_regular_file(path / fileName)) {
			std::cerr << "[Reconstruction] File \"" << path / fileName << "\" exists but is not a regular file." << std::endl;
			std::cerr << "[Reconstruction] Please try to remove the file manually." << std::endl;
			return false;
		}
		if (std::filesystem::exists(path / fileName))
			std::filesystem::remove(path / fileName);
	}
	std::ofstream fout;

	// Create a map from camera name to image ID.
	// Camera ID is the same as image ID.
	std::map<std::uint64_t, std::uint64_t> cameraName2ImageID{};
	std::uint64_t imageIDCounter = 0ULL;
	for (const auto& cameraParametersEntry : krt.cameraParametersMap) {
		std::uint64_t cameraName = cameraParametersEntry.first;
		cameraName2ImageID[cameraName] = imageIDCounter++;
	}

	// Write point3Ds.
	fout.open(path / "points3D.txt", std::ios::out);
	fout << std::setprecision(17);
	fout << "# 3D point list with one line of data per point:" << std::endl;
	fout << "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)" << std::endl;
	fout << "# Number of points: " << this->point3Ds.size() << std::endl;
	for (std::uint64_t point3DID = 0ULL; point3DID < this->point3Ds.size(); ++point3DID) {
		const Point3D& point3D = this->point3Ds[point3DID];
		if (point3D.tracks.size() == 0ULL) continue;
		fout << point3DID << " " << point3D.position(0) << " " << point3D.position(1) << " " << point3D.position(2) << " "
			<< static_cast<int>(point3D.color(0)) << " " << static_cast<int>(point3D.color(1)) << " " << static_cast<int>(point3D.color(2)) << " "
			<< point3D.error;
		for (const Track& track : point3D.tracks)
			fout << " " << cameraName2ImageID.at(track.cameraName) << " " << track.keypointID;
		fout << std::endl;
	}
	fout.close();

	// Write cameras.
	fout.open(path / "cameras.txt", std::ios::out);
	fout << std::setprecision(17);
	fout << "# Camera list with one line of data per camera:" << std::endl;
	fout << "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]" << std::endl;
	fout << "# Number of cameras: " << this->cameraDataMap.size() << std::endl;
	for (const auto& cameraDataEntry : this->cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const CameraData& cameraData = cameraDataEntry.second;
		fout << cameraName2ImageID.at(cameraName) << " PINHOLE 1334 2048 " << cameraData.intrinsics(0) << " " << cameraData.intrinsics(1) << " " << cameraData.intrinsics(2) << " " << cameraData.intrinsics(3) << std::endl;
	}
	fout.close();

	// Write images.
	fout.open(path / "images.txt", std::ios::out);
	fout << std::setprecision(17);
	fout << "# Image list with two lines of data per image:" << std::endl;
	fout << "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME" << std::endl;
	fout << "#   POINTS2D[] as (X, Y, POINT3D_ID)" << std::endl;
	fout << "# Number of images: " << this->cameraDataMap.size() << ", mean observations per image: " << static_cast<double>(this->numValidKeypoints) / static_cast<double>(this->cameraDataMap.size()) << std::endl;
	for (const auto& cameraDataEntry : this->cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const CameraData& cameraData = cameraDataEntry.second;
		Eigen::Quaterniond rotationQuat{}; // Rotation quaternion.
		Eigen::AngleAxisd rotationAngleAxis{}; // Rotation in angle-axis representation, stored as Eigen::AngleAxisd.
		rotationAngleAxis.angle() = cameraData.rotation.norm();
		rotationAngleAxis.axis() = cameraData.rotation.normalized();
		rotationQuat = Eigen::Quaterniond(rotationAngleAxis);
		fout << cameraName2ImageID.at(cameraName) << " "
			<< rotationQuat.w() << " " << rotationQuat.x() << " " << rotationQuat.y() << " " << rotationQuat.z() << " "
			<< cameraData.translation(0) << " " << cameraData.translation(1) << " " << cameraData.translation(2) << " "
			<< cameraName << " camera" << std::setw(6) << std::setfill('0') << cameraName << "_frame" << std::setw(6) << std::setfill('0') << this->frameName << ".png" << std::endl;
		for (std::uint64_t keypointID = 0; keypointID < cameraData.keypoints.size(); ++keypointID) {
			const Keypoint& keypoint = cameraData.keypoints[keypointID];
			if (keypointID > 0ULL)
				fout << " ";
			fout << keypoint.pixelPos(0) << " " << keypoint.pixelPos(1) << " " << keypoint.point3DID;
		}
		fout << std::endl;
	}
	fout.close();

	return true;

}

void Reconstruction::updatePoint3DErrors(void) {
	for (Point3D& point3D : this->point3Ds) {
		point3D.error = 0.0;
		for (const Track& track : point3D.tracks) {
			const CameraData& cameraData = this->cameraDataMap.find(track.cameraName)->second;
			const Keypoint& keypoint = cameraData.keypoints[track.keypointID];
			// Rotate and translate the point.
			Eigen::Vector3d p;
			ceres::AngleAxisRotatePoint(cameraData.rotation.data(), point3D.position.data(), p.data());
			p += cameraData.translation;
			// Perspective projection.
			Eigen::Vector2d projection(
				cameraData.intrinsics(0) * p(0) / p(2) + cameraData.intrinsics(2),
				cameraData.intrinsics(1) * p(1) / p(2) + cameraData.intrinsics(3)
			);
			// Compute error.
			point3D.error += (projection - keypoint.pixelPos).norm();
		}
		if (point3D.tracks.size() > 0ULL)
			point3D.error /= static_cast<double>(point3D.tracks.size());
	}
}

void Reconstruction::computeCostMaps(FeatureDatabase& featureDatabase) {
	this->costMaps.computeCostMaps(*this, featureDatabase);
}

std::string Reconstruction::summary(const KRT& krt) const {
	std::stringstream stream;
	stream << "[Reconstruction] Reprojection error summary:" << std::endl;
	// Print # of valid keypoints and 3D points.
	stream << "\tNumber of valid keypoints: " << this->numValidKeypoints << std::endl;
	stream << "\tNumber of valid 3D points: " << this->numValidPoints3D << std::endl;
	// Print cameras.
	stream << "\t" << this->cameraDataMap.size() << " registered cameras." << std::endl;
	stream << "\t";
	for (const auto& cameraDataEntry : this->cameraDataMap)
		stream << cameraDataEntry.first << " ";
	stream << std::endl;
	stream << "\t" << (krt.cameraParametersMap.size() - this->cameraDataMap.size()) << " not registered cameras." << std::endl;
	stream << "\t";
	for (const auto& cameraParametersEntry : krt.cameraParametersMap)
		if (!this->cameraDataMap.contains(cameraParametersEntry.first))
			stream << cameraParametersEntry.first << " ";
	stream << std::endl;
	// Print reprojection distances.
	double sumReprojectionDistance = 0.0;
	double maxReprojectionDistance = -1.0;
	for (const auto& cameraDataEntry : this->cameraDataMap) {
		std::uint64_t cameraName = cameraDataEntry.first;
		const CameraData& cameraData = cameraDataEntry.second;
		for (const Keypoint& keypoint : cameraData.keypoints) {
			if (keypoint.point3DID == -1) continue;
			// Rotate and translate the point.
			Eigen::Vector3d p;
			ceres::AngleAxisRotatePoint(cameraData.rotation.data(), this->point3Ds[keypoint.point3DID].position.data(), p.data());
			p += cameraData.translation;
			// Perspective projection.
			Eigen::Vector2d projection(
				cameraData.intrinsics(0) * p(0) / p(2) + cameraData.intrinsics(2),
				cameraData.intrinsics(1) * p(1) / p(2) + cameraData.intrinsics(3)
			);
			// Compute error.
			double reprojectionDistance = (projection - keypoint.pixelPos).norm();
			sumReprojectionDistance += reprojectionDistance;
			maxReprojectionDistance = std::max(maxReprojectionDistance, reprojectionDistance);
		}
	}
	double meanReprojectionDistance = sumReprojectionDistance / static_cast<double>(this->numValidKeypoints);
	stream << "\tMean reprojection distance: " << meanReprojectionDistance << std::endl;
	stream << "\tSum reprojection distance: " << sumReprojectionDistance << std::endl;
	stream << "\tMax reprojection distance: " << maxReprojectionDistance << std::endl;
	return stream.str();
}