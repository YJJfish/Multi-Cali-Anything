#include "KRT.hpp"
#include <fstream>
#include <sstream>
#include <iostream>

bool KRT::load(const std::filesystem::path& path) {
	if (!std::filesystem::is_regular_file(path)) {
		std::cerr << "[KRT] The KRT file \"" << path << "\" does not exist." << std::endl;
		return false;
	}
	this->cameraParametersMap.clear();
	std::ifstream fin(path, std::ios::in);
	while (true) {
		std::string line;
		std::stringstream stream;
		double temp{};
		std::uint64_t cameraName{};
		Eigen::Vector4d intrinsics{};
		Eigen::Matrix3d rotationMat{}; // Rotation matrix.
		Eigen::AngleAxisd rotationAngleAxis{}; // Rotation in axis-angle representation, stored as Eigen::AngleAxisd.
		Eigen::Vector3d rotation{}; // Rotation in axis-angle representation, stored as Eigen::Vector3d.
		Eigen::Vector3d translation{};

		// Camera name.
		std::getline(fin, line);
		if (line.empty()) break;
		stream << line;
		stream >> cameraName;
		line.clear(); stream.clear();

		// fx, 0.0, cx.
		std::getline(fin, line);
		if (line.empty()) break;
		stream << line;
		stream >> intrinsics(0) >> temp >> intrinsics(2);
		line.clear(); stream.clear();

		// 0.0, fy, cy.
		std::getline(fin, line);
		if (line.empty()) break;
		stream << line;
		stream >> temp >> intrinsics(1) >> intrinsics(3);
		line.clear(); stream.clear();

		// 0.0, 0.0, 1.0.
		std::getline(fin, line);
		if (line.empty()) break;

		// 0.0, 0.0, 0.0, 0.0, 0.0.
		std::getline(fin, line);
		if (line.empty()) break;

		// r00, r01, r02, t0.
		std::getline(fin, line);
		if (line.empty()) break;
		stream << line;
		stream >> rotationMat(0, 0) >> rotationMat(0, 1) >> rotationMat(0, 2) >> translation(0);
		line.clear(); stream.clear();

		// r10, r11, r12, t1.
		std::getline(fin, line);
		if (line.empty()) break;
		stream << line;
		stream >> rotationMat(1, 0) >> rotationMat(1, 1) >> rotationMat(1, 2) >> translation(1);
		line.clear(); stream.clear();

		// r20, r21, r22, t2.
		std::getline(fin, line);
		if (line.empty()) break;
		stream << line;
		stream >> rotationMat(2, 0) >> rotationMat(2, 1) >> rotationMat(2, 2) >> translation(2);
		line.clear(); stream.clear();

		// Convert the rotation matrix to axis-angle representation.
		rotationAngleAxis = Eigen::AngleAxisd(rotationMat);
		rotation = rotationAngleAxis.angle() * rotationAngleAxis.axis();

		// Empty line.
		std::getline(fin, line);

		// Store the data.
		this->cameraParametersMap.insert(std::make_pair(
			cameraName,
			CameraParameters{
				.rotation = rotation,
				.translation = translation,
				.intrinsics = intrinsics
			}
		));
	}
	return true;
}

std::string KRT::summary(void) const {
	std::stringstream stream;
	stream << "[KRT] Summary:" << std::endl;
	stream << "\tNumber of cameras: " << this->cameraParametersMap.size() << std::endl;
	for (const auto& cameraParametersEntry : this->cameraParametersMap) {
		stream << "\tCamera " << cameraParametersEntry.first << ":" << std::endl;
		stream << "\tRotation: " << cameraParametersEntry.second.rotation.transpose() << std::endl;
		stream << "\tTranslation: " << cameraParametersEntry.second.translation.transpose() << std::endl;
		stream << "\tIntrinsics: " << cameraParametersEntry.second.intrinsics.transpose() << std::endl;
	}
	return stream.str();
}